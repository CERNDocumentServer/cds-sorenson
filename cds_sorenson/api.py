# -*- coding: utf-8 -*-
#
# This file is part of CERN Document Server.
# Copyright (C) 2018 CERN.
#
# Invenio is free software; you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
#
# Invenio is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Invenio; if not, write to the
# Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
# MA 02111-1307, USA.
#
# In applying this license, CERN does not
# waive the privileges and immunities granted to it by virtue of its status
# as an Intergovernmental Organization or submit itself to any jurisdiction.
"""Invenio-Videos API implementation for use Sorenson and FFMPG."""

import json
import logging
import os
import tempfile
import time
from enum import Enum
from random import randint

import requests
from flask import current_app
from flask_iiif.utils import create_gif_from_frames
from invenio_db import db
from invenio_files_rest.models import FileInstance, ObjectVersion, \
    ObjectVersionTag, as_object_version

from .error import SorensonError
from .legacy_api import can_be_transcoded, get_all_distinct_qualities, \
    get_encoding_status, restart_encoding, start_encoding, stop_encoding
from .utils import cleanup_after_failure, eos_fuse_fail_safe, \
    filepath_for_samba, run_ffmpeg_command


class VideoMixing(object):
    """Mixing for video processing."""

    def __init__(self, video_object_version, *args, **kwargs):
        """Initialize Video object.

        :param video_object_version: Original video Invenio ObjectVersion,
        either the object or the UUID.
        :param kwargs:
          - extracted_metadata:
          - presets:
        """
        self.video = as_object_version(video_object_version)
        self._extracted_metadta = kwargs.get('extracted_metadta', {})
        self._presets = kwargs.get('presets', [])

    @property
    def presets(self):
        """Get the details of all the presets that are available for the video.

        :return: Dictionary with the details of each of the reset grouped by
        preset name. The content of dictionary depends on configuration of each
        video class content, your consumer app should be aware of that..
        """
        raise NotImplemented()

    def extract_metadata(self,
                         process_output_callback=None,
                         attach_to_video=False,
                         *args,
                         **kwargs):
        """Extract all the video metadata from the output of ffprobe.

        :param process_output_callback: function to process the ffprobe output,
        takes a dictionary and returns a dictionary.
        :param attach_to_video: If set to True the extracted metadata will be
        attached to the video `ObjectVersion` as `ObjectVersionTag` after
        running the `process_output_callback`. The `ObjectVersionTag` storage
        is key value base, where the value can be stringify into the database,
        keep this in mind when setting this value to true.
        :return: Dictionary with the extracted metadata.
        """
        raise NotImplemented()

    def create_thumbnails(self,
                          start=5,
                          end=95,
                          step=10,
                          progress_callback=None,
                          *args,
                          **kwargs):
        """Create thumbnail files.

        :param start: percentage to start extracting frames. Default 5.
        :param end: percentage to finish extracting frames. Default 95.
        :param step: percentage between frames. Default 10.
        :param progress_callback: function to report progress, takes an integer
        showing percentage of progress, a string with a message and a
        dictionary with more information.
        :return: List of `ObjectVersion` with the thumbnails.
        """
        raise NotImplemented()

    def create_gif_from_frames(self):
        """Create a gif file with the extracted frames.

        `create_thumbnail` needs to be run first, if there are not frames it
        will raise.
        """
        raise NotImplemented()

    def encode(self, preset_quality, callback_function=None, *args, **kwargs):
        """Encode the video using a preset.

        :param preset_quality: Name of the quality to encode the video with.
        If the preset does not apply to the current video raises.
        :param callback_function: function to report progress, takes and
        integer showing the percentage of the progress, a string with a message
        and a dictionary with mote information.
        :return: Dictionary containing the information of the encoding job.
        """
        raise NotImplemented()

    @staticmethod
    def stop_encoding_job(job_id, *ars, **kwargs):
        """Stop encoding.

        :param job_id: ID of the job to stop.
        """
        raise NotImplemented()

    @staticmethod
    def get_job_status(job_id):
        """Get status of a give job.

        :param job_id: ID of the job to track the status.
        :return: Tuple with an integer showing the percentage of
        the process, a string with a message and a dictionary with more
        information (perhaps `None`)
        """
        raise NotImplemented()


class SorensonStatus(Enum):
    """Sorenson status mapping."""

    PENDING = 'PENDING'
    """Not yet running."""

    STARTED = 'STARTED'
    """Running."""

    SUCCESS = 'SUCCESS'
    """Done."""

    FAILURE = 'FAILURE'
    """Error."""

    REVOKED = 'REVOKED'
    """Canceled."""

    @staticmethod
    def to_sorenson_status(response_status):
        """Convert Sorenson's status into something meaningful."""
        status_map = {
            0: (SorensonStatus.PENDING, 0),  # Undefined
            1: (SorensonStatus.PENDING, 0),  # Waiting
            2: (SorensonStatus.STARTED, 0.33),  # Downloading
            3: (SorensonStatus.STARTED, 0.66),  # Transcoding
            4: (SorensonStatus.STARTED, 0.99),  # Uploading
            5: (SorensonStatus.SUCCESS, 1),  # Finished
            6: (SorensonStatus.FAILURE, 0),  # Error
            7: (SorensonStatus.REVOKED, 0),  # Canceled
            8: (SorensonStatus.FAILURE, 0),  # Deleted
            9: (SorensonStatus.PENDING, 0),  # Hold
            10: (SorensonStatus.FAILURE, 0),  # Incomplete
        }
        return status_map[response_status]


class CDSVideo(VideoMixing):
    """Soreson/FFMPG Video implementation."""

    @property
    def duration(self):
        """Video duration in seconds (float) from extracted metadata."""
        return self.extracted_metadata['duration']

    @property
    def aspect_ratio(self):
        """Video aspect ratio from extracted metadata."""
        return self.extracted_metadata['display_aspect_ratio']

    @property
    def height(self):
        """Video height from extracted metadata."""
        return self.extracted_metadata['height']

    @property
    def width(self):
        """Video width from extracted metadata."""
        return self.extracted_metadata['width']

    @property
    def thumbnails(self):
        """List with all the video thumbnail if created."""
        pass

    @property
    def _sorenson_aspect_ratio(self):
        """Closest aspect ratio inside Soreonson preset configuration."""
        fractions_with_ar = {}
        for ar in current_app.config['CDS_SORENSON_PRESETS']:
            sorenson_w, sorenson_h = ar.split(':')
            sorenson_ar_fraction = float(sorenson_w) / float(sorenson_h)
            fractions_with_ar.setdefault(sorenson_ar_fraction, ar)
        # calculate the aspect ratio fraction
        unknown_ar_fraction = float(self.width) / self.height
        closest_fraction = min(
            fractions_with_ar.keys(),
            key=lambda x: abs(x - unknown_ar_fraction))
        return fractions_with_ar[closest_fraction]

    @property
    def _sorenson_height(self):
        """Video height or minimum height from Sorenson presets if smaller."""
        minimun_height = None
        for name, info in current_app.config['CDS_SORENSON_PRESETS'][
                self._soreson_aspect_ratio]:
            if not minimun_height or minimun_height > info['height']:
                minimun_height = info['height']

        return self.height if self.heigh > minimun_height else minimun_height

    @property
    def _sorenson_width(self):
        """Video width or minimum width from Sorenson presets if smaller."""
        minimun_width = None
        for name, info in current_app.config['CDS_SORENSON_PRESETS'][
                self._soreson_aspect_ratio]:
            if not minimun_width or minimun_width > info['width']:
                minimun_width = info['width']

        return self.width if self.heigh > minimun_width else minimun_width

    @property
    def extracted_metadata(self):
        """Get video metadata.

        If the metadata is not cached in the object it will call
        `extract_metadata` with `attach_to_video=False` parameter.
        """
        # TODO: perhaps we could read from DB the tags?
        if not self._extracted_metadta:
            self._extracted_metadta = self.extract_metadata(
                attach_to_video=False)
        return self._extracted_metadta

    @property
    def presets(self):
        """Return all the presets available for the current video.

        :return: List of dictionaries
        """
        if self._presets:
            return self._presets

        preset_config = current_app.config['CDS_SORENSON_PRESETS']
        all_presets = preset_config[self._sorenson_aspect_ratio]

        # Filter presets base on width and height of the video
        self._presets = dict()
        for name, preset_info in all_presets.iteritems():
            if self._sorenson_height < preset_info['hight'] or \
                   self._sorenson_width < preset_info['width']:
                # Preset to big for the video file
                continue
            # Add name inside for convenience, I am lazy!
            preset_info['name'] = name
            self._presets[name] = preset_info

        return self._presets

    def _sorenson_queue(self, preset_quality):
        """Given file size and preset quality decide which queue to use."""
        sorenson_queues = current_app.config['CDS_SORENSON_QUEUES']
        size_threshold = current_app.config['CDS_SORENSON_BIG_FILE_THRESHOLD']
        flast_preset = current_app.config['CDS_SORENSON_FAST_PRESET']

        if preset_quality == fast_preset:
            return sorenson_queues['fast']

        if self.video.file.size > size_threshold:
            return sorenson_queues['big_files']
        else:
            return sorenson_queues['default']

    def _build_subformat_key(preset_info):
        """Return the key for a subformat based on the preset_info."""
        return '{0}.mp4'.format(preset_info['name'])

    @staticmethod
    def _clean_file_name(uri):
        """Remove the .mp4 file extension from file name.

        For some reason the Sorenson Server adds the extension to the output
        file, creating ``data.mp4``. Our file storage does not use extensions
        and this is causing troubles.
        The best/dirtiest solution is to remove the file extension once the
        transcoded file is created.
        """
        real_path = '{0}.mp4'.format(uri)
        # Don't judge me for this :)
        fs = get_pyfs(real_path)._get_fs(False)[0]
        fs.move(real_path, uri)

    def generate_request_body(self, input_file, output_file, preset_info):
        """Generate JSON to be sent to Sorenson server to start encoding."""
        return dict(
            Name='CDS File:{0} Preset:{1}'.format(self.video.version_id,
                                                  preset_info['name']),
            QueueId=self._sorenson_queue(preset_info['name']),
            JobMediaInfo=dict(
                SourceMediaList=[
                    dict(
                        FileUri=input_file,
                        UserName=current_app.config['CDS_SORENSON_USERNAME'],
                        Password=current_app.config['CDS_SORENSON_PASSWORD'],
                    )
                ],
                DestinationList=[dict(FileUri=output_file)],
                CompressionPresetList=[
                    dict(PresetId=preset_info['preset_id'])
                ],
            ),
        )

    @eos_fuse_fail_safe
    def extract_metadata(self, attach_to_video=True, *args, **kwargs):
        """Use FFMPG to extract all video metadata."""
        cmd = ('ffprobe -v quiet -show_format -print_format json '
               '-show_streams -select_streams v:0 {input_file}'.format(kwargs))
        self._extracted_metadata = run_ffmpeg_command(cmd).decode('utf-8')
        if not self._extracted_metadata:
            # TODO: perhaps we want to try a different command, i.e. avi files
            raise RuntimeError('No metadata extracted for {0}'.format(
                self.video))

        process_output_callback = kwargs.get['process_output_callback'] or \
            default_extract_metadata_callback
        self._extracted_metadata = process_output_callback(
            self._extracted_metadata)

        if attach_to_video:
            for key, value in self._extracted_metadata.iteritems():
                ObjectVersionTag.create_or_update(self.video, key, value)

        return self._extracted_metadata

    @eos_fuse_fail_safe
    def create_thumbnails(self,
                          start=5,
                          end=95,
                          step=10,
                          progress_callback=None,
                          create_gif=True,
                          *args,
                          **kwargs):
        """Use FFMPEG to create thumbnail files."""
        duration = float(self.duration)
        step_time = duration * step / 100
        start_time = duration * start / 100
        end_time = (duration * end / 100) + 0.01  # FIXME WDF?

        number_of_thumbnails = ((end - start) / step) + 1

        assert all([
            0 < start_time < duration,
            0 < end_time < duration,
            0 < step_time < duration,
            start_time < end_time,
            (end_time - start_time) % step_time < 0.05  # FIXME WDF?
        ])

        thumbnail_name = current_app.config.get(
            'VIDEO_THUMBNAIL_NAME_TEMPLATE', 'frame-{0:d}.jpg')
        # Iterate over requested timestamps
        objs = []
        for i, timestamp in enumerate(range(start_time, end_time, step_time)):
            with tempfile.TemporaryDirectory() as o:
                # TODO: can we write for the final location like encode?
                output_file = os.path.join(o, thumbnail_name)
                # Construct ffmpeg command
                cmd = ('ffmpeg -accurate_seek -ss {timestamp} -i {input_file}'
                       ' -vframes 1 {output_file} -qscale:v 1').format(
                           timestamp=timestamp,
                           output_file=output_file,
                           **kwargs)

                # Run ffmpeg command
                run_ffmpeg_command(cmd)

                # Create ObjectVersion out of the tmp file
                with db.session.being_nested(), open(output_file) as f:
                    obj = ObjectVersion.create(
                        bucket=self.video.bucket,
                        key=thumbnail_name,
                        stream=f,
                        size=os.path.getsize(output_file))
                    ObjectVersionTag.create(obj, 'master', str(
                        self.video.version_id))
                    ObjectVersionTag.create(obj, 'media_type', 'image')
                    ObjectVersionTag.create(obj, 'context_type', 'thumbnail')
                    ObjectVersionTag.create(obj, 'content_type', 'jpg')
                    ObjectVersionTag.create(obj, 'timestamp',
                                            start_time + (i + 1) * step_time)
                    objs.append(obj)
                db.session.commit()

            # Report progress
            if progress_callback:
                progress_callback(number_of_thumbnails / i + 1)

        if create_gif:
            objs.append(self.create_gif(
                progress_callback=progres_callback, *args, **kwargs))
        return objs

    def create_gif(self, progress_callback=None, *args, **kwargs):
        """Use IIIF to create a GIF from the extracted frames."""
        images = []
        for frame in self.thumbnails:
            image = Image.open(file_opener_xrootd(f, 'rb'))
            # Convert image for better quality
            im = image.convert('RGB').convert(
                'P', palette=Image.ADAPTIVE, colors=255
            )
            images.append(im)

        if not images:
            # Most likely there are no thumbnails
            raise RuntimeError(
                'Before creating a gif you need to extract the thumbnails!')

        gif_image = create_gif_from_frames(images)

        gif_name = current_app.config.get('VIDEO_GIF_NAME', 'frames.gif')
        with db.session.begin_nested(), tempfile.TemporaryDirectory() as o:
            output_file = os.path.join(o, gif_name)
            gif_image.save(output_file, save_all=True)
            obj = ObjectVersion.create(
                bucket=self.video.bucket,
                key=git_name,
                stream=open(output_file),
                size=os.path.getsize(output_file))
            ObjectVersionTag.create(obj, 'master', str(
                self.video.version_id))
            ObjectVersionTag.create(obj, 'media_type', 'image')
            ObjectVersionTag.create(obj, 'context_type', 'preview')
            ObjectVersionTag.create(obj, 'content_type', 'gif')
        db.session.commit()

        return obj

    def encode(self, preset_quality, callback_function=None, *args, **kwargs):
        """Enconde a video."""
        preset_info = self.presets.get(preset_quality)
        assert preset_info, 'Unknown preset'

        with db.session.begin_nested():
            # Create FileInstance
            file_instance = FileInstance.create()

            # Create ObjectVersion
            obj_key = self._build_subformat_key(preset_info)
            obj = ObjectVersion.create(bucket=self.video.bucket, key=obj_key)

            # Extract new location
            bucket_location = self.video.bucket.location
            storage = file_instance.storage(default_location=bucket_location)
            directory, filename = storage._get_fs()

            # XRootDPyFS doesn't implement root_path
            try:
                # XRootD Safe
                output_file = os.path.join(directory.root_url,
                                           directory.base_path, filename)
            except AttributeError:
                output_file = os.path.join(directory.root_path, filename)

            input_file = filepath_for_samba(self.video)

            # Build the request of the encoding job
            json_params = self.generate_request_body(input_file, output_file,
                                                     preset_info)
            proxies = current_app.config['CDS_SORENSON_PROXIES']
            headers = {'Accept': 'application/json'}
            logging.debug('Sending job to Sorenson {0}'.format(json_params))
            response = requests.post(
                current_app.config['CDS_SORENSON_SUBMIT_URL'],
                headers=headers,
                json=json_params,
                proxies=proxies)

            if response.status_code != requests.codes.ok:
                # something is wrong - sorenson server is not responding or the
                # configuration is wrong and we can't contact sorenson server
                cleanup_after_failure(file_uri=output_file)
                db.session.rollback()
                raise SorensonError("{0}: {1}".format(response.status_code,
                                                      response.text))
            data = json.loads(response.text)
            logging.debug('Encoding Sorenson response {0}'.format(data))
            job_id = data.get('JobId')

        db.session.commit()

        # Continue here until the job is done
        status = SorensonStatus.PENDING
        with status != SorensonStatus.SUCCESS:
            status, percentage, info = CDSVideo.get_job_status(job_id)

            if callback_function:
                callback_function(status, percentage, info)

            if status == SorensonStatus.FAILURE:
                cleanup_after_failure(
                    object_version=obj, file_instance=file_instance)
                raise RuntimeError('Error encoding: {0} {1} {2}'.format(
                    status, precentage, info))
            elif status == SorensonStatus.REVOKED:
                cleanup_after_failure(
                    object_version=obj, file_instance=file_instance)
                return None
            # FIXME: better way to put this?
            time.sleep(randint(1, 10))

        # Set file's location, if job has completed
        self._clean_file_name(output_file)

        with db.session.begin_nested():
            fs = get_fs(output_file)
            checksum = fs.checksum()
            with fs.open() as f:
                try:
                    size = f.size
                except AttributeError:
                    # PyFileSystem returns a BufferedReader with no size
                    size = os.fstat(f.fileno()).st_size
            file_instance.set_uri(output_file, size, checksum)
            obj.set_file(file_instance)

        db.sesssion.commit()

        return {'job_id': job_id, 'preset': preset_info, 'object': obj}

    @staticmethod
    def get_job_status(job_id):
        """Get status of a given hob from Sorenson server.

        If the job can't be found in the current queue, it's probably done,
        so we check the archival queue.
        """
        current_jobs_url = (
            current_app.config['CDS_SORENSON_CURRENT_JOBS_STATUS_URL']
            .format(job_id=job_id))
        archive_jobs_url = (
            current_app.config['CDS_SORENSON_ARCHIVE_JOBS_STATUS_URL']
            .format(job_id=job_id))

        headers = {'Accept': 'application/json'}
        proxies = current_app.config['CDS_SORENSON_PROXIES']
        response = requests.get(
            current_jobs_url, headers=headers, proxies=proxies)

        if response.status_code == 404:
            # Check the archive URL
            response = requests.get(
                archive_jobs_url, headers=headers, proxies=proxies)

        if response.status_code != requests.codes.ok:
            # TODO Probably there is a better way to do this, retry?
            status_json = json.load(response.text) if response.text else {}
            return SorensonStatus.FAILURE, 0, status_json

        if response.text == '':
            return SorensonStatus.REVOKED, 0, {}

        status_json = json.loads(status)
        # there are different ways to get the status of a job, depending if
        # the job was successful, so we should check for the status code in
        # different places
        job_status = status_json.get('Status', {}).get('Status')
        job_progress = status_json.get('Status', {}).get('Progress') or 0

        if not job_status:
            # status not found? check in different place
            job_status = status_json.get('StatusStateId')

        status, p_factor = SorensonStatus.to_sorenson_status(job_status)

        return status, job_progress * p_factor, status_json

    @staticmethod
    def stop_encoding_job(job_id):
        """Stop encoding job."""
        delete_url = (current_app.config['CDS_SORENSON_DELETE_URL']
                      .format(job_id=job_id))
        headers = {'Accept': 'application/json'}
        proxies = current_app.config['CDS_SORENSON_PROXIES']
        response = requests.delete(
            delete_url, headers=headers, proxies=proxies)

        if response.status_code != requests.codes.ok:
            raise SorensonError("{0}: {1}".format(response.status_code,
                                                  response.text))
        return job_id


def default_extract_metadata_callback(extracted_metadata):
    """."""
    # TODO: decide which fields we need, do we just flatten the dict?
    return extracted_metadata


__all__ = (
    'CDSVideo',
    'start_encoding',
    'get_all_distinct_qualities',
    'get_encoding_status',
    'restart_encoding',
    'start_encoding',
    'stop_encoding',
    'can_be_transcoded',
)
