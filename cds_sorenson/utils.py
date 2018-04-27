# -*- coding: utf-8 -*-
#
# This file is part of CERN Document Server.
# Copyright (C) 2016, 2017, 2018 CERN.
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
"""API to use Sorenson transcoding server."""

from __future__ import absolute_import, print_function

import logging
import os
import os.path
import shutil
import tempfile
from contextlib import contextmanager
from itertools import chain
from subprocess import STDOUT, CalledProcessError, check_output
from time import sleep

import requests
from flask import current_app
from invenio_files_rest.models import ObjectVersion

from .error import FFmpegExecutionError, SorensonError


def generate_json_for_encoding(input_file,
                               output_file,
                               preset_id,
                               sorenson_queue=None):
    """Generate JSON that will be sent to Sorenson server to start encoding."""
    current_preset = _get_preset_config(preset_id)
    # Make sure the preset config exists for a given preset_id
    if not current_preset:
        raise SorensonError('Invalid preset "{0}"'.format(preset_id))

    return dict(
        Name='CDS File:{0} Preset:{1}'.format(input_file, preset_id),
        QueueId=sorenson_queue or
        current_app.config['CDS_SORENSON_DEFAULT_QUEUE'],
        JobMediaInfo=dict(
            SourceMediaList=[
                dict(
                    FileUri=input_file,
                    UserName=current_app.config['CDS_SORENSON_USERNAME'],
                    Password=current_app.config['CDS_SORENSON_PASSWORD'],
                )
            ],
            DestinationList=[dict(FileUri='{}'.format(output_file))],
            CompressionPresetList=[dict(PresetId=preset_id)],
        ),
    )


def get_status(job_id):
    """For a given job id, returns the status as JSON string.

    If the job can't be found in the current queue, it's probably done, so we
    check the archival queue. Raises an exception if there the response has a
    different code than 200.

    :param job_id: string with the job ID.
    :returns: JSON with the status or empty string if the job was not found.
    """
    current_jobs_url = (
        current_app.config['CDS_SORENSON_CURRENT_JOBS_STATUS_URL']
        .format(job_id=job_id))
    archive_jobs_url = (
        current_app.config['CDS_SORENSON_ARCHIVE_JOBS_STATUS_URL']
        .format(job_id=job_id))

    headers = {'Accept': 'application/json'}
    proxies = current_app.config['CDS_SORENSON_PROXIES']

    response = requests.get(current_jobs_url, headers=headers, proxies=proxies)

    if response.status_code == 404:
        response = requests.get(
            archive_jobs_url, headers=headers, proxies=proxies)

    if response.status_code == requests.codes.ok:
        return response.text
    else:
        raise SorensonError("{0}: {1}".format(response.status_code,
                                              response.text))


def _get_preset_config(preset_id):
    """Return preset config based on the preset_id."""
    for outer_dict in current_app.config['CDS_SORENSON_PRESETS'].values():
        for inner_dict in outer_dict.values():
            if inner_dict['preset_id'] == preset_id:
                return inner_dict


def filepath_for_samba(obj):
    """Adjust file path for Samba protocol.

    Sorenson has the eos directory mounted through samba, so the paths
    need to be adjusted.
    """
    # Compatibility trick for legacy API
    filepath = obj.file.uri if isinstance(obj, ObjectVersion) else obj
    samba_dir = current_app.config['CDS_SORENSON_SAMBA_DIRECTORY']
    eos_dir = current_app.config['CDS_SORENSON_CDS_DIRECTORY']
    return filepath.replace(eos_dir, samba_dir)


def run_ffmpeg_command(cmd, obj, **kwargs):
    """Run ffmpeg command and capture errors."""
    kwargs.setdefault('stderr', STDOUT)
    try:
        return check_output(cmd.split(), **kwargs)
    except CalledProcessError as e:
        raise FFmpegExecutionError(e)


def cleanup_after_failure(*args, **kwargs):
    """."""
    # TODO


def eos_fuse_fail_safe(f):
    """Try to run on FUSE and if not bring the file home.

    Assumptions:
    - The method will use `input_file` as key in the kwargs to get the real
      path to the file.
    """
    def wrapper(self, *args, **kwargs):
        if 'input_file' in kwargs:
            # Don't care, you know what you are doing
            return f(self, *args, **kwargs)

        # Try first to use EOS FUSE
        kwargs['input_file'] = self.video.file.uri.replace(
            current_app.config['VIDEOS_XROOTD_ENDPOINT'], '')
        try:
            # sometimes FUSE is slow to update, retry a couple of times
            sleep(2)
            return f(self, *args, **kwargs)
        except Exception as e:
            logging.error(
                '#EOS_FUSE_ERROR: file not accesible via FUSE {0}'.format(
                    self.video))

        # Surprise fuse didn't work! Copy the file to tmp
        logging.info('Copying file to local file system')
        temp_folder = tempfile.mkdtemp()
        temp_location = os.path.join(temp_folder, 'data')
        with open(temp_location, 'wb') as dst:
            shutil.copyfileobj(get_pyfs(obj.file.uri).open(), dst)
        kwargs['input_file'] = temp_location
        try:
            result = f(self, *args, *kwargs)
        finally:
            shutil.rmtree(temp_folder)
        return result


def get_pyfs(path, *args, **kwargs):
    """."""
    if path.startswith(('root://', 'roots://')):
        from invenio_xrootd.storage import EOSFileStorage
        return EOSFileStorage(path)
    else:
        from invenio_files_rest.storage.pyfs import PyFSFileStorage
        return PyFSFileStorage(path)
