# -*- coding: utf-8 -*-
#
# This file is part of CERN Document Server.
# Copyright (C) 2016, 2017 CERN.
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

import json
from collections import OrderedDict
from itertools import chain

import requests
from flask import current_app

from .error import InvalidAspectRatioError, InvalidResolutionError, \
    SorensonError, TooHighResolutionError
from .proxies import current_cds_sorenson
from .utils import _filepath_for_samba, generate_json_for_encoding, get_status


def start_encoding(input_file, output_file, desired_quality,
                   display_aspect_ratio, max_height=None, max_width=None,
                   **kwargs):
    """Encode a video that is already in the input folder.

    :param input_file: string with the filename, something like
        /eos/cds/test/sorenson/8f/m2/728-jsod98-8s9df2-89fg-lksdjf/data where
        the last part "data" is the filename and the last directory is the
        bucket id.
    :param output_file: the file to output the transcoded file.
    :param desired_quality: desired quality to transcode to.
    :param display_aspect_ratio: the video's aspect ratio
    :param max_height: maximum height we want to encode
    :param max_width: maximum width we want to encode
    :param kwargs: other technical metadata
    :returns: job ID.
    """
    input_file = _filepath_for_samba(input_file)
    output_file = _filepath_for_samba(output_file)

    aspect_ratio, preset_config = _get_quality_preset(desired_quality,
                                                      display_aspect_ratio,
                                                      video_height=max_height,
                                                      video_width=max_width)

    current_app.logger.debug(
        'Transcoding {0} to quality {1} and aspect ratio {2}'.format(
            input_file, desired_quality, aspect_ratio))

    # Build the request of the encoding job
    json_params = generate_json_for_encoding(input_file, output_file,
                                             preset_config['preset_id'])
    proxies = current_app.config['CDS_SORENSON_PROXIES']
    headers = {'Accept': 'application/json'}
    response = requests.post(current_app.config['CDS_SORENSON_SUBMIT_URL'],
                             headers=headers, json=json_params,
                             proxies=proxies)

    data = json.loads(response.text)

    if response.status_code == requests.codes.ok:
        job_id = data.get('JobId')
        return job_id, aspect_ratio, preset_config
    else:
        # something is wrong - sorenson server is not responding or the
        # configuration is wrong and we can't contact sorenson server
        raise SorensonError("{0}: {1}".format(response.status_code,
                                              response.text))


def stop_encoding(job_id):
    """Stop encoding job.

    :param job_id: string with the job ID.
    :returns: None.
    """
    delete_url = (current_app.config['CDS_SORENSON_DELETE_URL']
                             .format(job_id=job_id))
    headers = {'Accept': 'application/json'}
    proxies = current_app.config['CDS_SORENSON_PROXIES']

    response = requests.delete(delete_url, headers=headers, proxies=proxies)
    if response.status_code != requests.codes.ok:
        raise SorensonError("{0}: {1}".format(response.status_code,
                                              response.text))


def get_encoding_status(job_id):
    """Get status of a given job from the Sorenson server.

    If the job can't be found in the current queue, it's probably done, so we
    check the archival queue.

    :param job_id: string with the job ID.
    :returns: tuple with the status message and progress in %.
    """
    status = get_status(job_id)
    if status == '':
        # encoding job was canceled
        return "Canceled", 100
    status_json = json.loads(status)
    # there are different ways to get the status of a job, depending if
    # the job was successful, so we should check for the status code in
    # different places
    job_status = status_json.get('Status', {}).get('Status')
    job_progress = status_json.get('Status', {}).get('Progress')
    if job_status:
        return current_app.config['CDS_SORENSON_STATUSES'].get(job_status), \
            job_progress
    # status not found? check in different place
    job_status = status_json.get('StatusStateId')
    if job_status:
        # job is probably either finished or failed, so the progress will
        # always be 100% in this case
        return current_app.config['CDS_SORENSON_STATUSES'].get(job_status), 100
    # No status was found (which shouldn't happen)
    raise SorensonError('No status found for job: {0}'.format(job_id))


def restart_encoding(job_id, input_file, output_file, desired_quality,
                     display_aspect_ratio, **kwargs):
    """Try to stop the encoding job and start a new one.

    It's impossible to get the input_file and preset_quality from the
    job_id, if the job has not yet finished, so we need to specify all
    parameters for stopping and starting the encoding job.
    """
    try:
        stop_encoding(job_id)
    except SorensonError:
        # If we failed to stop the encoding job, ignore it - in the worst
        # case the encoding will finish and we will overwrite the file.
        pass
    return start_encoding(input_file, output_file, desired_quality,
                          display_aspect_ratio, **kwargs)


def _get_available_aspect_ratios(pairs=False):
    """Return all available aspect ratios.

    :param pairs: if True, will return aspect ratios as pairs of integers
    """
    ratios = [key for key in current_app.config['CDS_SORENSON_PRESETS']]
    if pairs:
        ratios = [tuple(map(int, ratio.split(':', 1))) for ratio in ratios]
    return ratios


def get_all_distinct_qualities():
    """Return all distinct available qualities, independently of presets.

    :returns all the qualities without duplications. For example, if presets A
    has [240p, 360p, 480p] and presets B has [240p, 480p], the result will be
    [240p, 360p, 480p].
    """
    # get all possible qualities
    all_qualities = [
        outer_dict.keys()
        for outer_dict in current_app.config['CDS_SORENSON_PRESETS'].values()
    ]
    # remove duplicates while preserving ordering
    all_distinct_qualities = OrderedDict.fromkeys(chain(*all_qualities))
    return list(all_distinct_qualities)


def can_be_transcoded(subformat_desired_quality, video_aspect_ratio,
                      video_width=None, video_height=None):
    """Return the details of the subformat that will be generated.

    :param subformat_desired_quality: the quality desired for the subformat
    :param video_aspect_ratio: the original video aspect ratio
    :param video_width: the original video width
    :param video_height: the original video height
    :returns a dict with aspect ratio, width and height if the subformat can
    be generated, or False otherwise
    """
    try:
        ar, conf = _get_quality_preset(subformat_desired_quality,
                                       video_aspect_ratio,
                                       video_height=video_height,
                                       video_width=video_width)
        return dict(quality=subformat_desired_quality, aspect_ratio=ar,
                    width=conf['width'], height=conf['height'])
    except (InvalidAspectRatioError, InvalidResolutionError,
            TooHighResolutionError) as _:
        return None


def _get_closest_aspect_ratio(width, height):
    """Return the closest configured aspect ratio to the given height/width.

    :param height: video height
    :param width: video width
    """
    # calculate the aspect ratio fraction
    unknown_ar_fraction = float(width) / height

    # find the closest aspect ratio fraction to the unknown
    closest_fraction = min(current_cds_sorenson.aspect_ratio_fractions.keys(),
                           key=lambda x: abs(x - unknown_ar_fraction))
    return current_cds_sorenson.aspect_ratio_fractions[closest_fraction]


def _get_quality_preset(subformat_desired_quality, video_aspect_ratio,
                        video_height=None, video_width=None):
    """Return the transcoding config for a given aspect ratio and quality.

    :param subformat_desired_quality: the desired quality for transcoding
    :param video_aspect_ratio: the video's aspect ratio
    :param video_height: maximum output height for transcoded video
    :param video_width: maximum output width for transcoded video
    :returns the transcoding config for a given inputs
    """
    try:
        ar_presets = current_app.config['CDS_SORENSON_PRESETS'][
            video_aspect_ratio]
    except KeyError:
        if video_height and video_width:
            video_aspect_ratio = _get_closest_aspect_ratio(video_height,
                                                           video_width)
            ar_presets = current_app.config['CDS_SORENSON_PRESETS'][
                video_aspect_ratio]
        else:
            raise InvalidAspectRatioError(video_aspect_ratio)

    try:
        preset_config = ar_presets[subformat_desired_quality]
    except KeyError:
        raise InvalidResolutionError(video_aspect_ratio,
                                     subformat_desired_quality)

    if (video_height and video_height < preset_config['height']) or \
            (video_width and video_width < preset_config['width']):
        raise TooHighResolutionError(video_aspect_ratio, video_height,
                                     video_width, preset_config['height'],
                                     preset_config['width'])

    return video_aspect_ratio, preset_config
