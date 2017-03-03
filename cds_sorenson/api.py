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
    SorensonError
from .utils import _filepath_for_samba, generate_json_for_encoding, get_status


def start_encoding(input_file, output_file, preset_quality,
                   display_aspect_ratio, **kwargs):
    """Encode a video that is already in the input folder.

    :param input_file: string with the filename, something like
        /eos/cds/test/sorenson/8f/m2/728-jsod98-8s9df2-89fg-lksdjf/data where
        the last part "data" is the filename and the last directory is the
        bucket id.
    :param output_file: the file to output the transcoded file.
    :param preset_quality: quality of the preset.
    :param display_aspect_ratio: the video's aspect ratio
    :param kwargs: other technical metadata
    :returns: job ID.
    """
    input_file = _filepath_for_samba(input_file)
    output_file = _filepath_for_samba(output_file)

    current_app.logger.debug('Encoding {0} with preset quality {1}'
                             .format(input_file, preset_quality))

    preset_id = get_preset_id(preset_quality, display_aspect_ratio)

    # Build the request of the encoding job
    json_params = generate_json_for_encoding(input_file, output_file,
                                             preset_id)
    proxies = current_app.config['CDS_SORENSON_PROXIES']
    headers = {'Accept': 'application/json'}
    response = requests.post(current_app.config['CDS_SORENSON_SUBMIT_URL'],
                             headers=headers, json=json_params,
                             proxies=proxies)

    data = json.loads(response.text)

    if response.status_code == requests.codes.ok:
        job_id = data.get('JobId')
        return job_id
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


def restart_encoding(job_id, input_file, output_file, preset_quality,
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
    return start_encoding(input_file, output_file, preset_quality,
                          display_aspect_ratio, **kwargs)


def get_presets_by_aspect_ratio(aspect_ratio):
    """Return the list of preset IDs for a given aspect ratio."""
    try:
        inner_dict = current_app.config['CDS_SORENSON_PRESETS'][aspect_ratio]
        return [preset['preset_id'] for preset in inner_dict.values()]
    except KeyError:
        raise InvalidAspectRatioError(aspect_ratio)


def get_available_aspect_ratios(pairs=False):
    """Return all available aspect ratios.

    :param pairs: if True, will return aspect ratios as pairs of integers
    """
    ratios = [key for key in current_app.config['CDS_SORENSON_PRESETS']]
    if pairs:
        ratios = [tuple(map(int, ratio.split(':', 1))) for ratio in ratios]
    return ratios


def get_available_preset_qualities():
    """Return all available preset qualities."""
    all_qualities = [
        outer_dict.keys()
        for outer_dict in current_app.config['CDS_SORENSON_PRESETS'].values()
    ]
    return list(OrderedDict.fromkeys(chain(*all_qualities)))


def get_preset_id(preset_quality, display_aspect_ratio, **kwargs):
    """Return the preset ID of the requested quality on given aspect ratio.

    :param preset_quality: the preset quality to use
    :param display_aspect_ratio: the video's aspect ratio
    :returns the corresponding preset ID or `None` if the given aspect ratio
    does not support this quality
    """
    try:
        aspect_ratio = current_app.config['CDS_SORENSON_PRESETS'][
            display_aspect_ratio]
        try:
            return aspect_ratio[preset_quality]['preset_id']
        except KeyError:
            raise InvalidResolutionError(display_aspect_ratio, preset_quality)
    except KeyError:
        raise InvalidAspectRatioError(display_aspect_ratio)


def get_preset_info(aspect_ratio, preset_quality):
    """Return technical information about given preset."""
    return current_app.config['CDS_SORENSON_PRESETS'].get(
        aspect_ratio, {}).get(preset_quality)
