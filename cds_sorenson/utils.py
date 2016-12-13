# -*- coding: utf-8 -*-
#
# This file is part of CERN Document Server.
# Copyright (C) 2016 CERN.
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

from os.path import dirname, normpath

import requests
from flask import current_app

from .error import SorensonError


def generate_json_for_encoding(input_file, preset_id, output_file):
    """Generate JSON that will be sent to Sorenson server to start encoding."""
    current_preset = _get_preset_config(preset_id)
    # Make sure the preset config exists for a given preset_id
    if not current_preset:
        raise SorensonError('Invalid preset "{0}"'.format(preset_id))

    return dict(
        Name='CDS File:{0} Preset:{1}'.format(input_file, preset_id),
        QueueId=current_app.config['CDS_SORENSON_DEFAULT_QUEUE'],
        JobMediaInfo=dict(
            SourceMediaList=dict(
                FileUri=input_file,
                UserName=current_app.config['CDS_SORENSON_USERNAME'],
                Password=current_app.config['CDS_SORENSON_PASSWORD'],
            ),
            DestinationList=[dict(FileUri='{}/'.format(output_file))],
            CompressionPresetList=[dict(PresetId=preset_id)],
        ),
    )


def name_generator(master_name, preset):
    """Generate the output name for slave file.

    :param master_name: string with the name of the master file.
    :param preset: dictionary with the preset information.
    :returns: string with the slave name for this preset.
    """
    return ("{master_name}-{video_bitrate}-kbps-{width}x{height}-audio-"
            "{audio_bitrate}-kbps-stereo.mp4".format(master_name='master_name',
                                                     **preset))


def get_status(job_id):
    """For a given job id, returns the status as JSON string.

    If the job can't be found in the current queue, it's probably done, so we
    check the archival queue. Raises an exception if there the response has a
    different code than 200.

    :param job_id: string with the job ID.
    :returns: JSON with the status or empty string if the job was not found.
    """
    current_jobs_url = (current_app
                        .config['CDS_SORENSON_CURRENT_JOBS_STATUS_URL']
                        .format(job_id=job_id))
    archive_jobs_url = (current_app
                        .config['CDS_SORENSON_ARCHIVE_JOBS_STATUS_URL']
                        .format(job_id=job_id))

    headers = {'Accept': 'application/json'}

    response = requests.get(current_jobs_url, headers=headers)

    if response.status_code == 404:
        response = requests.get(archive_jobs_url, headers=headers)

    if response.status_code == requests.codes.ok:
        return response.text
    else:
        raise SorensonError(response.status_code)


def _get_preset_config(preset_id):
    """Return preset config based on the preset_id."""
    PRESETS = current_app.config['CDS_SORENSON_PRESETS']
    presets_list = [item for presets in PRESETS.values() for item in presets]
    for preset in presets_list:
        if preset_id == preset.get('preset_id'):
            return preset
