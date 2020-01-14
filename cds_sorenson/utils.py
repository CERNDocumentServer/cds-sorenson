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

from itertools import chain

import requests
from flask import current_app

from .error import SorensonError


def generate_json_for_encoding(input_file, output_file, preset_id):
    """Generate JSON that will be sent to Sorenson server to start encoding."""
    current_preset = _get_preset_config(preset_id)
    # Make sure the preset config exists for a given preset_id
    if not current_preset:
        raise SorensonError('Invalid preset "{0}"'.format(preset_id))

    return dict(
        Name='CDS File:{0} Preset:{1}'.format(input_file, preset_id),
        QueueId=current_app.config['CDS_SORENSON_DEFAULT_QUEUE'],
        JobMediaInfo=dict(
            SourceMediaList=[dict(
                FileUri=input_file,
                UserName=current_app.config['CDS_SORENSON_USERNAME'],
                Password=current_app.config['CDS_SORENSON_PASSWORD'],
            )],
            DestinationList=[dict(FileUri='{}'.format(output_file))],
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
    proxies = current_app.config['CDS_SORENSON_PROXIES']

    api_username = current_app.config['CDS_SORENSON_API_USERNAME']
    api_password = current_app.config['CDS_SORENSON_API_PASSWORD']
    auth = (api_username, api_password) if api_username and api_password else \
        None

    response = requests.get(
        current_jobs_url, headers=headers, proxies=proxies, auth=auth)

    if response.status_code == 404:
        response = requests.get(
            archive_jobs_url, headers=headers, proxies=proxies, auth=auth)

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


def _filepath_for_samba(filepath):
    """Adjust file path for Samba protocol.

    Sorenson has the eos directory mounted through samba, so the paths
    need to be adjusted.
    """
    samba_dir = current_app.config['CDS_SORENSON_SAMBA_DIRECTORY']
    eos_dir = current_app.config['CDS_SORENSON_CDS_DIRECTORY']
    return filepath.replace(eos_dir, samba_dir)
