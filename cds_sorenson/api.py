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

import json

import requests
from flask import current_app

from .error import SorensonError
from .utils import generate_json_for_encoding, get_status


def start_encoding(input_file, preset_ID, output_file=None):
    """Encode a video that is already in the input folder.

    :param input_file: string with the filename, something like
        /eos/cds/test/sorenson/8f/m2/728-jsod98-8s9df2-89fg-lksdjf/data where
        the last part "data" is the filename and the last directory is the
        bucket id.
    :param preset_ID: id of the preset.
    :param output_file: the file to output the transcoded file.
    :returns: job ID.
    """
    current_app.logger.debug('Encoding {0} with the preset {1}'
                             .format(input_file, preset_ID))

    # Build the request of the encoding job
    json_params = generate_json_for_encoding(input_file, preset_ID,
                                             output_file)

    headers = {'Accept': 'application/json'}
    response = requests.post(current_app.config['CDS_SORENSON_SUBMIT_URL'],
                             headers=headers, json=json_params)

    data = json.loads(response.text)

    if response.status_code == requests.codes.ok:
        job_id = data.get('JobId')
        return job_id
    else:
        # something is wrong - sorenson server is not responding or the
        # configuration is wrong and we can't contact sorenson server
        raise SorensonError(response.status_code)


def stop_encoding(job_id):
    """Stop encoding job.

    :param job_id: string with the job ID.
    :returns: None.
    """
    delete_url = (current_app.config['CDS_SORENSON_DELETE_URL']
                             .format(job_id=job_id))
    headers = {'Accept': 'application/json'}

    response = requests.delete(delete_url, headers=headers)
    if response.status_code != requests.codes.ok:
        raise SorensonError(response.status_code)


def get_encoding_status(job_id):
    """Get status of a given job from the Sorenson server.

    If the job can't be found in the current queue, it's probably done, so we
    check the archival queue.

    :param job_id: string with the job ID.
    :returns: tuple with the status message and progress in %.
    """
    SORENSON_STATUSES = {
        0: 'Undefined',
        1: 'Waiting',
        2: 'Downloading',
        3: 'Transcoding',
        4: 'Uploading',
        5: 'Finished',
        6: 'Error',
        7: 'Canceled',
        8: 'Deleted',
        9: 'Hold',
        10: 'Incomplete'
    }

    status = get_status(job_id)
    if status == '':
        # encoding job was canceled
        return "Canceled", 100
    status_json = json.loads(status)
    # there are different ways to get the status of a job, depending if
    # the job was successful, so we should check for the status code in
    # different places
    job_status = status_json.get('Status', None).get('Status')
    job_progress = status_json.get('Status', None).get('Progress')
    if job_status:
        return SORENSON_STATUSES.get(job_status), job_progress
    # status not found? check in different place
    job_status = status_json.get('StatusStateId')
    if job_status:
        # job is probably either finished or failed, so the progress will
        # always be 100% in this case
        return SORENSON_STATUSES.get(job_status), 100
    # No status was found (which shouldn't happen)
    raise SorensonError('No status found for job: {0}'.format(job_id))


def restart_encoding(job_id, input_file, preset_ID, output_file=None):
    """Try to stop the encoding job and start a new one.

    It's impossible to get the input_file and preset_ID from the job_id, if
    the job has not yet finished, so we need to specify all parameters for
    stopping and starting the encoding job.
    """
    try:
        stop_encoding(job_id)
    except SorensonError:
        # If we failed to stop the encoding job, ignore it - in the worst
        # case the encoding will finish and we will overwrite the file.
        pass
    return start_encoding(input_file, preset_ID, output_file)
