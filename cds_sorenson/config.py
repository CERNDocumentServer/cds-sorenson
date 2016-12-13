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

"""CDS Sorenson default application configuration."""

from __future__ import absolute_import, print_function


CDS_SORENSON_USERNAME = ''
CDS_SORENSON_PASSWORD = ''
"""Username and password to access the video on the filesystem.

Important: The username and password will be visible in the metadata on the
Sorenson server, so please use a separate account just for Sorenson!
"""

CDS_SORENSON_SUBMIT_URL = 'http://sorenson01.cern.ch/api/jobs'
"""Sorenson endpoint for submitting a new transcoding job."""

CDS_SORENSON_DELETE_URL = 'http://sorenson01.cern.ch/api/jobs/{job_id}'
"""Sorenson endpoint for deleting a transcoding job."""

CDS_SORENSON_CURRENT_JOBS_STATUS_URL = \
    'http://sorenson01.cern.ch/api/jobs/status/{job_id}'
"""Sorenson endpoint for getting the status of a job waiting in the queue."""

CDS_SORENSON_ARCHIVE_JOBS_STATUS_URL = \
    'http://sorenson01.cern.ch/api/jobs/archive/{job_id}'
"""Sorenson endpoint for getting the status of an archived (done) job."""

CDS_SORENSON_DEFAULT_QUEUE = '064153dd-ade2-4824-8458-88e6ea03d395'
"""Default queue for all transcoding jobs."""

CDS_SORENSON_PRESETS = {
    '16:9': [
        {'resolution': '360p',
         'width': 640,
         'height': 360,
         'audio_bitrate': 64,
         'video_bitrate': 836,
         'total_bitrate': 900,
         'frame_rate': 25,
         'preset_id': 'dc2187a3-8f64-4e73-b458-7370a88d92d7'},
        {'resolution': '1080p',
         'width': 1920,
         'height': 1080,
         'audio_bitrate': 128,
         'video_bitrate': 5872,
         'total_bitrate': 6000,
         'frame_rate': 25,
         'preset_id': 'd9683573-f1c6-46a4-9181-d6048b2db305'},
        {'resolution': '720p',
         'width': 1280,
         'height': 720,
         'audio_bitrate': 128,
         'video_bitrate': 2672,
         'total_bitrate': 2800,
         'frame_rate': 25,
         'preset_id': '79e9bde9-adcc-4603-b686-c7e2cb2d73d2'},
        {'resolution': '480p',
         'width': 854,
         'height': 480,
         'audio_bitrate': 64,
         'video_bitrate': 1436,
         'total_bitrate': 1500,
         'frame_rate': 25,
         'preset_id': '9bd7c93f-88fa-4e59-a811-c81f4b0543db'},
        {'resolution': '240p',
         'width': 432,
         'height': 240,
         'audio_bitrate': 64,
         'video_bitrate': 386,
         'total_bitrate': 450,
         'frame_rate': 15,
         'preset_id': '55f586de-15a0-45cd-bd30-bb6cf5bfe2b8'}],
    '4:3': [
        {'resolution': '360p',
         'width': 480,
         'height': 360,
         'audio_bitrate': 64,
         'video_bitrate': 686,
         'total_bitrate': 750,
         'frame_rate': 25,
         'preset_id': '2b048f02-eca3-4e68-8eb6-f82375b1d15b'},
        {'resolution': '1080p',
         'width': 1440,
         'height': 1080,
         'audio_bitrate': 128,
         'video_bitrate': 5372,
         'total_bitrate': 5500,
         'frame_rate': 25,
         'preset_id': '216d5415-7d11-471c-bd06-0c013f657494'},
        {'resolution': '720p',
         'width': 960,
         'height': 720,
         'audio_bitrate': 128,
         'video_bitrate': 2372,
         'total_bitrate': 2500,
         'frame_rate': 25,
         'preset_id': '28ec9d35-00f3-400b-a955-dfb52f9d45ae'},
        {'resolution': '480p',
         'width': 640,
         'height': 480,
         'audio_bitrate': 64,
         'video_bitrate': 1136,
         'total_bitrate': 1200,
         'frame_rate': 25,
         'preset_id': '7b7a3cae-2ca1-4b80-b756-b01fbdd46f78'},
        {'resolution': '240p',
         'width': 320,
         'height': 240,
         'audio_bitrate': 64,
         'video_bitrate': 286,
         'total_bitrate': 350,
         'frame_rate': 15,
         'preset_id': 'a3214691-7f2b-47ff-a868-7bce6f5dbb7c'}],
    '3:2': [
        {'resolution': '360p',
         'width': 540,
         'height': 360,
         'audio_bitrate': 64,
         'video_bitrate': 736,
         'total_bitrate': 800,
         'frame_rate': 25,
         'preset_id': '52e22f47-b459-44a1-b41e-0219fe7d06c3'},
        {'resolution': '1080p',
         'width': 1620,
         'height': 1080,
         'audio_bitrate': 128,
         'video_bitrate': 5472,
         'total_bitrate': 5600,
         'frame_rate': 25,
         'preset_id': 'a3784f67-777a-42bc-8aa4-1a585d49276b'},
        {'resolution': '720p',
         'width': 1080,
         'height': 720,
         'audio_bitrate': 128,
         'video_bitrate': 2472,
         'total_bitrate': 2600,
         'frame_rate': 25,
         'preset_id': 'c3a1f9b0-b1dd-4987-b2b1-cd7936f114ed'},
        {'resolution': '480p',
         'width': 720,
         'height': 480,
         'audio_bitrate': 64,
         'video_bitrate': 1236,
         'total_bitrate': 1300,
         'frame_rate': 25,
         'preset_id': 'e23bc6dd-e879-4e62-8692-48f6c9dd5bcc'},
        {'resolution': '240p',
         'width': 360,
         'height': 240,
         'audio_bitrate': 64,
         'video_bitrate': 316,
         'total_bitrate': 380,
         'frame_rate': 15,
         'preset_id': '4ee80866-a960-41a7-887d-50041e991300'}],
    '20:9': [
        {'resolution': '240p',
         'width': 534,
         'height': 240,
         'audio_bitrate': 64,
         'video_bitrate': 316,
         'total_bitrate': 380,
         'frame_rate': 15,
         'preset_id': 'aba0570f-51de-4ad7-9af4-2ece661ddc7f'}],
    '256:135': [
        {'resolution': '360p',
         'width': 680,
         'height': 360,
         'audio_bitrate': 64,
         'video_bitrate': 836,
         'total_bitrate': 900,
         'frame_rate': 25,
         'preset_id': '89aeb4af-3d72-442d-8bc9-32b54244526a'},
        {'resolution': '1080p',
         'width': 2040,
         'height': 1080,
         'audio_bitrate': 128,
         'video_bitrate': 5872,
         'total_bitrate': 6000,
         'frame_rate': 25,
         'preset_id': '4e2d0677-317d-4aa3-9228-bdca00005f9f'},
        {'resolution': '720p',
         'width': 1360,
         'height': 720,
         'audio_bitrate': 128,
         'video_bitrate': 2672,
         'total_bitrate': 2800,
         'frame_rate': 25,
         'preset_id': 'dac209c0-1d2b-4cef-907d-882c30407690'},
        {'resolution': '480p',
         'width': 906,
         'height': 480,
         'audio_bitrate': 64,
         'video_bitrate': 1436,
         'total_bitrate': 1500,
         'frame_rate': 25,
         'preset_id': '6da3e029-9cf4-46e6-8e5a-98dd4eddbe60'},
        {'resolution': '240p',
         'width': 454,
         'height': 240,
         'audio_bitrate': 64,
         'video_bitrate': 386,
         'total_bitrate': 450,
         'frame_rate': 15,
         'preset_id': 'aa20a566-31ce-4e9c-b0ab-edc6b5f4146d'}],
    '64:35': [
        {'resolution': '360p',
         'width': 658,
         'height': 360,
         'audio_bitrate': 64,
         'video_bitrate': 836,
         'total_bitrate': 900,
         'frame_rate': 25,
         'preset_id': 'a1579abc-ac74-4273-9671-d758cb3c413e'},
        {'resolution': '1080p',
         'width': 1976,
         'height': 1080,
         'audio_bitrate': 128,
         'video_bitrate': 5872,
         'total_bitrate': 6000,
         'frame_rate': 25,
         'preset_id': 'b4426e61-60fc-44ef-8281-8e838a107f8e'}],
    '2:1': [
        {'resolution': '360p',
         'width': 720,
         'height': 360,
         'audio_bitrate': 64,
         'video_bitrate': 836,
         'total_bitrate': 900,
         'frame_rate': 25,
         'preset_id': '9ad2850f-40f3-45cd-9ab5-d12925294a17'},
        {'resolution': '1024p',
         'width': 2048,
         'height': 1024,
         'audio_bitrate': 128,
         'video_bitrate': 5872,
         'total_bitrate': 6000,
         'frame_rate': 25,
         'preset_id': '0149f7e7-e286-4604-a80a-23021b7d71b4'},
        {'resolution': '720p',
         'width': 1440,
         'height': 720,
         'audio_bitrate': 128,
         'video_bitrate': 2672,
         'total_bitrate': 2800,
         'frame_rate': 25,
         'preset_id': 'b219ac63-00b4-4fef-8192-346fcf0cfe24'},
        {'resolution': '480p',
         'width': 960,
         'height': 480,
         'audio_bitrate': 64,
         'video_bitrate': 1436,
         'total_bitrate': 1500,
         'frame_rate': 25,
         'preset_id': '120ebe70-1862-4dce-b4fb-6ddfc7b7f364'},
        {'resolution': '240p',
         'width': 480,
         'height': 240,
         'audio_bitrate': 64,
         'video_bitrate': 386,
         'total_bitrate': 450,
         'frame_rate': 15,
         'preset_id': 'd910e3a5-5925-498f-8ce7-1c36e35c0d12'}]
}
"""List of presets available on Sorenson server.

The first preset of each list is the previewer (slave small enough to be
quickly created but not too small to not be very pixelated).
"""

CDS_SORENSON_NAME_GENERATOR = 'cds_sorenson.utils.name_generator'
"""Generator for output file names."""
