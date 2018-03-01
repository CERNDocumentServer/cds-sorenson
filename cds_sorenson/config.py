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

"""CDS Sorenson default application configuration."""

from __future__ import absolute_import, print_function

from collections import OrderedDict

CDS_SORENSON_USERNAME = ''
CDS_SORENSON_PASSWORD = ''
"""Username and password to access the video on the filesystem.

Not needed anymore as we set the access rights on eos for webcast user.

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

CDS_SORENSON_PRESETS = OrderedDict([
    ('16:9', OrderedDict([
        ('360p', {
            'width': 640,
            'height': 360,
            'audio_bitrate': 64,
            'video_bitrate': 836,
            'total_bitrate': 900,
            'frame_rate': 25,
            'smil': True,
            'preset_id': 'dc2187a3-8f64-4e73-b458-7370a88d92d7'}),
        ('1080p', {
            'width': 1920,
            'height': 1080,
            'audio_bitrate': 128,
            'video_bitrate': 5872,
            'total_bitrate': 6000,
            'frame_rate': 25,
            'smil': True,
            'download': True,
            'type': 'hd',
            'preset_id': 'd9683573-f1c6-46a4-9181-d6048b2db305'}),
        ('720p', {
            'width': 1280,
            'height': 720,
            'audio_bitrate': 128,
            'video_bitrate': 2672,
            'total_bitrate': 2800,
            'frame_rate': 25,
            'smil': True,
            'preset_id': '79e9bde9-adcc-4603-b686-c7e2cb2d73d2'}),
        ('480p', {
            'width': 854,
            'height': 480,
            'audio_bitrate': 64,
            'video_bitrate': 1436,
            'total_bitrate': 1500,
            'frame_rate': 25,
            'smil': True,
            'preset_id': '9bd7c93f-88fa-4e59-a811-c81f4b0543db'}),
        ('240p', {
            'width': 432,
            'height': 240,
            'audio_bitrate': 64,
            'video_bitrate': 386,
            'total_bitrate': 450,
            'frame_rate': 15,
            'smil': True,
            'preset_id': '55f586de-15a0-45cd-bd30-bb6cf5bfe2b8'}),
        ('2160p', {
            'width': 3840,
            'height': 2160,
            'audio_bitrate': 128,
            'video_bitrate': 19872,
            'total_bitrate': 20000,
            'frame_rate': 25,
            'smil': True,
            'download': True,
            'type': 'ultra hd',
            'preset_id': '71d9865f-779d-4421-b62e-df93135b41c6'}),
        ('2160ph265', {
            # Different codec: H.265 - for download only - no SMIL file
            'width': 3840,
            'height': 2160,
            'audio_bitrate': 128,
            'video_bitrate': 19872,
            'total_bitrate': 20000,
            'frame_rate': 25,
            'download': True,
            'type': 'ultra hd',
            'preset_id': 'a0f072d3-c319-47e1-bdc4-787ad10be63c'}),
        ('1080ph265', {
            # Different codec: H.265 - for download only - no SMIL file
            'width': 1920,
            'height': 1080,
            'audio_bitrate': 128,
            'video_bitrate': 5872,
            'total_bitrate': 6000,
            'frame_rate': 25,
            'download': True,
            'type': 'hd',
            'preset_id': 'fd15cb19-6750-4872-a82b-e4625b842c30'})])),
    ('4:3', OrderedDict([
        ('360p', {
            'width': 480,
            'height': 360,
            'audio_bitrate': 64,
            'video_bitrate': 686,
            'total_bitrate': 750,
            'frame_rate': 25,
            'smil': True,
            'preset_id': '2b048f02-eca3-4e68-8eb6-f82375b1d15b'}),
        ('1080p', {
            'width': 1440,
            'height': 1080,
            'audio_bitrate': 128,
            'video_bitrate': 5372,
            'total_bitrate': 5500,
            'frame_rate': 25,
            'smil': True,
            'type': 'hd',
            'download': True,
            'preset_id': '216d5415-7d11-471c-bd06-0c013f657494'}),
        ('720p', {
            'width': 960,
            'height': 720,
            'audio_bitrate': 128,
            'video_bitrate': 2372,
            'total_bitrate': 2500,
            'frame_rate': 25,
            'smil': True,
            'type': 'hd',
            'download': True,
            'preset_id': '28ec9d35-00f3-400b-a955-dfb52f9d45ae'}),
        ('480p', {
            'width': 640,
            'height': 480,
            'audio_bitrate': 64,
            'video_bitrate': 1136,
            'total_bitrate': 1200,
            'frame_rate': 25,
            'smil': True,
            'preset_id': '7b7a3cae-2ca1-4b80-b756-b01fbdd46f78'}),
        ('240p', {
            'width': 320,
            'height': 240,
            'audio_bitrate': 64,
            'video_bitrate': 286,
            'total_bitrate': 350,
            'frame_rate': 15,
            'smil': True,
            'preset_id': 'a3214691-7f2b-47ff-a868-7bce6f5dbb7c'})])),
    ('3:2', OrderedDict([
        ('360p', {
            'width': 540,
            'height': 360,
            'audio_bitrate': 64,
            'video_bitrate': 736,
            'total_bitrate': 800,
            'frame_rate': 25,
            'smil': True,
            'preset_id': '52e22f47-b459-44a1-b41e-0219fe7d06c3'}),
        ('1080p', {
            'width': 1620,
            'height': 1080,
            'audio_bitrate': 128,
            'video_bitrate': 5472,
            'total_bitrate': 5600,
            'frame_rate': 25,
            'smil': True,
            'type': 'hd',
            'download': True,
            'preset_id': 'a3784f67-777a-42bc-8aa4-1a585d49276b'}),
        ('720p', {
            'width': 1080,
            'height': 720,
            'audio_bitrate': 128,
            'video_bitrate': 2472,
            'total_bitrate': 2600,
            'frame_rate': 25,
            'smil': True,
            'type': 'hd',
            'download': True,
            'preset_id': 'c3a1f9b0-b1dd-4987-b2b1-cd7936f114ed'}),
        ('480p', {
            'width': 720,
            'height': 480,
            'audio_bitrate': 64,
            'video_bitrate': 1236,
            'total_bitrate': 1300,
            'frame_rate': 25,
            'smil': True,
            'preset_id': 'e23bc6dd-e879-4e62-8692-48f6c9dd5bcc'}),
        ('240p', {
            'width': 360,
            'height': 240,
            'audio_bitrate': 64,
            'video_bitrate': 316,
            'total_bitrate': 380,
            'frame_rate': 15,
            'smil': True,
            'preset_id': '4ee80866-a960-41a7-887d-50041e991300'})])),
    ('256:135', OrderedDict([
        ('360p', {
            'width': 680,
            'height': 360,
            'audio_bitrate': 64,
            'video_bitrate': 836,
            'total_bitrate': 900,
            'frame_rate': 25,
            'smil': True,
            'preset_id': '89aeb4af-3d72-442d-8bc9-32b54244526a'}),
        ('1080p', {
            'width': 2040,
            'height': 1080,
            'audio_bitrate': 128,
            'video_bitrate': 5872,
            'total_bitrate': 6000,
            'frame_rate': 25,
            'smil': True,
            'download': True,
            'type': 'hd',
            'preset_id': '4e2d0677-317d-4aa3-9228-bdca00005f9f'}),
        ('720p', {
            'width': 1360,
            'height': 720,
            'audio_bitrate': 128,
            'video_bitrate': 2672,
            'total_bitrate': 2800,
            'frame_rate': 25,
            'smil': True,
            'type': 'hd',
            'download': True,
            'preset_id': 'dac209c0-1d2b-4cef-907d-882c30407690'}),
        ('480p', {
            'width': 906,
            'height': 480,
            'audio_bitrate': 64,
            'video_bitrate': 1436,
            'total_bitrate': 1500,
            'frame_rate': 25,
            'smil': True,
            'preset_id': '6da3e029-9cf4-46e6-8e5a-98dd4eddbe60'}),
        ('240p', {
            'width': 454,
            'height': 240,
            'audio_bitrate': 64,
            'video_bitrate': 386,
            'total_bitrate': 450,
            'frame_rate': 15,
            'smil': True,
            'preset_id': 'aa20a566-31ce-4e9c-b0ab-edc6b5f4146d'}),
        ('2160p', {
            'width': 4096,
            'height': 2160,
            'audio_bitrate': 128,
            'video_bitrate': 19872,
            'total_bitrate': 20000,
            'frame_rate': 25,
            'smil': True,
            'download': True,
            'type': 'ultra hd',
            'preset_id': 'beed7d75-4b78-46b7-9645-1a30413735ba'}),
        ('2160ph265', {
            # Different codec H.265 - for preview only - no SMIL file
            'width': 4096,
            'height': 2160,
            'audio_bitrate': 128,
            'video_bitrate': 19872,
            'total_bitrate': 20000,
            'frame_rate': 25,
            'download': True,
            'type': 'ultra hd',
            'preset_id': 'bddf6b9f-c15a-4333-9809-bdb9a244056b'})])),
    ('2:1', OrderedDict([
        ('360p', {
            'width': 720,
            'height': 360,
            'audio_bitrate': 64,
            'video_bitrate': 836,
            'total_bitrate': 900,
            'frame_rate': 25,
            'smil': True,
            'preset_id': '9ad2850f-40f3-45cd-9ab5-d12925294a17'}),
        ('1024p', {
            'width': 2048,
            'height': 1024,
            'audio_bitrate': 128,
            'video_bitrate': 5872,
            'total_bitrate': 6000,
            'frame_rate': 25,
            'smil': True,
            'type': 'hd',
            'download': True,
            'preset_id': '0149f7e7-e286-4604-a80a-23021b7d71b4'}),
        ('720p', {
            'width': 1440,
            'height': 720,
            'audio_bitrate': 128,
            'video_bitrate': 2672,
            'total_bitrate': 2800,
            'frame_rate': 25,
            'smil': True,
            'preset_id': 'b219ac63-00b4-4fef-8192-346fcf0cfe24'}),
        ('480p', {
            'width': 960,
            'height': 480,
            'audio_bitrate': 64,
            'video_bitrate': 1436,
            'total_bitrate': 1500,
            'frame_rate': 25,
            'smil': True,
            'preset_id': '120ebe70-1862-4dce-b4fb-6ddfc7b7f364'}),
        ('240p', {
            'width': 480,
            'height': 240,
            'audio_bitrate': 64,
            'video_bitrate': 386,
            'total_bitrate': 450,
            'frame_rate': 15,
            'smil': True,
            'preset_id': 'd910e3a5-5925-498f-8ce7-1c36e35c0d12'})]))
])
"""List of presets available on Sorenson server.

The first preset of each list is the previewer (slave small enough to be
quickly created but not too small to not be very pixelated).

Optional parameters for each preset configuration:
smil: specifies if this file should be added to the SMIL file, by default it is
    set to ``False``.
download: specifies if this file should be shown in the list of downloadable
    formats, by default is set to ``False``.
"""

CDS_SORENSON_NAME_GENERATOR = 'cds_sorenson.utils.name_generator'
"""Generator for output file names."""

CDS_SORENSON_PROXIES = {}
"""Proxies to connect to Sorenson, quite useful for testing.

The structure of the dictionary is as follows:

.. code-block:: python

    CDS_SORENSON_PROXIES = {
        'http': 'socks5://127.0.0.1:8123',
        'https': 'socks5://127.0.0.1:8123',
    }

Its value can be also set using environment variables:

.. code-block:: console

    $ export APP_CDS_SORENSON_PROXIES_HTTP="socks5://127.0.0.1:8123"
    $ export APP_CDS_SORENSON_PROXIES_HTTPS="socks5://127.0.0.1:8123"

This example assumes ``ssh`` tunneling:

.. code-block:: console

    $ ssh -fN -D 8123 <user-name>@wn03
"""

CDS_SORENSON_STATUSES = {
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
    10: 'Incomplete',
}
"""Statuses returned from Sorenson."""

CDS_SORENSON_SAMBA_DIRECTORY = 'file://media-smb.cern.ch/mediacds/'
"""Sorenson's EOS internal mounting point via samba."""

CDS_SORENSON_CDS_DIRECTORY = 'root://eosmedia.cern.ch//eos/media/cds/'
"""Video base file location in CDS."""
