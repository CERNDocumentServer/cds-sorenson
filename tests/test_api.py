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


"""Test API functions."""

from __future__ import absolute_import, print_function

import pytest
from flask import Flask
from mock import MagicMock, patch

from cds_sorenson import CDSSorenson
from cds_sorenson.api import get_available_aspect_ratios, \
    get_available_preset_qualities, get_encoding_status, get_preset_id, \
    get_preset_info, get_presets_by_aspect_ratio, restart_encoding, \
    start_encoding, stop_encoding
from cds_sorenson.error import InvalidAspectRatioError, \
    InvalidResolutionError, SorensonError


class MockRequests(object):
    """Mock the requests library.

    We need to mock it like that, so we can count the number of times the
    delete function was called and raise exception if it was called more than
    once
    """

    called = 0
    codes = MagicMock()
    codes.ok = 200

    class MockResponse(object):
        """Mock of the Response object."""

        def __init__(self):
            self.status_code = 200

    @classmethod
    def delete(cls, delete_url, headers, **kwargs):
        """Mock the get method."""
        MockRequests.called += 1
        if MockRequests.called > 1:
            raise SorensonError()
        else:
            return cls.MockResponse()


def test_version():
    """Test version import."""
    from cds_sorenson import __version__
    assert __version__


def test_init():
    """Test extension initialization."""
    app = Flask('testapp')
    ext = CDSSorenson(app)
    assert 'cds-sorenson' in app.extensions

    app = Flask('testapp')
    ext = CDSSorenson()
    assert 'cds-sorenson' not in app.extensions
    ext.init_app(app)
    assert 'cds-sorenson' in app.extensions


@patch('cds_sorenson.api.requests.post')
def test_start_encoding(requests_post_mock, app, start_response):
    """Test if starting encoding works."""
    filename = 'file://cernbox-smb.cern.ch/eoscds/test/sorenson_input/' \
               '1111-dddd-3333-aaaa/data.mp4'
    # Random preset from config
    aspect_ratio, quality = '16:9', '360p'

    # Mock sorenson response
    sorenson_response = MagicMock()
    sorenson_response.text = start_response
    sorenson_response.status_code = 200
    requests_post_mock.return_value = sorenson_response

    job_id = start_encoding(filename, '', quality, aspect_ratio)
    assert job_id == "1234-2345-abcd"


@patch('cds_sorenson.api.requests.get')
def test_encoding_status(requests_get_mock, app, running_job_status_response):
    """Test if getting encoding status works."""
    job_id = "1234-2345-abcd"

    # Mock sorenson response
    sorenson_response = MagicMock()
    sorenson_response.text = running_job_status_response
    sorenson_response.status_code = 200
    requests_get_mock.return_value = sorenson_response

    encoding_status = get_encoding_status(job_id)
    assert encoding_status == ('Hold', 55.810001373291016)


@patch('cds_sorenson.api.requests.delete')
def test_stop_encoding(requests_delete_mock, app):
    """Test if stopping encoding works."""
    job_id = "1234-2345-abcd"

    # Mock sorenson response
    sorenson_response = MagicMock()
    sorenson_response.status_code = 200
    requests_delete_mock.return_value = sorenson_response

    returned_value = stop_encoding(job_id)
    # In case of some problems, we should get an exception
    assert returned_value is None


@patch('cds_sorenson.api.requests', MockRequests)
def test_stop_encoding_twice_fails(app):
    """Test if stopping the same job twice fails."""
    job_id = "1234-2345-abcd"

    # Stop encoding works for the first time...
    stop_encoding(job_id)
    # ... and fails for the second
    with pytest.raises(SorensonError):
        stop_encoding(job_id)


@patch('cds_sorenson.api.requests.post')
@patch('cds_sorenson.api.requests.delete')
def test_restart_encoding(requests_delete_mock, requests_post_mock, app,
                          start_response):
    """Test if restarting encoding works."""
    job_id = "1111-2222-aaaa"
    filename = '/sorenson_input/1111-dddd-3333-aaaa/data.mp4'
    # Random preset from config
    aspect_ratio, quality = '16:9', '360p'

    # Mock sorenson responses
    delete_response = MagicMock()
    delete_response.status_code = 200
    requests_delete_mock.return_value = delete_response

    post_response = MagicMock()
    post_response.text = start_response
    post_response.status_code = 200
    requests_post_mock.return_value = post_response

    job_id = restart_encoding(job_id, filename, '', quality, aspect_ratio)
    assert job_id == "1234-2345-abcd"


def test_invalid_request(app):
    """Test exceptions for invalid requests."""
    invalid_preset_quality = '522p'
    invalid_aspect_ratio = '15:3'

    with pytest.raises(InvalidAspectRatioError) as exc:
        start_encoding('input', '', '480p', invalid_aspect_ratio)
    assert invalid_aspect_ratio in str(exc)

    with pytest.raises(InvalidResolutionError) as exc:
        start_encoding('input', '', invalid_preset_quality, '16:9')
    assert '16:9' in str(exc)
    assert invalid_preset_quality in str(exc)


def test_get_presets_by_aspect_ratio(app):
    """Test `get_presets_by_aspect_ratio` function."""
    assert get_presets_by_aspect_ratio('16:9') == [
        'dc2187a3-8f64-4e73-b458-7370a88d92d7',
        'd9683573-f1c6-46a4-9181-d6048b2db305',
        '79e9bde9-adcc-4603-b686-c7e2cb2d73d2',
        '9bd7c93f-88fa-4e59-a811-c81f4b0543db',
        '55f586de-15a0-45cd-bd30-bb6cf5bfe2b8',
    ]


def test_available_aspect_ratios(app):
    """Test `get_available_aspect_ratios` function."""
    assert get_available_aspect_ratios() == ['16:9', '4:3', '3:2', '20:9',
                                             '256:135', '64:35', '2:1']
    assert get_available_aspect_ratios(pairs=True) == [
        (16, 9), (4, 3), (3, 2), (20, 9), (256, 135), (64, 35), (2, 1)]


def test_available_preset_qualities(app):
    """Test `get_available_preset_qualities` function."""
    assert get_available_preset_qualities() == ['360p', '1080p', '720p',
                                                '480p', '240p', '1024p']


def test_get_preset_id(app):
    """Test `get_preset_id` function."""
    assert get_preset_id('360p',
                         '16:9') == 'dc2187a3-8f64-4e73-b458-7370a88d92d7'
    assert get_preset_id('480p',
                         '2:1') == '120ebe70-1862-4dce-b4fb-6ddfc7b7f364'
    with pytest.raises(InvalidAspectRatioError):
        get_preset_id('480p', '27:9')
    with pytest.raises(InvalidResolutionError):
        get_preset_id('480p', '20:9')


def test_get_preset_info(app):
    """Test `get_preset_info` function."""
    info_keys = ['width', 'height', 'audio_bitrate', 'video_bitrate',
                 'total_bitrate', 'frame_rate', 'preset_id']
    for aspect_ratio in get_available_aspect_ratios():
        for preset_quality in \
                app.config['CDS_SORENSON_PRESETS'][aspect_ratio].keys():
            assert all([key in get_preset_info(aspect_ratio, preset_quality)
                        for key in info_keys])
