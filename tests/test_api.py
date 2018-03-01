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
from cds_sorenson.api import _get_available_aspect_ratios, \
    _get_closest_aspect_ratio, _get_quality_preset, can_be_transcoded, \
    get_all_distinct_qualities, get_encoding_status, restart_encoding, \
    start_encoding, stop_encoding
from cds_sorenson.error import InvalidAspectRatioError, \
    InvalidResolutionError, SorensonError, TooHighResolutionError


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

    job_id, _, _ = start_encoding(filename, '', quality, aspect_ratio)
    assert job_id == "1234-2345-abcd"

    with pytest.raises(TooHighResolutionError):
        start_encoding(filename, '', quality, aspect_ratio, max_height=240)


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

    job_id, _, _ = restart_encoding(job_id, filename, '', quality,
                                    aspect_ratio)
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


def test_available_aspect_ratios(app):
    """Test `get_available_aspect_ratios` function."""
    assert _get_available_aspect_ratios() == ['16:9', '4:3', '3:2', '256:135',
                                              '2:1']
    assert _get_available_aspect_ratios(pairs=True) == [
        (16, 9), (4, 3), (3, 2), (256, 135), (2, 1)]


def test_available_preset_qualities(app):
    """Test `get_all_distinct_qualities` function."""
    assert sorted(get_all_distinct_qualities()) == sorted(
        ['240p', '360p', '480p', '720p', '1024p', '1080p', '1080ph265',
         '2160p', '2160ph265'])


def test_get_quality_preset(app):
    """Test `get_preset_id` function."""
    _, config = _get_quality_preset('360p', '16:9')
    assert config['preset_id'] == 'dc2187a3-8f64-4e73-b458-7370a88d92d7'

    _, config = _get_quality_preset('480p', '2:1')
    assert config['preset_id'] == '120ebe70-1862-4dce-b4fb-6ddfc7b7f364'

    with pytest.raises(InvalidAspectRatioError):
        _get_quality_preset('480p', '27:9')
    with pytest.raises(InvalidResolutionError):
        _get_quality_preset('1080p', '2:1')
    with pytest.raises(TooHighResolutionError):
        _get_quality_preset('1080p', '16:9', video_height=720)
    with pytest.raises(TooHighResolutionError):
        _get_quality_preset('1080p', '16:9', video_width=1280)
    with pytest.raises(TooHighResolutionError):
        _get_quality_preset('1080p', '16:9', video_height=720,
                            video_width=1280)

    ar, config = _get_quality_preset('720p', '16:9')
    assert ar == '16:9'
    assert config is not None


def test_all_get_quality_preset(app):
    """Test `test_get_quality_preset` function."""
    info_keys = ['width', 'height', 'audio_bitrate', 'video_bitrate',
                 'total_bitrate', 'frame_rate', 'preset_id']
    for aspect_ratio in _get_available_aspect_ratios():
        for preset_quality in \
                app.config['CDS_SORENSON_PRESETS'][aspect_ratio].keys():
            _, config = _get_quality_preset(preset_quality, aspect_ratio)
            assert all([key in config for key in info_keys])


def test_can_be_transcoded(app):
    """Test `test_can_be_transcoded` function."""
    # standard case
    result = can_be_transcoded('360p', '16:9')
    assert result['quality'] == '360p'
    assert result['aspect_ratio'] == '16:9'
    assert result['width'] == 640
    assert result['height'] == 360

    # aspect ratio fallback
    result = can_be_transcoded('720p', '5:4', 1024, 768)
    assert result['quality'] == '720p'
    assert result['aspect_ratio'] == '4:3'
    assert result['width'] == 960
    assert result['height'] == 720

    # no transcoding
    assert not can_be_transcoded('720p', '5:4')
    assert not can_be_transcoded('720p', '5:4', 480, 360)


def test_no_smil_config_option(app):
    """Test that some formats should not be added to the SMIL file."""
    _, config = _get_quality_preset('1080ph265', '16:9')
    assert not config.get('smil')

    _, config = _get_quality_preset('1080p', '16:9')
    assert config.get('smil')


def test_get_closest_aspect_ratio(app):
    """Test aspect ratio fallback for unknown aspect ratios."""
    # not configured aspect ratios
    # 11:9
    assert _get_closest_aspect_ratio(352, 288) == '4:3'
    # 25:14
    assert _get_closest_aspect_ratio(400, 224) == '16:9'
    # 5:4
    assert _get_closest_aspect_ratio(720, 576) == '4:3'
    assert _get_closest_aspect_ratio(400, 320) == '4:3'
    assert _get_closest_aspect_ratio(360, 288) == '4:3'
    assert _get_closest_aspect_ratio(320, 256) == '4:3'
    # 40:27
    assert _get_closest_aspect_ratio(720, 486) == '3:2'
    # 20:9
    assert _get_closest_aspect_ratio(600, 270) == '2:1'
    # 64:35
    assert _get_closest_aspect_ratio(4096, 2240) == '16:9'
    # 295:162
    assert _get_closest_aspect_ratio(720, 395) == '16:9'
    # 295:216
    assert _get_closest_aspect_ratio(720, 527) == '4:3'
