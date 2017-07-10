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


"""Test utils functions."""

from __future__ import absolute_import, print_function

from jsonschema import validate

from cds_sorenson.utils import _get_preset_config, generate_json_for_encoding


def test_generate_json_for_encoding(app):
    """Test if the JSON generated for Sorenson has a correct structure.

    Wrong JSON won't return any meaningful errors, so it's very hard to debug
    """
    sorenson_schema = {
      "$schema": "http://json-schema.org/draft-04/schema#",
      "title": "Schema for Sorenson encoding.",
      "type": "object",
      "properties": {
        "QueueId": {
          "type": "string"
        },
        "JobMediaInfo": {
          "type": "object",
          "properties": {
            "CompressionPresetList": {
              "type": "array",
              "items": [
                {
                  "type": "object",
                  "properties": {
                    "PresetId": {
                      "type": "string"
                    }
                  },
                  "required": [
                    "PresetId"
                  ]
                }
              ]
            },
            "DestinationList": {
              "type": "array",
              "items": [
                {
                  "type": "object",
                  "properties": {
                    "FileUri": {
                      "type": "string"
                    }
                  },
                  "required": [
                    "FileUri"
                  ]
                }
              ]
            },
            "SourceMediaList": {
              "type": "array",
              "items": [
                {
                  "type": "object",
                  "properties": {
                    "Password": {
                      "type": "string"
                    },
                    "UserName": {
                      "type": "string"
                    },
                    "FileUri": {
                      "type": "string"
                    }
                  },
                  "required": [
                    "Password",
                    "UserName",
                    "FileUri"
                  ]
                }
              ]
            }
          },
          "required": [
            "CompressionPresetList",
            "DestinationList",
            "SourceMediaList"
          ]
        },
        "Name": {
          "type": "string"
        }
      },
      "required": [
        "QueueId",
        "JobMediaInfo",
        "Name"
      ]
    }

    expected_output = {
        'QueueId': '064153dd-ade2-4824-8458-88e6ea03d395',
        'JobMediaInfo': {
            'CompressionPresetList': [{
                'PresetId': 'dc2187a3-8f64-4e73-b458-7370a88d92d7'
            }],
            'DestinationList': [{
                'FileUri': '/tmp/test_output_file.mp4'
            }],
            'SourceMediaList': [{
                'Password': '',
                'UserName': '',
                'FileUri': '/tmp/test_input_file.mp4'
            }]},
        'Name': 'CDS File:/tmp/test_input_file.mp4 ' +
                'Preset:dc2187a3-8f64-4e73-b458-7370a88d92d7'
    }
    output = generate_json_for_encoding('/tmp/test_input_file.mp4',
                                        '/tmp/test_output_file.mp4',
                                        'dc2187a3-8f64-4e73-b458-7370a88d92d7')

    validate(output, sorenson_schema)
    assert output == expected_output


def test_get_preset_config(app):
    """Test `_get_preset_config` function."""
    assert _get_preset_config('dc2187a3-8f64-4e73-b458-7370a88d92d7') == {
        'width': 640,
        'height': 360,
        'audio_bitrate': 64,
        'video_bitrate': 836,
        'total_bitrate': 900,
        'frame_rate': 25,
        'smil': True,
        'preset_id': 'dc2187a3-8f64-4e73-b458-7370a88d92d7',
    }
