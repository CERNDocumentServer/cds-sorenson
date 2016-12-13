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


"""Pytest configuration."""

from __future__ import absolute_import, print_function

import pytest
from flask import Flask

from cds_sorenson import CDSSorenson


@pytest.fixture()
def config():
    """Custom config for testing purposes."""
    return dict(
        TESTING=True
    )


@pytest.yield_fixture()
def app(config):
    """Flask application fixture."""
    app_ = Flask('testapp')

    CDSSorenson(app_)

    app_.config.update(config)

    with app_.app_context():
        yield app_


@pytest.fixture()
def start_response():
    """Example response when starting new encoding job."""
    return u'{"JobId":"1234-2345-abcd"}'


@pytest.fixture()
def running_job_status_response():
    """Example response when checking the status of running encoding job."""
    return u"""{
        "EncodeServerId":null,
        "JobId":"11111111-aaaa",
        "LastUpdate":null,
        "LastUpdateIso8601":null,
        "Name":"CDS_TEST.mp4",
        "QueueId":"00000000-0000-0000-0000-000000000000",
        "Status": {
            "Created":null,
            "CreatedIso8601":null,
            "DestinationName":null,
            "Duration":"PT0S",
            "ErrorCode":null,
            "Id":"00000000-0000-0000-0000-000000000000",
            "Modified":null,
            "ModifiedIso8601":null,
            "PresetName":null,
            "Progress":55.810001373291016,
            "RetryCount":null,
            "Status":9,
            "StatusMessage":null,
            "TimeFinished":null,
            "TimeFinishedIso8601":null,
            "TimeStarted":"\\/Date(1471610064000+0200)\\/",
            "TimeStartedIso8601":"2016-08-19T12:34:24.0000000Z"
        },
        "TimeSubmitted":"\\/Date(1471609988000+0200)\\/",
        "TimeSubmittedIso8601":"2016-08-19T12:33:08.0000000Z",
        "TotalSourceSize":28335481
    }"""


@pytest.fixture()
def finished_job_status_response():
    """Example response when checking the status of finished encoding job."""
    return u"""{
       "TimeSubmittedIso8601":"2016-10-19T12:30:12.0000000Z",
       "PercentCompleteOverall":100,
       "ArchiveMetadataList":[

       ],
       "QueueName":"big_files",
       "ThumbTime":None,
       "OutPoint":None,
       "TimeSubmitted":"/Date(1476880212000+0200)/",
       "TimeArchivedIso8601":"2016-10-19T12:31:10.0000000Z",
       "TimeStartedIso8601":"2016-10-19T12:30:12.0000000Z",
       "StatusState":"Finished",
       "TimeNotified":None,
       "OriginalJobId":"3c826c49-3d89-44e9-ac71-e9f02d5702ec",
       "OutputList":[
          {
             "FileSize":16090019,
             "FileVidDataRate":"2000000",
             "DurationSeconds":60.095,
             "FileFrameRate":"23.976",
             "FileHeight":360,
             "FileAudDataRate":"192000",
             "DestinationId":"eac9a16b-ffe2-44c6-8bcd-8b49f6da3656",
             "OutputId":"260af3c3-646a-45c1-b7aa-4997c1baceef",
             "FileType":"VideoOutput",
             "FileName":"data-3c826c49-3d89-44e9-ac71-e9f02d5702ec-YouTube_480p.mp4",
             "FileWidth":640
          }
       ],
       "WatchFolderName":None,
       "TimeFinished":"/Date(1476880269000+0200)/",
       "ErrorCode":None,
       "TimeNotifiedIso8601":None,
       "TimeArchived":"/Date(1476880270000+0200)/",
       "CompressionPresetList":[
          {
             "PresetId":"2c5a86db-1018-4ff8-a5ad-daebd4cb4ff4",
             "WatchFolderId":None,
             "PresetXmlBase64Data":"91dHB1dD4=",
             "JobPresetId":"a9bb37af-dab3-4a30-86b9-1b78cbc4b43d",
             "UriLocation":None,
             "JobId":None
          }
       ],
       "SourceMediaList":[
          {
             "S3BucketName":None,
             "FileSize":-1,
             "ModifiedIso8601":None,
             "CredentialId":None,
             "WatermarkInfo":{
                "WatermarkImageUserName":None,
                "WatermarkImageCredentialId":None,
                "WatermarkImagePassword":None,
                "WatermarkImageUri":None
             },
             "FileUri":"file://cernbox-smb.cern.ch/eoscds/test/sorenson_input/1111-dddd-3333-aaaa/data.mp4",
             "CompressOrder":0,
             "CreatedIso8601":None,
             "JobSubmittedIso8601":None,
             "Password":None,
             "Type":1,
             "FileName":None,
             "EncodeAllAudioTracks":False,
             "AutoDeleteSource":False,
             "IsWatchfolder":False,
             "JobId":None,
             "Modified":None,
             "WatchFolderId":None,
             "UserName":"",
             "Created":None,
             "JobSubmitted":None,
             "SourceId":"00000000-0000-0000-0000-000000000000"
          }
       ],
       "InPoint":None,
       "ServerName":"sorenson02.cern.ch",
       "JobId":"3c826c49-3d89-44e9-ac71-e9f02d5702ec",
       "TimeStarted":"/Date(1476880212000+0200)/",
       "DestinationList":[
          {
             "S3BucketName":None,
             "ModifiedIso8601":None,
             "DestinationMetadataList":[

             ],
             "DestinationId":"eac9a16b-ffe2-44c6-8bcd-8b49f6da3656",
             "ThumbUri":None,
             "FileNamingMethod":"Undefined",
             "Type":1,
             "WatchFolderId":None,
             "JobId":None,
             "FileUri":"file://cernbox-smb.cern.ch/eoscds/test/sorenson_output/1111-dddd-3333-aaaa/",
             "ThumbFilePattern":None,
             "CreatedIso8601":None,
             "ExtensionNamingMethod":"Undefined",
             "CredentialId":None,
             "S3ThumbBucket":None,
             "Created":None,
             "Password":None,
             "UserName":None,
             "Modified":None,
             "FileName":None,
             "DestinationName":None
          }
       ],
       "StatusStateId":5,
       "StatusList":[
          {
             "RetryCount":0,
             "TimeStartedIso8601":"2016-10-19T12:30:13.0000000Z",
             "Status":"Downloading",
             "TimeFinishedIso8601":"2016-10-19T12:30:13.0000000Z",
             "TimeStarted":"/Date(1476880213000+0200)/",
             "Duration":"PT0S",
             "Progress":100,
             "ErrorCode":None,
             "Message":None,
             "PresetName":None,
             "TimeFinished":"/Date(1476880213000+0200)/",
             "DestinationName":None
          },
          {
             "RetryCount":0,
             "TimeStartedIso8601":"2016-10-19T12:30:49.0000000Z",
             "Status":"Transcoding",
             "TimeFinishedIso8601":"2016-10-19T12:31:08.0000000Z",
             "TimeStarted":"/Date(1476880249000+0200)/",
             "Duration":"PT19S",
             "Progress":100,
             "ErrorCode":None,
             "Message":None,
             "PresetName":"YouTube_480p",
             "TimeFinished":"/Date(1476880268000+0200)/",
             "DestinationName":None
          },
          {
             "RetryCount":0,
             "TimeStartedIso8601":"2016-10-19T12:31:08.0000000Z",
             "Status":"Uploading",
             "TimeFinishedIso8601":"2016-10-19T12:31:09.0000000Z",
             "TimeStarted":"/Date(1476880268000+0200)/",
             "Duration":"PT1S",
             "Progress":100,
             "ErrorCode":None,
             "Message":None,
             "PresetName":None,
             "TimeFinished":"/Date(1476880269000+0200)/",
             "DestinationName":None
          }
       ],
       "StatusMessage":"",
       "Name":"CDS-1c7b1844-d820-4207-85ca-8ed54bc6805",
       "TimeFinishedIso8601":"2016-10-19T12:31:09.0000000Z",
       "SourceMedia":{
          "S3BucketName":None,
          "FileSize":-1,
          "ModifiedIso8601":None,
          "CredentialId":None,
          "WatermarkInfo":{
             "WatermarkImageUserName":None,
             "WatermarkImageCredentialId":None,
             "WatermarkImagePassword":None,
             "WatermarkImageUri":None
          },
          "FileUri":"file://cernbox-smb.cern.ch/eoscds/test/sorenson_input/1111-dddd-3333-aaaa/data.mp4",
          "CompressOrder":0,
          "CreatedIso8601":None,
          "JobSubmittedIso8601":None,
          "Password":None,
          "Type":1,
          "FileName":None,
          "EncodeAllAudioTracks":False,
          "AutoDeleteSource":False,
          "IsWatchfolder":False,
          "JobId":None,
          "Modified":None,
          "WatchFolderId":None,
          "UserName":"",
          "Created":None,
          "JobSubmitted":None,
          "SourceId":"00000000-0000-0000-0000-000000000000"
       }
    }"""
