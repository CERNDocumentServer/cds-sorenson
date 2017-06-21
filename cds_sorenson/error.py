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


class SorensonError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, error_message=''):
        """Initialize exception with error message."""
        self.error_message = error_message

    def __str__(self):
        """Error message."""
        return self.error_message


class InvalidAspectRatioError(SorensonError):
    """Error for invalid aspect ratios."""

    def __init__(self, aspect_ratio):
        """Initialize exception with aspect ratio."""
        self.aspect_ratio = aspect_ratio

    def __str__(self):
        """Error message."""
        return 'Aspect ratio "{0}" is not supported.'.format(self.aspect_ratio)


class InvalidResolutionError(SorensonError):
    """Error for invalid resolutions."""

    def __init__(self, aspect_ratio, resolution):
        """Initialize exception with aspect ratio and resolution."""
        self.aspect_ratio = aspect_ratio
        self.resolution = resolution

    def __str__(self):
        """Error message."""
        return 'Aspect ratio "{0}" does not support resolution {1}.'.format(
            self.aspect_ratio, self.resolution)


class TooHighResolutionError(SorensonError):
    """The resolution is over the required maximum."""

    def __init__(self, aspect_ratio, max_height, max_width, height, width):
        """Initialize exception."""
        self._aspect_ratio = aspect_ratio
        self._max_height = max_height
        self._max_weight = max_width
        self._height = height
        self._width = width

    def __str__(self):
        """Error message."""
        return ('Aspect ratio "{0}" resolution {1}x{2} is higher than the '
                'maximum resolution accepted {3}x{4}.').format(
                    self._aspect_ratio, self._width, self._height,
                    self._max_weight, self._max_height)
