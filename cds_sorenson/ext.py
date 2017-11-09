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

"""CDS API to use Sorenson transcoding server."""

from __future__ import absolute_import, print_function

import os

from werkzeug.utils import cached_property

from . import config


class CDSSorenson(object):
    """CDS Sorenson extension."""

    def __init__(self, app=None):
        """Extension initialization."""
        if app:
            self.app = app
            self.init_app(app)

    def init_app(self, app):
        """Flask application initialization."""
        self.init_config(app)
        app.extensions['cds-sorenson'] = self

    def init_config(self, app):
        """Initialize configuration."""
        for k in dir(config):
            if k.startswith('CDS_SORENSON_'):
                app.config.setdefault(k, getattr(config, k))
        # Set the proxies
        if (os.environ.get('APP_CDS_SORENSON_PROXIES_HTTP') and
                os.environ.get('APP_CDS_SORENSON_PROXIES_HTTPS')):
            app.config['CDS_SORENSON_PROXIES'] = {
                'http': os.environ.get('APP_CDS_SORENSON_PROXIES_HTTP'),
                'https': os.environ.get('APP_CDS_SORENSON_PROXIES_HTTPS')
            }

    @cached_property
    def aspect_ratio_fractions(self):
        """Map aspect ratios with their computed fractions."""
        fractions_with_ar = {}
        for ar in self.app.config['CDS_SORENSON_PRESETS']:
            sorenson_w, sorenson_h = ar.split(':')
            sorenson_ar_fraction = float(sorenson_w) / float(sorenson_h)
            fractions_with_ar.setdefault(sorenson_ar_fraction, ar)

        return fractions_with_ar
