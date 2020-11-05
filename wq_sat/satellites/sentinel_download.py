# -*- coding: utf-8 -*-

# Copyright 2018 Spanish National Research Council (CSIC)
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""
Given two dates and region, download N Sentinel Collections scenes from ESA
Sentinel dataHUB.
The downloaded Sentinel collection scenes are compatible with:
S2MSI1C: Top-of-atmosphere reflectances in cartographic geometry
or S2MSI2A: Bottom-of-atmosphere reflectance in cartographic geometry
Parameters
----------
inidate: datetime.strptime("YYYY-MM-dd", "%Y-%m-%d")
enddate: datetime.strptime("YYYY-MM-dd", "%Y-%m-%d")
region: name of one reservoir saved in the "coord_reservoirs.json" file
coordinates : dict. Coordinates of the region to search.
Example: {"W": -2.830, "S": 41.820, "E": -2.690, "N": 41.910}}
platform : str. Satellite to use from the Sentinel family
producttype : str. Dataset type.
cloud: int
path : path
Author: Daniel García Díaz
Email: garciad@ifca.unican.es
Institute of Physics of Cantabria (IFCA)
Advanced Computing and e-Science
Date: Sep 2018
"""
#imports apis
import requests
import os

# Subfunctions
from wq_sat.utils import config

class download_sentinel:

    def __init__(self, inidate, enddate, region, coordinates=None, platform='Sentinel-2', producttype="S2MSI1C", cloud=100,
                 output_path=None):

        self.session = requests.Session()

        #Search parameters
        self.inidate = inidate
        self.enddate = enddate
        self.coord = coordinates
        self.producttype = producttype
        self.platform = platform
        self.region = region
        self.cloud = int(cloud)

        #work path
        if output_path:
            self.output_path = os.path.join(output_path, self.region)
        else:
            path = config.get_data_path()
            self.output_path = os.path.join(path, self.region)
            
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
            

        #ESA APIs
        self.api_url = 'https://scihub.copernicus.eu/apihub/'
        self.credentials = config.load_credentials()['sentinel']

    def search(self, omit_corners=True):

        # Post the query to Copernicus
        query = {'footprint': '"Intersects(POLYGON(({0} {1},{2} {1},{2} {3},{0} {3},{0} {1})))"'.format(self.coord['W'],
                                                                                                        self.coord['S'],
                                                                                                        self.coord['E'],
                                                                                                        self.coord['N']),
                 'producttype': self.producttype,
                 'platformname': self.platform,
                 'beginposition': '[{} TO {}]'.format(self.inidate, self.enddate),
                 'cloudcoverpercentage': '[0 TO {}]'.format(self.cloud)
                 }

        data = {'format': 'json',
                'start': 0,  # offset
                'rows': 100,
                'limit': 100,
                'orderby': '',
                'q': ' '.join(['{}:{}'.format(k, v) for k, v in query.items()])
                }

        response = self.session.post(self.api_url + 'search?',
                                 data=data,
                                 auth=(self.credentials['user'], self.credentials['password']),
                                 headers={'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'})

        response.raise_for_status()

        # Parse the response
        json_feed = response.json()['feed']

        if 'entry' in json_feed.keys():
            results = json_feed['entry']
            if isinstance(results, dict):  # if the query returns only one product, products will be a dict not a list
                results = [results]
        else:
            results = []

        # Remove results that are mainly corners
        def keep(r):
            for item in r['str']:
                if item['name'] == 'size':
                    units = item['content'].split(' ')[1]
                    mult = {'KB': 1, 'MB': 1e3, 'GB': 1e6}[units]
                    size = float(item['content'].split(' ')[0]) * mult
                    break
            if size > 0.5e6:  # 500MB
                return True
            else:
                return False
        results[:] = [r for r in results if keep(r)]

        print('Found {} results from Sentinel'.format(json_feed['opensearch:totalResults']))
        print('Retrieving {} results'.format(len(results)))

        return results


    def download(self):

        #results of the search
        results = self.search()
        if not isinstance(results, list):
            results = [results]
            
        downloaded_data = [f for f in os.listdir(self.output_path) if os.path.isfile(os.path.join(self.output_path, f))]
        s2_tiles = []

        for r in results:

            url, tile_id = r['link'][0]['href'], r['title']
            save_dir = os.path.join(self.output_path, '{}.zip'.format(tile_id))
            
            if '{}.tif'.format(tile_id) in downloaded_data: 
                print ('File {} already downloaded'.format(tile_id))
                continue
                
            print('Downloading {} ...'.format(tile_id))
            s2_tiles.append(tile_id)
            
            response = self.session.get(url, stream=True, allow_redirects=True, auth=(self.credentials['user'],
                                                                                      self.credentials['password']))

            with open(save_dir, 'wb') as f:
                f.write(response.content)
                
        return s2_tiles