"""
Utils to interact with GDAL.

Date: February 2019
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia
"""

import re
import os

from osgeo import gdal, osr

def lonlat_to_xy(lon, lat, ds):
    """

    Parameters
    ----------
    lon: str
    lat: str
    ds: GDAL Dataset

    Returns
    -------
    A pair of coordinates in meters/pixels ????
    """

    xoff, a, b, yoff, d, e = ds.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    srsLatLon = osr.SpatialReference()
    srsLatLon.SetWellKnownGeogCS("WGS84")
    ct = osr.CoordinateTransformation(srsLatLon, srs)

    (xp, yp, h) = ct.TransformPoint(lon, lat, 0.)
    xp -= xoff
    yp -= yoff
    # matrix inversion
    det_inv = 1. / (a * e - d * b)
    x = (e * xp - b * yp) * det_inv
    y = (-d * xp + a * yp) * det_inv
    return int(x), int(y)