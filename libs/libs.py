from urllib.request import Request, urlopen
from json import loads
from math import radians, cos, sin, asin, sqrt, isclose
import pandas as pd
import numpy as np

def gmap_geocode_coords(address: str, G_API_KEY: str) -> tuple:
    '''
    Get Google API gets lat/lon from street address
    '''
    addr_ = address.replace(" ", "+")
    request = Request(
        (f"https://maps.googleapis.com/maps/api/geocode/json?"
         f"address={addr_}&key={G_API_KEY}"))
    response = urlopen(request).read()
    places = loads(response)

    if places['status'] == 'OK':
        lat = places['results'][0]['geometry']['location']['lat']
        lon = places['results'][0]['geometry']['location']['lng']
        return (lon, lat)

    else:
        print(f"Failed: {places['status']}")  


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # km
    return c * r



def get_nearest_grid(prism_dat: pd.DataFrame, lon_lat: tuple):
    lon = lon_lat[0]
    lat = lon_lat[1]

    # Filter to reduce apply function search
    indat = prism_dat[(prism_dat['lon'] >= (lon - .5)) & (prism_dat['lon'] <= (lon + .5))]
    indat = indat[(indat['lat'] >= (lat - .5)) & (indat['lat'] <= (lat + .5))]

    # Get distances for each grid
    indat['dist'] = indat.apply(lambda x: haversine(x['lon'], x['lat'], lon, lat), axis=1)

    # Get min index from dist and return gridNumber
    grid_number = indat.loc[indat[['dist']].idxmin()[0], 'gridNumber']

    return grid_number