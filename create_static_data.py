import numpy as np
import pandas as pd
import geopandas as gpd
#import matplotlib.pyplot as plt
from itertools import chain
from shapely.geometry import Point
from shapely.geometry import LineString

state_gdf = gpd.read_file('data/tl_2022_us_state/tl_2022_us_state.shp')
state_fip_name_map = state_gdf[['STATEFP','STUSPS','NAME']].copy()
state_fip_name_map = state_fip_name_map.rename(columns={'NAME':'STNAME'})

# Replace 'counties.geojson' with the path to your data file
county_gdf = gpd.read_file('data/tl_2022_us_county/tl_2022_us_county.shp')

county_centroids = county_gdf[['STATEFP','COUNTYFP','GEOID','NAME','NAMELSAD','LSAD','INTPTLAT','INTPTLON']].copy()

# Merge in State information
county_centroids = county_centroids.merge(state_fip_name_map, on='STATEFP', how='left')


county_centroids.loc[:, 'INTPTLAT'] = county_centroids['INTPTLAT'].astype(float)
county_centroids.loc[:, 'INTPTLON'] = county_centroids['INTPTLON'].astype(float)

# Assuming your DataFrame is called `county_centroids`
county_centroids['geometry'] = county_centroids.apply(lambda row: Point(row['INTPTLON'], row['INTPTLAT']), axis=1)

# Now you have a GeoDataFrame with the 'geometry' column containing Point geometries.
county_centroids = gpd.GeoDataFrame(county_centroids[['STATEFP','STUSPS','STNAME','COUNTYFP','GEOID','NAME','NAMELSAD','LSAD','geometry']].copy(), geometry='geometry')

zip_code_gdf = gpd.read_file('data/zip_code_coordinates/ZIP_Code_Population_Weighted_Centroids.geojson')
zip_code_gdf = zip_code_gdf.rename(columns={'USPS_ZIP_PREF_STATE_1221':'STUSPS','USPS_ZIP_PREF_CITY_1221':'CITYUSPS'})
zip_code_centroids = zip_code_gdf[['STD_ZIP5','CITYUSPS','STUSPS','geometry']].copy()

# Merge in State information
zip_code_centroids = zip_code_centroids.merge(state_fip_name_map, on='STUSPS', how='left')
zip_code_centroids = zip_code_centroids[['STATEFP','STUSPS','STNAME','CITYUSPS','STD_ZIP5','geometry']]

county_centroids.to_file('data/block data/county_centroids.geojson', index=False, driver='GeoJSON')
zip_code_centroids.to_file('data/block data/zip_code_centroids.geojson', index=False, driver='GeoJSON')

## Convective Storm Event Index ##
conv_cat_map0 = pd.DataFrame({'event_index': [0],
                         'category': ['N/S']})
conv_cat_map1 = pd.DataFrame({'event_index': np.arange(1,7),
                         'category': ['TSTM','MRGL','SLGT','ENH','MDT','HIGH']})
conv_cat_map = pd.concat([conv_cat_map0,conv_cat_map1],ignore_index=True)
conv_cat_map.to_csv('convective_storm_cat_map.csv',index=False)

## Winter Storm Index ##
wint_cat_map0 = pd.DataFrame({'event_index': [0],
                         'category': ['N/S']})
wint_cat_map1 = pd.DataFrame({'event_index': np.arange(1,6),
                         'category': ['LIMITED','MINOR','MODERATE','MAJOR','EXTREME']})
wint_cat_map = pd.concat([wint_cat_map0,wint_cat_map1],ignore_index=True)
wint_cat_map.to_csv('winter_storm_cat_map.csv',index=False)
print('done')