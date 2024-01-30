import re
import requests
import zipfile
import os
from math import ceil, floor
import numpy as np
import pandas as pd
import geopandas as gpd
#import matplotlib.pyplot as plt
from itertools import chain
from shapely.geometry import Point
from shapely.geometry import LineString
import alphashape


## Storm 'category' ordered by 'event_index'
conv_cat_map = pd.read_csv('convective_storm_cat_map.csv')
wint_cat_map = pd.read_csv('winter_storm_cat_map.csv')
print('Ordered storm categories loaded')

## Load static files
county_centroids = gpd.read_file('static_data/outputs/county_centroids.geojson')
zip_code_centroids = gpd.read_file('static_data/outputs/zipcode_centroids.geojson')



conv_day_20230331 = ['day1otlk_20230331_1630_cat.nolyr','day2otlk_20230331_1730_cat.nolyr']


conv_base_url_123 = 'https://www.spc.noaa.gov/products/outlook/archive/2023/'
conv_storm_urls = [conv_base_url_123 + file for file in conv_day_20230331]
print('List of conv links created')

def get_storm_table(storm_gdf,block_file,my_cols_keep):

    #storm_gdf = gpd.read_file(link)

    # Create an empty DataFrame with a column for each polygon
    result_df = gpd.GeoDataFrame()
    for idx, polygon in enumerate(storm_gdf['geometry']):
        result_df[f'polygon_{idx + 1}'] = block_file['geometry'].within(polygon).astype(int)


    result_df.index = block_file.index

    # Multiply each value by the column index + 1
    modified_values = result_df.values * np.arange(1, len(result_df.columns) + 1)

    # Use np.argmax to find the maximum column index for each row
    if (modified_values.size > 0):
        mutually_exclusive = np.max(modified_values, axis=1)
    else:
        mutually_exclusive = 0

    final_df = block_file[my_cols_keep].copy()
    final_df['event_index'] = mutually_exclusive

    return final_df

def add_issue_times(my_df,time_dict):
    for time_item in time_dict:
        my_df[time_item] = time_dict[time_item]
    return my_df

## Combine all convective storm geojsons into a single gdf
for idx,link in enumerate(conv_storm_urls):
    conv_storm_by_day_gdf = gpd.read_file(link)
    conv_storm_by_day_gdf['outlook_day'] = idx + 1
    if idx == 0:
        conv_storm_gdf = conv_storm_by_day_gdf
    else:
        conv_storm_gdf = pd.concat([conv_storm_gdf, conv_storm_by_day_gdf], ignore_index=True)

print('All 8 days of forecast of convective storms combined into a single gdf')

def format_issue_time(x):
    day = floor(x / 10000) * 10000
    quarter_day_hour = ceil((x % 10000) / 600) * 600 ## {24, 18, 12, 6}
    if (quarter_day_hour == 2400):
        day = day + 10000
        quarter_day_hour = 0
    return day + quarter_day_hour

conv_storm_issue_time = int(conv_storm_gdf['ISSUE'].max())
conv_storm_issue_time = format_issue_time(conv_storm_issue_time)
conv_storm_start_time = int(conv_storm_gdf['VALID'].min())
conv_storm_end_time = int(conv_storm_gdf['EXPIRE'].max())
conv_issue_time_dict = {'ISSUE_TIME' : conv_storm_issue_time, 'START_TIME' : conv_storm_start_time, 'END_TIME' : conv_storm_end_time}


## Collapse the outlook day (combine the shapes by category)
conv_agg_gdf = conv_storm_gdf[['LABEL','geometry']].dissolve(by='LABEL')
conv_agg_gdf = conv_agg_gdf.reset_index()
conv_agg_gdf = conv_agg_gdf.rename(columns={'LABEL':'category'})
conv_agg_gdf = pd.merge(conv_agg_gdf,conv_cat_map,how='left',on='category')
print('covective storm gdf dissolved by category')


## Sort storm data by severity
conv_final_gdf = conv_agg_gdf.sort_values(by='event_index')
conv_final_gdf = conv_final_gdf[conv_final_gdf['event_index'] > 0].reset_index(drop=True)
print('Convective storm shapes sorted by event index')

# ## Columns to keep
# county_cols_to_keep = ['NAMELSAD','STUSPS','geometry']
# zip_cols_to_keep = ['STD_ZIP5','STUSPS','COUNTYNAME','geometry']

## Columns to keep
county_cols_to_keep = ['NAMELSAD']
zip_cols_to_keep = ['STD_ZIP5']
print('County and zipcode to keep')

## Find which counties & zipcodes are affected by convective storms in the next 8 days
conv_county_membership = get_storm_table(conv_final_gdf,county_centroids,county_cols_to_keep)
conv_zip_membership = get_storm_table(conv_final_gdf,zip_code_centroids,zip_cols_to_keep)
print('County and zipcode convective storm membership table created')

## Winter Storm Index ##
# wint_cat_map0 = pd.DataFrame({'event_index': [0],
#                          'category': ['N/S']})
# wint_cat_map1 = pd.DataFrame({'event_index': np.arange(1,6),
#                          'category': ['LIMITED','MINOR','MODERATE','MAJOR','EXTREME']})
# wint_cat_map = pd.concat([wint_cat_map0,wint_cat_map1],ignore_index=True)



import time
from datetime import datetime, timezone

def round_to_quarter_day(dt):
    # Round the hour to the nearest 0, 6, 12, or 18
    rounded_hour = (dt.hour // 6) * 6
    return dt.replace(hour=rounded_hour, minute=0, second=0, microsecond=0)

# Current time in UTC from time.time()
current_utc_time = datetime.now(timezone.utc)

# Round to the closest quarter day in UTC
rounded_utc_datetime = round_to_quarter_day(current_utc_time)

# Format to 'YYYYMMDDHHMM'
formatted_utc_time = rounded_utc_datetime.strftime('%Y%m%d%H%M')
print(formatted_utc_time)
script_execution_time = formatted_utc_time


## Write out the combined shapes of convective storms
conv_final_gdf = add_issue_times(conv_final_gdf,conv_issue_time_dict)
conv_storm_file_name_arch = 'data_archive/convective_storm_shapes_issue_' + str(202303311630) + '.geojson'
conv_final_gdf.to_file(conv_storm_file_name_arch, index=False, driver='GeoJSON')
print('convective storms shapes written out')

## Merge storm categories into blocks
conv_county_membership = pd.merge(conv_county_membership,conv_cat_map,how='left',on='event_index')
conv_zip_membership = pd.merge(conv_zip_membership,conv_cat_map,how='left',on='event_index')
print('Category merged into membership using event_index')

## Write out the impact level for blocks
conv_county_membership = add_issue_times(conv_county_membership,conv_issue_time_dict)
conv_county_file_name_arch = 'data_archive/convective_storm_county_issue_' + str(202303311630) + '.csv'
conv_county_membership.to_csv(conv_county_file_name_arch, index=False)

conv_zip_membership = add_issue_times(conv_zip_membership,conv_issue_time_dict)
conv_zip_file_name_arch = 'data_archive/convective_storm_zipcode_issue_' + str(202303311630) + '.csv'
conv_zip_membership.to_csv(conv_zip_file_name_arch, index=False)
print('county and zipcode convective storm membership written out')


