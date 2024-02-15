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
county_centroids = gpd.read_file('county_centroids.geojson')
zip_code_centroids = gpd.read_file('zipcode_centroids.geojson')

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


## Columns to keep
county_cols_to_keep = ['NAMELSAD','STUSPS']
zip_cols_to_keep = ['STD_ZIP5','STUSPS']
print('County and zipcode to keep')

directory_path = 'data_archive'
# List all files and directories in the specified path
entries = os.listdir(directory_path)
# Filter out directories, keep only files
file_names = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
conv_shapes_file_names = [file for file in file_names if 'convective_storm_shapes' in file]
wint_shapes_file_names = [file for file in file_names if 'winter_storm_shapes' in file]

def load_files_to_dict_list(file_list,directory=''):
    loaded_file_dict = {}
    for file_name in file_list:
        file_path = directory + '/' + file_name
        if '.csv' in file_name:
            key = file_name.replace('.csv', '')
            loaded_file_dict[key] = pd.read_csv(file_path)
            if 'STD_ZIP5' in loaded_file_dict[key].columns:
                loaded_file_dict[key]['STD_ZIP5'] = loaded_file_dict[key]['STD_ZIP5'].astype(str).str.zfill(5)
        elif '.geojson' in file_name:
            key = file_name.replace('.geojson', '')
            loaded_file_dict[key] = gpd.read_file(file_path)
    return loaded_file_dict

conv_archive_dict = load_files_to_dict_list(conv_shapes_file_names,directory_path)
conv_archive_dict.keys()

for conv_final_gdf_name in conv_archive_dict:
    #conv_final_gdf_name = 'convective_storm_shapes_issue_202401300600'
    conv_final_gdf = conv_archive_dict[conv_final_gdf_name]
    run_time = conv_final_gdf_name.split('issue_')[1]
    print(conv_final_gdf_name)

    if 'ISSUE_TIME' not in conv_final_gdf.columns:
        print('delete')
        print(conv_final_gdf_name)
        os.remove('data_archive/convective_storm_shapes_issue_' + run_time + '.geojson')
        os.remove('data_archive/convective_storm_county_issue_' + run_time + '.csv') if os.path.isfile('data_archive/convective_storm_county_issue_' + run_time + '.csv') else print('county file not found')
        os.remove('data_archive/convective_storm_zipcode_issue_' + run_time + '.csv') if os.path.isfile('data_archive/convective_storm_zipcode_issue_' + run_time + '.csv') else print('zipcode file not found')
        print('files deleted')
        continue


    conv_issue_time_dict = {'ISSUE_TIME': conv_final_gdf['ISSUE_TIME'].iloc[0],
                            'START_TIME': conv_final_gdf['START_TIME'].iloc[0],
                            'END_TIME': conv_final_gdf['END_TIME'].iloc[0]}

    ## Find which counties & zipcodes are affected by convective storms in the next 8 days
    conv_county_membership = get_storm_table(conv_final_gdf,county_centroids,county_cols_to_keep)
    conv_zip_membership = get_storm_table(conv_final_gdf,zip_code_centroids,zip_cols_to_keep)
    print('County and zipcode convective storm membership table created')

    ## Merge storm categories into blocks
    conv_county_membership = pd.merge(conv_county_membership, conv_cat_map, how='left', on='event_index')
    conv_zip_membership = pd.merge(conv_zip_membership, conv_cat_map, how='left', on='event_index')
    print('Category merged into membership using event_index')

    ## Write out the impact level for blocks
    conv_county_membership = add_issue_times(conv_county_membership,conv_issue_time_dict)

    conv_county_file_name_arch = 'data_archive/convective_storm_county_issue_' + str(run_time) + '.csv'
    conv_county_membership.to_csv(conv_county_file_name_arch, index=False)

    conv_zip_membership = add_issue_times(conv_zip_membership,conv_issue_time_dict)

    conv_zip_file_name_arch = 'data_archive/convective_storm_zipcode_issue_' + str(run_time) + '.csv'
    conv_zip_membership.to_csv(conv_zip_file_name_arch, index=False)
    print('county and zipcode convective storm membership written out')



wint_archive_dict = load_files_to_dict_list(wint_shapes_file_names,directory_path)
wint_archive_dict.keys()

for wint_final_gdf_name in wint_archive_dict:
    #wint_final_gdf_name = 'winter_storm_shapes_issue_202401280600'
    wint_final_gdf_reduced = wint_archive_dict[wint_final_gdf_name]
    run_time = wint_final_gdf_name.split('issue_')[1]
    
    if 'ISSUE_TIME' not in wint_final_gdf_reduced.columns:
        print('delete')
        print(wint_final_gdf_name)
        os.remove('data_archive/winter_storm_shapes_issue_' + run_time + '.geojson')
        os.remove('data_archive/winter_storm_county_issue_' + run_time + '.csv') if os.path.isfile('data_archive/winter_storm_county_issue_' + run_time + '.csv') else print('county file not found')
        os.remove('data_archive/winter_storm_zipcode_issue_' + run_time + '.csv') if os.path.isfile('data_archive/winter_storm_zipcode_issue_' + run_time + '.csv') else print('zipcode file not found')
        print('files deleted')
        continue
    
    wint_issue_time_dict = {'ISSUE_TIME': wint_final_gdf_reduced['ISSUE_TIME'].iloc[0],
                            'START_TIME': wint_final_gdf_reduced['START_TIME'].iloc[0],
                            'END_TIME': wint_final_gdf_reduced['END_TIME'].iloc[0]}


    ## Find which counties & zipcodes are affected by convective storms in the next 8 days
    wint_county_membership = get_storm_table(wint_final_gdf_reduced,county_centroids,county_cols_to_keep)
    wint_zip_membership = get_storm_table(wint_final_gdf_reduced,zip_code_centroids,zip_cols_to_keep)
    print('Find which counties & zipcodes are affected by convective storms in the next 3 days')


    wint_county_membership = pd.merge(wint_county_membership,wint_cat_map,how='left',on='event_index')
    wint_county_membership = add_issue_times(wint_county_membership,wint_issue_time_dict)
    print('save latest county membership')

    wint_county_file_name_arch = 'data_archive/winter_storm_county_issue_' + str(run_time) + '.csv'
    wint_county_membership.to_csv(wint_county_file_name_arch, index=False)
    print('save archive county membership')

    wint_zip_membership = pd.merge(wint_zip_membership,wint_cat_map,how='left',on='event_index')
    wint_zip_membership = add_issue_times(wint_zip_membership,wint_issue_time_dict)
    print('save latest county membership')

    wint_county_file_name_arch = 'data_archive/winter_storm_zipcode_issue_' + str(run_time) + '.csv'
    wint_zip_membership.to_csv(wint_county_file_name_arch, index=False)
    print('save archive county membership')

    print('done')