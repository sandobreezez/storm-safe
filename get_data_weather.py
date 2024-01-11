import re
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
#import matplotlib.pyplot as plt
from itertools import chain
from shapely.geometry import Point
from shapely.geometry import LineString
import alphashape


## Load static files
county_centroids = gpd.read_file('data/block data/county_centroids.geojson')
zip_code_centroids = gpd.read_file('data/block data/zip_code_centroids.geojson')

conv_base_url_123 = 'https://www.spc.noaa.gov/products/outlook/day2otlk_cat.nolyr.geojson'
conv_geo_links_123 = [conv_base_url_123.replace('2', str(i)) for i in range(1, 4)]

conv_base_url_45678 = 'https://www.spc.noaa.gov/products/exper/day4-8/day5prob.nolyr.geojson'
conv_geo_links_45678 = [conv_base_url_45678.replace('5', str(i)) for i in range(4, 9)]

print(conv_geo_links_123)
print(conv_geo_links_45678)

conv_storm_urls = conv_geo_links_123 + conv_geo_links_45678

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
    mutually_exclusive = np.max(modified_values, axis=1)

    final_df = block_file[my_cols_keep].copy()
    final_df['event_index'] = mutually_exclusive

    return final_df

## Convective Storm Event Index ##
conv_cat_map0 = pd.DataFrame({'event_index': [0],
                         'category': ['N/S']})
conv_cat_map1 = pd.DataFrame({'event_index': np.arange(1,7),
                         'category': ['TSTM','MRGL','SLGT','ENH','MDT','HIGH']})
conv_cat_map = pd.concat([conv_cat_map0,conv_cat_map1],ignore_index=True)

## Combine all convective storm geojsons into a single gdf
for idx,link in enumerate(conv_storm_urls):
    conv_storm_by_day_gdf = gpd.read_file(link)
    conv_storm_by_day_gdf['outlook_day'] = idx + 1
    if idx == 0:
        conv_storm_gdf = conv_storm_by_day_gdf
    else:
        conv_storm_gdf = pd.concat([conv_storm_gdf, conv_storm_by_day_gdf], ignore_index=True)

## Collapse the outlook day (combine the shapes by category)
conv_agg_gdf = conv_storm_gdf[['LABEL','geometry']].dissolve(by='LABEL')
conv_agg_gdf = conv_agg_gdf.reset_index()
conv_agg_gdf = conv_agg_gdf.rename(columns={'LABEL':'category'})
conv_agg_gdf = pd.merge(conv_agg_gdf,conv_cat_map,how='left',on='category')

## Sort storm data by severity
conv_final_gdf = conv_agg_gdf.sort_values(by='event_index')
conv_final_gdf = conv_final_gdf[conv_final_gdf['event_index'] > 0].reset_index(drop=True)

## Find which counties & zipcodes are affected by convective storms in the next 8 days
conv_count_membership = get_storm_table(conv_final_gdf,county_centroids,['NAMELSAD','STUSPS','geometry'])
conv_zip_membership = get_storm_table(conv_final_gdf,zip_code_centroids,['STD_ZIP5','STUSPS','geometry'])

## Write out the combined shapes of convective storms
conv_final_gdf.to_file('convective_storm_shapes.geojson', index=False, driver='GeoJSON')

## Write out the impact level for blocks
conv_count_membership.to_file('convective_storm_county.geojson', index=False, driver='GeoJSON')
conv_zip_membership.to_file('convective_storm_zipcode.geojson', index=False, driver='GeoJSON')


## Winter Storm Index ##
wint_cat_map0 = pd.DataFrame({'event_index': [0],
                         'category': ['N/S']})
wint_cat_map1 = pd.DataFrame({'event_index': np.arange(1,6),
                         'category': ['LIMITED','MINOR','MODERATE','MAJOR','EXTREME']})
wint_cat_map = pd.concat([wint_cat_map0,wint_cat_map1],ignore_index=True)

## Winter Storm overall shapes
wint_storm_gdf = gpd.read_file('data/winter storm data/WSSI_OVERALL_Days_1_3_latest/WSSI_OVERALL_Days_1_3.shp')
wint_storm_gdf = wint_storm_gdf.to_crs(epsg=4326) # Reproject to WGS84
wint_agg_gdf = wint_storm_gdf[['IMPACT','geometry']].rename(columns={'IMPACT':'category'})
wint_agg_gdf = pd.merge(wint_agg_gdf,wint_cat_map,how='left',on='category')

## Sort by severity
wint_final_gdf = wint_agg_gdf.sort_values(by='event_index')
wint_final_gdf = wint_final_gdf[wint_final_gdf['event_index'] > 0].reset_index(drop=True)

def reduce_shapes(multi_poly):
    return alphashape.alphashape(extract_boundary_points(multi_poly), 5)

def extract_boundary_points(multi_poly):
    # Function to extract boundary points from a polygon
    def extract_boundary_points_poly(polygon):
        # Include exterior boundary points
        points = list(polygon.exterior.coords)

        # Include interior (holes) boundary points if any
        for interior in polygon.interiors:
            points.extend(interior.coords)

        return points

    if multi_poly.geom_type == 'MultiPolygon':
        # List to store all boundary points of all polygons in the MultiPolygon
        all_boundary_points = []
        # Iterate through each polygon in the MultiPolygon
        for polygon in multi_poly.geoms:
            all_boundary_points.extend(extract_boundary_points(polygon))
        return all_boundary_points
    elif multi_poly.geom_type == 'Polygon':
        return extract_boundary_points_poly(multi_poly)
    else:
        return TypeError


wint_final_gdf_reduced = wint_final_gdf.copy()
wint_final_gdf_reduced['geometry'] = wint_final_gdf_reduced['geometry'].apply(reduce_shapes)

## Find which counties & zipcodes are affected by convective storms in the next 8 days
wint_count_membership = get_storm_table(wint_final_gdf_reduced,county_centroids,['NAMELSAD','STUSPS','geometry'])
wint_zip_membership = get_storm_table(wint_final_gdf_reduced,zip_code_centroids,['STD_ZIP5','STUSPS','geometry'])

## Write out the combined shapes of convective storms
wint_final_gdf_reduced.to_file('winter_storm_shapes.geojson', index=False, driver='GeoJSON')

## Write out the impact level for blocks
wint_count_membership.to_file('winter_storm_county.geojson', index=False, driver='GeoJSON')
wint_zip_membership.to_file('winter_storm_zipcode.geojson', index=False, driver='GeoJSON')
print('done')