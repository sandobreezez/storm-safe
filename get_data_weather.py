import re
import requests
import zipfile
import os
import time
from math import ceil, floor
import numpy as np
import pandas as pd
import geopandas as gpd
#import matplotlib.pyplot as plt
from itertools import chain
from shapely.geometry import Point
from shapely.geometry import LineString
import alphashape

if os.getenv('VIRTUAL_ENV') == None:
    os.chdir('/home/sandobreezez/storm-safe/')
## Storm 'category' ordered by 'event_index'
conv_cat_map = pd.read_csv('convective_storm_cat_map.csv')
wint_cat_map = pd.read_csv('winter_storm_cat_map.csv')
print('Ordered storm categories loaded')

## Load static files
county_centroids = gpd.read_file('county_centroids.geojson')
zip_code_centroids = gpd.read_file('zipcode_centroids.geojson')

conv_base_url_123 = 'https://www.spc.noaa.gov/products/outlook/day2otlk_cat.nolyr.geojson'
conv_geo_links_123 = [conv_base_url_123.replace('2', str(i)) for i in range(1, 4)]

conv_base_url_45678 = 'https://www.spc.noaa.gov/products/exper/day4-8/day5prob.nolyr.geojson'
conv_geo_links_45678 = [conv_base_url_45678.replace('5', str(i)) for i in range(4, 9)]

#conv_storm_urls = conv_geo_links_123 + conv_geo_links_45678
conv_storm_urls = conv_geo_links_123
print('List of conv links created')


def get_storm_table(storm_gdf,block_file,my_cols_keep):
    storm_gdf = storm_gdf.explode(index_parts=False, ignore_index=True)
    membership = gpd.tools.sjoin(block_file[my_cols_keep + ['geometry']],storm_gdf, predicate="within", how='left')
    membership = membership[membership['event_index'] >= 1][my_cols_keep + ['category','event_index']]
    return membership

def add_issue_times(my_df,time_dict):
    for time_item in time_dict:
        my_df[time_item] = time_dict[time_item]
    return my_df

def make_geometry_mutually_exclusive(geoseries):
    # Find number of rows
    nrows = len(geoseries.index)

    # Combine shapes
    for i in range(0,nrows):
        idx = nrows - i - 1 # we will combine in reverse order so the most severe cat is by itself and the least is inclusive of all
        if idx + 1 == nrows:
            next # ignore the most severe cat
        else:
            geoseries[idx] = geoseries[idx].union(geoseries[idx+1])

    for i in range(0, nrows - 1):
        idx = i  # we will subtract cat(i + 1) from cat(i)
        geoseries[idx] = geoseries[idx].difference(geoseries[idx + 1])

    return geoseries

## Combine all convective storm geojsons into a single gdf
for idx,link in enumerate(conv_storm_urls):
    conv_storm_by_day_gdf = gpd.read_file(link)
    conv_storm_by_day_gdf = conv_storm_by_day_gdf.to_crs(epsg=4326)  # Reproject to WGS84
    conv_storm_by_day_gdf['outlook_day'] = idx + 1
    if idx == 0:
        conv_storm_gdf = conv_storm_by_day_gdf
    else:
        conv_storm_gdf = pd.concat([conv_storm_gdf, conv_storm_by_day_gdf], ignore_index=True)

print('All 3 days of forecast of convective storms combined into a single gdf')

def format_issue_time(x):
    x = int(x)
    day = floor(x / 10000) * 10000
    quarter_day_hour = ceil((x % 10000) / 600) * 600 ## {24, 18, 12, 6}
    if (quarter_day_hour == 2400):
        day = day + 10000
        quarter_day_hour = 0
    return day + quarter_day_hour


conv_storm_issue_time = format_issue_time(int(conv_storm_gdf['ISSUE'].max()))
conv_storm_start_time = format_issue_time(int(conv_storm_gdf['VALID'].min()))
conv_storm_end_time = format_issue_time(int(conv_storm_gdf['EXPIRE'].max()))
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

## Columns to keep
county_cols_to_keep = ['NAMELSAD','STUSPS']
zip_cols_to_keep = ['STD_ZIP5','STUSPS']
print('County and zipcode to keep')

conv_final_gdf['geometry'] = make_geometry_mutually_exclusive(conv_final_gdf['geometry'])

conv_county_membership = get_storm_table(conv_final_gdf,county_centroids,county_cols_to_keep)
conv_zip_membership = get_storm_table(conv_final_gdf,zip_code_centroids,zip_cols_to_keep)
print('County and zipcode convective storm membership table created')

# URL of the file to be downloaded
wint_data_url = 'https://origin.wpc.ncep.noaa.gov/wwd/wssi/gis/shp/WSSI_OVERALL_Days_1_3_latest.zip'

# Local path where you want to save the downloaded file
wint_data_save_path = 'winter_storm_data/WSSI_OVERALL_Days_1_3_latest.zip'

# Send a GET request to the URL
response = requests.get(wint_data_url)

# Check if the request was successful
if response.status_code == 200:
    # Open the file in write-binary mode and save the content
    with open(wint_data_save_path, 'wb') as file:
        file.write(response.content)
    print("File downloaded and saved successfully!")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")


# Specify the path to your zip file
zip_file_path = 'winter_storm_data/WSSI_OVERALL_Days_1_3_latest.zip'

# Specify the directory to extract the files into
extract_to_directory = 'winter_storm_data/latest_unzipped'

# Opening the zip file in read mode
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extracting all the contents of the zip file
    zip_ref.extractall(extract_to_directory)

# Specify the path of the file to be deleted
file_path = 'winter_storm_data/WSSI_OVERALL_Days_1_3_latest.zip'

# Check if file exists
if os.path.exists(file_path):
    # Delete the file
    os.remove(file_path)
    print("File deleted successfully")
else:
    print("The file does not exist")

def convert_timestamp(input_string):
    if input_string[-1] != 'Z':
        # Rearrange the string to 'YYYY-MM-DD HHMM'
        input_string = f'20{input_string[-2:]}-{input_string[4:6]}-{input_string[7:9]} {input_string[:2]}00'
    # Use regex to remove non-digit characters
    formatted_string = re.sub(r'\D', '', input_string)
    return formatted_string

wint_storm_gdf = gpd.read_file('winter_storm_data/latest_unzipped/WSSI_OVERALL_Days_1_3.shp')
wint_storm_gdf = wint_storm_gdf.to_crs(epsg=4326) # Reproject to WGS84
#wint_storm_issue_time = int(re.sub(r'\D', '', wint_storm_gdf['ISSUE_TIME'].iloc[0])) ## keep only digits
wint_storm_issue_time = format_issue_time(int(convert_timestamp(wint_storm_gdf['ISSUE_TIME'].iloc[0])))
wint_storm_start_time = format_issue_time(int(convert_timestamp(wint_storm_gdf['START_TIME'].iloc[0])))
wint_storm_end_time = format_issue_time(int(convert_timestamp(wint_storm_gdf['END_TIME'].iloc[0])))
wint_issue_time_dict = {'ISSUE_TIME' : wint_storm_issue_time, 'START_TIME' : wint_storm_start_time, 'END_TIME' : wint_storm_end_time}


## Winter Storm overall shapes
wint_agg_gdf = wint_storm_gdf[['IMPACT','geometry']].rename(columns={'IMPACT':'category'})
wint_agg_gdf = pd.merge(wint_agg_gdf,wint_cat_map,how='left',on='category')
print('Grab overall winter shapes')

## Sort by severity
wint_final_gdf = wint_agg_gdf.sort_values(by='event_index')
wint_final_gdf = wint_final_gdf[wint_final_gdf['event_index'] > 0].reset_index(drop=True)
print('Sort winter storm by severity')

polygon_count = 0
for geom in wint_final_gdf.geometry:
    if geom.geom_type == 'MultiPolygon':
        # Count the number of polygons in each MultiPolygon
        polygon_count += len(list(geom.geoms))
print('Number of polygons in winter forecast: ')
print(polygon_count)


def gen_rand_poly_v1(multi_poly_expl_gdf,num_points_per_area=1000):
    # generates random points in the bounds of a multipolygon
    # for any given random point, the chance that it is in polygon(i) should be propotional to area(i)
    # We create points for the entire multipolygon because the computation is too slow when doing it on individual polygons
    total_area = multi_poly_expl_gdf['area'].sum()
    multi_poly_filt = multi_poly_expl_gdf.geometry.unary_union
    multi_poly_gdf_filt = gpd.GeoDataFrame({'geometry': [multi_poly_filt]})

    num_points = min(int(total_area * num_points_per_area),1000000) # if we get a few really large shapes it may slow things down
    sample_points = multi_poly_gdf_filt.geometry.sample_points(size=num_points).explode().tolist()
    sample_points_tuple = [point.coords[0] for point in sample_points]
    return sample_points_tuple

def gen_rand_poly_v2(multi_poly,area_thresh=1,num_points_per_area=1000):
    # generates random points in the bounds of a multipolygon using the minimum bounding rectangle of the entire multipolygon
    multi_poly_gdf = gpd.GeoDataFrame({'geometry': [multi_poly]})
    multi_poly_expl_gdf = multi_poly_gdf.explode(ignore_index=True)
    multi_poly_expl_gdf['area'] = multi_poly_expl_gdf['geometry'].area
    multi_poly_expl_gdf = multi_poly_expl_gdf[multi_poly_expl_gdf['area'] >= area_thresh]
    print(multi_poly_expl_gdf)

    if multi_poly_expl_gdf.empty:
        return []
    min_bound_rect = multi_poly_expl_gdf.unary_union.envelope
    min_bound_rect_area = int(min_bound_rect.area)
    minx, miny, maxx, maxy = min_bound_rect.bounds

    num_points = min_bound_rect_area * num_points_per_area
    x = np.random.uniform(minx, maxx, num_points)
    y = np.random.uniform(miny, maxy, num_points)
    df = pd.DataFrame()
    df['points'] = list(zip(x, y))
    df['points'] = df['points'].apply(Point)
    gdf_points = gpd.GeoDataFrame(df, geometry='points')
    Sjoin = gpd.tools.sjoin(gdf_points, multi_poly_expl_gdf, predicate="within", how='left')
    sample_points = gdf_points[Sjoin.index_right.isna() == False].geometry.tolist()
    sample_points_tuple = [point.coords[0] for point in sample_points]
    # This can be improved to reduce the randomness of which polygons end up with simulated points
    # The idea would be to generate a certain number of uniform_random points from 0 to 1
    # We would then find the minimum bounding rectangle (mbr) for each polygon (this is a relatively fast computation)
    # Find the number of points to generate for each shape, ie: mbr.area(i)/shape.area(i) * num_points_per_area // 1
    # generate and assign simulation to each shape
    # tranform the simulated point, ie: x * (shape.maxx(i) - shape.minx(i)) + shape.minx(i), this can be done in pandas super fast
    # sjoin with the simulated points with the shapes
    # the speed should be similar to this function but each shape will have a guaranteed chance to include a certain number of points
    return sample_points_tuple

def extract_exterior_points(expl_gdf):
    return expl_gdf['geometry'].apply(lambda poly: (poly.exterior.coords)).explode(ignore_index=True).tolist()

def extract_interior_points(expl_gdf):
    def get_interior_coords(poly):
        coords = []
        for interior in poly.interiors:
            coords.extend(interior.coords)
        return coords
    interior_points = expl_gdf['geometry'].apply(get_interior_coords).explode(ignore_index=True).tolist()
    interior_points = [point for point in interior_points if str(point) != 'nan']
    return interior_points

def extract_centroid_points(expl_gdf):
    expl_gdf['x'] = expl_gdf.geometry.centroid.apply(lambda p: p.x) # p for point
    expl_gdf['y'] = expl_gdf.geometry.centroid.apply(lambda p: p.y)
    centroid_points = [(x,y) for x,y in zip(expl_gdf.x , expl_gdf.y)]
    return centroid_points

def extract_all_points(multi_poly,area_cut_off=.0001):
    # The reduce_shapes function takes in a list of tuple points as its main arguement to redraw the shapes
    # The complete list of points we provide includes the exterior and interior perimeter and centroid points which are trivial
    # The simulated points are generated to preserve some form of the original shape that would be otherwise detroyed when submitting
    # only the perimeter/centroid points. In some case the algorithm would draw a shape that includes some perimeter points but not all
    # and kind of do this around the shape. Instead of looking like a complete polygon, it would look more like a broken up fence.
    multi_poly_gdf = gpd.GeoDataFrame({'geometry': [multi_poly]})
    multi_poly_expl_gdf = multi_poly_gdf.explode(ignore_index=True)
    multi_poly_expl_gdf['area'] = multi_poly_expl_gdf['geometry'].area
    multi_poly_expl_gdf = multi_poly_expl_gdf[multi_poly_expl_gdf['area'] >= area_cut_off]


    simulated_points = gen_rand_poly_v1(multi_poly_expl_gdf,num_points_per_area=1000)
    centroid_points = extract_centroid_points(multi_poly_expl_gdf)
    exterior_points = extract_exterior_points(multi_poly_expl_gdf)
    # The inclusion of interior points can be argued, we can consider removing this.
    interior_points = extract_interior_points(multi_poly_expl_gdf)

    all_points = simulated_points + centroid_points + exterior_points + interior_points

    return all_points

def reduce_shapes(multi_poly):
    # The idea behind this is to reduce the complexity of the geojson for a few reasons
    # 1. Currently we find use centroid of counties & zipcodes because it would be too computationally expensive to use zipcode shapes.
    #    So if we have a bunch of sparse shapes in and around a county/zipcode, we could still miss the centroid.
    #    This means that for an otherwise affected county, we may have completely underreported the severity. No bueno..
    # 2. It decreases the file size substaintially, so it is effiecient in terms of load speed and storage space.
    # 3. In my opinion it looks better when displayed in a plot and it is perhaps easier to draw conclusions from.
    return alphashape.alphashape(extract_all_points(multi_poly), 5)

start_time = time.time()
wint_final_gdf_reduced = wint_final_gdf.copy()
wint_final_gdf_reduced['geometry'] = wint_final_gdf_reduced['geometry'].apply(reduce_shapes)
wint_final_gdf_reduced['geometry'] = make_geometry_mutually_exclusive(wint_final_gdf_reduced['geometry'])
print("--- %s seconds ---" % (time.time() - start_time))


#wint_final_gdf['geometry'] = make_geometry_mutually_exclusive(wint_final_gdf['geometry'])

polygon_count = 0
for geom in wint_final_gdf_reduced.geometry:
    if geom.geom_type == 'MultiPolygon':
        # Count the number of polygons in each MultiPolygon
        polygon_count += len(list(geom.geoms))
print('Number of polygons in winter forecast reduced: ')
print(polygon_count)

## Find which counties & zipcodes are affected by convective storms in the next 8 days
wint_county_membership = get_storm_table(wint_final_gdf_reduced,county_centroids,county_cols_to_keep)
wint_zip_membership = get_storm_table(wint_final_gdf_reduced,zip_code_centroids,zip_cols_to_keep)
print('Find which counties & zipcodes are affected by convective storms in the next 3 days')


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
conv_final_gdf.to_file('convective_storm_shapes.geojson', index=False, driver='GeoJSON')

conv_storm_file_name_arch = 'data_archive/convective_storm_shapes_issue_' + str(script_execution_time) + '.geojson'
conv_final_gdf.to_file(conv_storm_file_name_arch, index=False, driver='GeoJSON')
print('convective storms shapes written out')

## Write out the impact level for blocks
conv_county_membership.to_csv('convective_storm_county.csv', index=False)
conv_county_file_name_arch = 'data_archive/convective_storm_county_issue_' + str(script_execution_time) + '.csv'
conv_county_membership.to_csv(conv_county_file_name_arch, index=False)

conv_zip_membership.to_csv('convective_storm_zipcode.csv', index=False)
conv_zip_file_name_arch = 'data_archive/convective_storm_zipcode_issue_' + str(script_execution_time) + '.csv'
conv_zip_membership.to_csv(conv_zip_file_name_arch, index=False)
print('county and zipcode convective storm membership written out')


## Write out the combined shapes of convective storms
wint_final_gdf_reduced = add_issue_times(wint_final_gdf_reduced,wint_issue_time_dict)
wint_final_gdf_reduced.to_file('winter_storm_shapes.geojson', index=False, driver='GeoJSON')

wint_storm_file_name_arch = 'data_archive/winter_storm_shapes_issue_' + str(script_execution_time) + '.geojson'
wint_final_gdf_reduced.to_file(wint_storm_file_name_arch, index=False, driver='GeoJSON')
print('winter storms shapes written out')

## Write out the impact level for blocks
wint_county_membership.to_csv('winter_storm_county.csv', index=False)
wint_county_file_name_arch = 'data_archive/winter_storm_county_issue_' + str(script_execution_time) + '.csv'
wint_county_membership.to_csv(wint_county_file_name_arch, index=False)

wint_zip_membership.to_csv('winter_storm_zipcode.csv', index=False)
wint_zip_file_name_arch = 'data_archive/winter_storm_zipcode_issue_' + str(script_execution_time) + '.csv'
wint_zip_membership.to_csv(wint_zip_file_name_arch, index=False)
print('county and zipcode winter storm membership written out')

print('done')