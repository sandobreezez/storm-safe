import re
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from itertools import chain
from shapely.geometry import Point
from shapely.geometry import LineString

# Function to extract unique event types from the text
def extract_events(text):
    # Regular expression to match event lines
    event_pattern = re.compile(r"\.\.\.\s+(\w+)\s+\.\.\.")

    # Find event types
    event_matches = event_pattern.findall(text)

    # Get unique event types
    unique_events = list(set(event_matches))

    return unique_events

def extract_string(my_text, sub1, sub2):
    # getting index of substrings
    idx1 = my_text.index(sub1)
    print(idx1)
    idx2 = my_text[idx1:].index(sub2) + idx1
    print(idx2)

    # get string from next character
    output = my_text[idx1 + len(sub1) + 1: idx2]
    print(output)
    return output


# Load the text file from the provided URL



def filter_catprob(event, item):
    if event == 'CATEGORICAL':
        return item.isalpha()
    else:
        return item <= 1


def string_to_list(text):
    # takes all string and splits it by <space> character
    t = []
    for s in test.replace('\n', ' ').split(' '):
        try:
            t.append(float(s))
        except:
            if s.isalpha():
                t.append(s)
            else:
                t.append(-1)

    t2 = filter(lambda x: x != -1, t)
    t3 = list(t2)
    return t3


def create_table(event, my_list):
    # takes arguements event and my_list and returns a table with columns event, probability, latitude, longitude

    # gets the probabilities/categories and their idx for a given event
    probs = [(i, item) for i, item in enumerate(my_list) if filter_catprob(event, str(item))]
    prob_values = [p[1] for p in probs]
    prob_idx = [p[0] for p in probs]
    prob_idx.append(len(my_list))

    coordinate_list_list = []
    for i, p in enumerate(prob_values):
        start_idx = prob_idx[i] + 1
        end_idx = prob_idx[i + 1] - 1
        coordinate_list_list.append(my_list[start_idx:end_idx])

    prob_col = [[p] * len(coordinate_list_list[i]) for i, p in enumerate(prob_values)]
    prob_col = list(chain(*prob_col))
    ll_col = list(chain(*coordinate_list_list))

    lat_col = [float(int(p / 10000) / 100) for p in ll_col]
    long_col = [float(int(p % 10000) / 100) for p in ll_col]

    d = {'event': event, 'probability': prob_col, 'latitude': lat_col, 'longitude': long_col}
    return pd.DataFrame(data=d)

state_gdf = gpd.read_file('data/tl_2022_us_state/tl_2022_us_state.shp')
convex_hull = state_gdf[~state_gdf['NAME'].isin(['Alaska','Hawaii'])].unary_union.convex_hull
exterior_coords = list(convex_hull.exterior.coords)

texas = state_gdf.loc[state_gdf['NAME'] == 'Texas', 'geometry'].squeeze()
texas_points = list(zip(*texas.exterior.xy))

arizona = state_gdf.loc[state_gdf['NAME'] == 'Arizona', 'geometry'].squeeze()
arizona_points = list(zip(*arizona.exterior.xy))

new_mexico = state_gdf.loc[state_gdf['NAME'] == 'New Mexico', 'geometry'].squeeze()
new_mexico_points = list(zip(*new_mexico.exterior.xy))

single_gon = 0
multi_gon = 0
neither = 0
for index, row in state_gdf.iterrows():
    # 'index' is the row index
    # 'row' is a pandas Series containing data for that row
    if row['geometry'].geom_type == 'Polygon':
        single_gon += 1
    elif row['geometry'].geom_type == 'MultiPolygon':
        print(row['NAME'])
        print(len(list(row['geometry'].geoms)))
        multi_gon += 1
    else:
        neither += 1

print(single_gon)
print(multi_gon)
print(neither)

plt.clf()
for index, row in state_gdf.iterrows():
    # 'index' is the row index
    # 'row' is a pandas Series containing data for that row
    if row['geometry'].geom_type == 'Polygon' and row['NAME'] != 'Guam':
        my_state = row['geometry']
        x, y = my_state.exterior.xy
#        my_state_points = list(zip(*my_state.exterior.xy))
#        print(row['NAME'])
#        for xy in my_state_points:
#            # print(xy[0])
#            # print(xy[1])
#            x.append(xy[0])
#            y.append(xy[1])

        plt.plot(x, y, label=row['NAME'])
    #print(index, row)

# Add labels and title
plt.xlabel('longtitude')
plt.ylabel('latitude')
plt.title('Storm')
plt.legend()
# Show the plot
plt.show()

x, y = [], []
for xy in texas_points:
    #print(xy[0])
    #print(xy[1])
    x.append(xy[0])
    y.append(xy[1])

plt.plot(x, y, label = 'Texas border')


#new_mexico_points


#url = "https://www.spc.noaa.gov/products/outlook/archive/2023/KWNSPTSDY1_202304160100.txt"
url = "https://www.spc.noaa.gov/products/outlook/archive/2023/KWNSPTSDY1_202310251200.txt"
response = requests.get(url)
text = response.text

event = 'CATEGORICAL'
event_types = extract_events(text)
test = extract_string(text, ' ' + event, '&&')

test2 = string_to_list(test)

test3 = create_table(event, test2)

test3['coord_sum'] = test3['latitude'] + test3['longitude']

test4 = test3.assign(shape_break=lambda x: x['coord_sum'].apply(lambda y: 0 if y < 199 else 1))

test5 = test4.assign(
    shape=test4.groupby(['event', 'probability'], group_keys=False).apply(lambda g: g.shape_break.cumsum()) + 1)

test5['shape_key'] = test5['event'] + test5['probability'] + test5['shape'].astype(str)

test6 = test5.loc[test5['shape_break'] != 1]

test7 = test6.assign(longitude=lambda x: x['longitude'].apply(lambda y: y + 100 if y < 26 else y))

import matplotlib.pyplot as plt

# Sample data (replace with your own x and y values)
plt.clf()
x, y = [], []
for xy in texas_points:
    #print(xy[0])
    #print(xy[1])
    x.append(xy[0])
    y.append(xy[1])

plt.plot(x, y, label = 'Texas border')

x, y = [], []
for xy in new_mexico_points:
    #print(xy[0])
    #print(xy[1])
    x.append(xy[0])
    y.append(xy[1])
#
# plt.plot(x, y, label = 'New Mexico border')
#
# #plt.clf()
# x, y = [], []
# for xy in arizona_points:
#     #print(xy[0])
#     #print(xy[1])
#     x.append(xy[0])
#     y.append(xy[1])
#
# plt.plot(x, y, label = 'Arizona border')
#
x,y = [], []
for shape_key in list(set(test7['shape_key'])):
    x = list(test7.loc[test7['shape_key'] == shape_key]['longitude'] * -1)
    y = list(test7.loc[test7['shape_key'] == shape_key]['latitude'])

# Create a basic line plot
    plt.plot(x, y, label = shape_key)
#
# # Add labels and title
# plt.xlabel('longtitude')
# plt.ylabel('latitude')
# plt.title('Storm')
# plt.legend()
# # Show the plot
# plt.show()
#
#
# # Assuming your DataFrame is called df
# storm_geometry = test7.groupby('shape_key').apply(lambda x: LineString(x[['longitude', 'latitude']].values))
# storm_gdf = gpd.GeoDataFrame(storm_geometry.groupby('shape_key').first(), geometry=geometry)
#
# # Print the extracted information
# print("Probabilistic Outlook Information:")
# for row in test6:
#     print(row)




# Replace 'counties.geojson' with the path to your data file
gdf = gpd.read_file('data/tl_2022_us_county/tl_2022_us_county.shp')

county_centroids = gdf[['STATEFP','COUNTYFP','GEOID','NAME','NAMELSAD','LSAD','INTPTLAT','INTPTLON']].copy()
county_centroids.loc[:, 'INTPTLAT'] = county_centroids['INTPTLAT'].astype(float)
county_centroids.loc[:, 'INTPTLON'] = county_centroids['INTPTLON'].astype(float)

# Assuming your DataFrame is called `county_centroids`
county_centroids['geometry'] = county_centroids.apply(lambda row: Point(row['INTPTLON'], row['INTPTLAT']), axis=1)

# Now you have a GeoDataFrame with the 'geometry' column containing Point geometries.
county_centroids = gpd.GeoDataFrame(county_centroids[['STATEFP','COUNTYFP','GEOID','NAME','NAMELSAD','LSAD','geometry']].copy(), geometry='geometry')

#county_centroids['TSTM'] = 0
#county_centroids['MRGL'] = 0

geojson = "https://www.spc.noaa.gov/products/outlook/day1otlk_cat.nolyr.geojson"
storm_gdf = gpd.read_file(geojson)

# Create an empty DataFrame with a column for each polygon
result_df = gpd.GeoDataFrame()
for idx, polygon in enumerate(storm_gdf['geometry']):
    result_df[f'polygon_{idx + 1}'] = county_centroids['geometry'].within(polygon).astype(int)

result_df.index = county_centroids.index

# Multiply each value by the column index + 1
modified_values = result_df.values * np.arange(1, len(result_df.columns) + 1)

# Use np.argmax to find the maximum column index for each row
mutually_exclusive = np.max(modified_values, axis=1)

result_df['mutually_exclusive'] = mutually_exclusive

result_df['mutually_exclusive'] = np.argmax(result_df.values, axis=1)
result_df['mutually_exclusive'] = result_df.apply(lambda row: row.index[row == 1].max(), axis=1).str.replace('polygon_', '').astype(int)


import pandas as pd

# Sample DataFrame with columns 'col1', 'col2', 'col3', etc.
data = {
    'col1': [0, 1, 0],
    'col2': [1, 0, 1],
    'col3': [0, 0, 1],
    # Add more columns as needed
}

df = pd.DataFrame(data)

# Create a new column 'mutually_exclusive'
df['mutually_exclusive'] = df.apply(lambda row: row.index[row == 1].max(), axis=1).str.replace('col', '').astype(int)

print(df)

result_df['CATEGORY'] = result_df.apply(lambda row: row['']

result_df = gpd.GeoDataFrame()
for idx, row in storm_gdf.iterrows():
    name, polygon = row['LABEL'], row['geometry']
    result_df[name] = county_centroids['geometry'].within(polygon).astype(int)
# Make sure the result GeoDataFrame has the same index as the points GeoDataFrame
result_df.index = county_centroids.index

county_centroids = pd.concat([county_centroids, result_df], axis=1)

county_centroids = county_centroids['longitude'].apply(lambda y: y + 100 if y < 26 else y))


for index1, row1 in storm_gdf.iterrows():
    county_centroids[row1['LABEL']] = 0
    for index2, row2 in county_centroids.iterrows():
        if row1['geometry'].contains(row2['centroid_point']) == True:
            county_centroids[row1['LABEL']].iloc[index2] = 1


# Define a suitable projected CRS (supposedly this is what US census uses)
projected_crs = 'EPSG:4269'

# Reproject the GeoDataFrame to the projected CRS
gdf['geometry'] = gdf['geometry'].to_crs(projected_crs)
gdf['centroid'] = gdf['geometry'].centroid


import re
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from itertools import chain
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely import wkt
# Replace 'counties.geojson' with the path to your data file
#gdf = gpd.read_file('data/zip_code_coordinates/ZIP_Code_Population_Weighted_Centroids.geojson')

# Load the CSV file as a pandas DataFrame
df = pd.read_csv('total_outlook_shapes.csv')

# Convert the 'geometry' column from WKT to Shapely geometries
df['geometry'] = df['geometry'].apply(wkt.loads)

# Convert the DataFrame to a GeoDataFrame
conv_storm_combined_gdf = gpd.GeoDataFrame(df, geometry='geometry')

## Convective Storm Event Index ##
conv_cat_map0 = pd.DataFrame({'event_index': [0],
                         'category': ['N/S']})
conv_cat_map1 = pd.DataFrame({'event_index': np.arange(1,7),
                         'category': ['TSTM','MRGL','SLGT','ENH','MDT','HIGH']})
conv_cat_map = pd.concat([conv_cat_map0,conv_cat_map1],ignore_index=True)

conv_agg_gdf = conv_storm_gdf[['LABEL','geometry']].dissolve(by='LABEL')
conv_agg_gdf = conv_agg_gdf.reset_index()
conv_agg_gdf = conv_agg_gdf.rename(columns={'LABEL':'category'})
conv_agg_gdf = pd.merge(conv_agg_gdf,conv_cat_map,how='left',on='category')

conv_final_gdf = conv_agg_gdf.sort_values(by='event_index')
conv_final_gdf = conv_final_gdf.reset_index(drop=True)

county_centroids_df = pd.read_csv('data/block data/county_centroids.csv')
county_centroids_df['geometry'] = county_centroids_df['geometry'].apply(wkt.loads)
county_centroids = gpd.GeoDataFrame(county_centroids_df, geometry='geometry')

zip_code_centriods_df = pd.read_csv('data/block data/zip_code_centroids.csv')
zip_code_centriods_df['geometry'] = zip_code_centriods_df['geometry'].apply(wkt.loads)
zip_code_centriods = gpd.GeoDataFrame(zip_code_centriods_df, geometry='geometry')

test_membership = get_storm_table(conv_final_gdf,zip_code_centriods,['STNAME','STD_ZIP5','geometry'])

## Winter Storm Index ##
wint_cat_map0 = pd.DataFrame({'event_index': [0],
                         'category': ['N/S']})
wint_cat_map1 = pd.DataFrame({'event_index': np.arange(1,6),
                         'category': ['LIMITED','MINOR','MODERATE','MAJOR','EXTREME']})
wint_cat_map = pd.concat([wint_cat_map0,wint_cat_map1],ignore_index=True)

winter_storm_gdf = gpd.read_file('data/winter storm data/WSSI_OVERALL_Days_1_3_latest/WSSI_OVERALL_Days_1_3.shp')
winter_storm_gdf = winter_storm_gdf.to_crs(epsg=4326) # Reproject to WGS84
winter_agg_gdf = winter_storm_gdf[['IMPACT','geometry']].rename(columns={'IMPACT':'category'})
winter_agg_gdf = pd.merge(winter_agg_gdf,wint_cat_map,how='left',on='category')

winter_final_gdf = winter_agg_gdf.sort_values(by='event_index')
winter_final_gdf = winter_final_gdf.reset_index(drop=True)

winter_final_gdf_simplified = winter_final_gdf.copy()
winter_final_gdf_simplified['geometry'] = winter_final_gdf_simplified['geometry'].simplify(tolerance=0.01)

import time
start_time = time.time()
test_membership2 = get_storm_table(winter_final_gdf_simplified,county_centroids.head(),['NAMELSAD','STUSPS','geometry'])
end_time = time.time()

elapsed_time = end_time - start_time
print(elapsed_time)

polygon_count = 0
for geom in winter_final_gdf.geometry[3]:
    if geom.type == 'MultiPolygon':
        # Count the number of polygons in each MultiPolygon
        polygon_count += len(list(geom.geoms))

print("Number of polygons in all MultiPolygons:", polygon_count)

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Example MultiPolygon (replace this with your actual MultiPolygon)
poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
multi_poly = MultiPolygon([poly1, poly2])

multi_poly = winter_final_gdf.geometry[0]

# Extract centroids
centroids = [poly.centroid for poly in multi_poly.geoms]

# Convert centroids to a NumPy array for clustering
centroid_coords = np.array([[point.x, point.y] for point in centroids])

# Number of clusters
k = 100  # Adjust this based on your requirement

# Apply K-Means clustering
kmeans = KMeans(n_clusters=k)
clusters = kmeans.fit_predict(centroid_coords)

# Combine shapes in each cluster using convex hull
clustered_shapes = []
for i in range(k):
    # Get polygons in this cluster
    cluster_polygons = [multi_poly.geoms[j] for j in range(len(multi_poly.geoms)) if clusters[j] == i]

    # Combine polygons using unary_union + convex hull
    combined_shape = gpd.GeoSeries(cluster_polygons).unary_union.convex_hull
    clustered_shapes.append(combined_shape)

clustered_shapes = gpd.GeoSeries(clustered_shapes)

plt.clf()
plt.cla()
plt.close()
# Visualization
# Original MultiPolygon
multi_poly = winter_final_gdf.geometry[0]
multi_poly = gpd.GeoSeries([poly for poly in multi_poly.geoms])
multi_poly.boundary.plot(color='blue')

# Clustered Shapes
clustered_shapes.boundary.plot(color='green')

# Clustered Shapes Combined
# Combine all intersecting or touching shapes
combined_shape = clustered_shapes.unary_union
combined_shape = gpd.GeoSeries([poly for poly in combined_shape.geoms])
combined_shape.boundary.plot(color='red')

# Original Combined
#combined_shape_orig = multi_poly_series.unary_union
#combined_shape_orig = gpd.GeoSeries([poly for poly in combined_shape_orig.geoms])
#combined_shape_orig.boundary.plot(color='purple')

# Original Simplified & Combined
multi_poly_simplified = multi_poly.simplify(tolerance=0.01)
multi_poly_simplified = multi_poly_simplified.unary_union
multi_poly_simplified = gpd.GeoSeries([poly for poly in multi_poly_simplified.geoms])
multi_poly_simplified.boundary.plot(color='purple')

plt.show()


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd

plt.clf()
plt.cla()
plt.close()

# Cartopy Projection
proj = ccrs.LambertConformal(central_longitude=-95, central_latitude=35)

# Creating the plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': proj})

# Add map features
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)

# Plot the original MultiPolygon
multi_poly.plot(ax=ax, edgecolor='blue', facecolor='none', transform=ccrs.PlateCarree())

# Plot the Clustered Shapes
combined_shape.plot(ax=ax, edgecolor='red', facecolor='none', transform=ccrs.PlateCarree())

# Adjust the extent of the map
ax.set_extent([-130, -60, 20, 55], crs=ccrs.PlateCarree())

plt.show()

import alphashape
import geopandas as gpd

# Example points (replace with your data)
points = gpd.GeoSeries([Point(1, 1), Point(2, 3), Point(3, 1), Point(4, 4), Point(5, 2)])

# Generate the concave hull
alpha = 0.5  # Adjust based on your requirement
concave_hull_p0 = alphashape.alphashape(points, 0.0)
concave_hull_p5 = alphashape.alphashape(points, .5)
concave_hull_p75 = alphashape.alphashape(points, .75)
concave_hull_p9 = alphashape.alphashape(points, .9)

concave_hull_p0_series = gpd.GeoSeries(concave_hull_p0)
concave_hull_p5_series = gpd.GeoSeries(concave_hull_p5)
concave_hull_p75_series = gpd.GeoSeries(concave_hull_p75)
concave_hull_p9_series = gpd.GeoSeries(concave_hull_p9)

plt.clf()
plt.cla()
plt.close()

concave_hull_p0_series.boundary.plot(color='blue')
concave_hull_p5_series.boundary.plot(color='purple')
concave_hull_1p0_series.boundary.plot(color='red')

plt.show()

from shapely.geometry import MultiPolygon, Polygon
import matplotlib.pyplot as plt

# Example MultiPolygon (consisting of two polygons in this case)
multi_poly = MultiPolygon([Polygon([(0, 0), (1, 0), (1,.5), (1, 1), (.5,1), (0, 1)]),
                           Polygon([(2, 2), (2.5,2), (3, 2), (3,2.5), (3, 3), (2, 3)])])

multi_poly = winter_final_gdf.geometry[1]

plt.clf()
plt.cla()
plt.close()

fig, ax = plt.subplots()

multi_poly_series = gpd.GeoSeries([poly for poly in multi_poly.geoms])
multi_poly_series.boundary.plot(ax=ax,color='blue',label='Original')

concave_hull_multi_poly = alphashape.alphashape(extract_boundary_points(multi_poly), 5)
concave_hull_multi_poly_series = gpd.GeoSeries(concave_hull_multi_poly)
concave_hull_multi_poly_series.boundary.plot(ax=ax,color='red',label='Concave Simplified')

plt.show()



def reduce_shapes(multi_poly):
    return alphashape.alphashape(extract_boundary_points(multi_poly), 5)

winter_final_gdf_reduced = winter_final_gdf.copy()
winter_final_gdf_reduced['geometry'] = winter_final_gdf_reduced['geometry'].apply(reduce_shapes)

import time
start_time = time.time()
test_membership2 = get_storm_table(winter_final_gdf_reduced,county_centroids,['NAMELSAD','STUSPS','geometry'])
end_time = time.time()

elapsed_time = end_time - start_time
print(elapsed_time)

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


# List to store all boundary points of all polygons in the MultiPolygon
all_boundary_points = []

# Iterate through each polygon in the MultiPolygon
for polygon in multi_poly:
    all_boundary_points.extend(extract_boundary_points(polygon))

# Now, all_boundary_points contains boundary points of all polygons in the MultiPolygon
