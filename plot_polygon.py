import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon as ShapelyMultiPolygon

plt.clf()
# Set up the map projection and create a new subplot
proj = ccrs.LambertConformal(central_longitude=-95, central_latitude=35)
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': proj})

# Add map features
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)

# Set the extent to cover the continental US (you can adjust as needed)
ax.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())

# Assume `storm_gdf['geometry'][0]` is a Shapely Polygon, and we add it to the map
#shapely_polygon = storm_gdf['geometry'][0]  # This should be a Shapely Polygon
shapely_polygon = all_gdf.unary_union
if isinstance(shapely_polygon, (ShapelyPolygon, ShapelyMultiPolygon)):
    ax.add_geometries([shapely_polygon], ccrs.PlateCarree(),
                      facecolor='red', edgecolor='blue', alpha=0.5)

plt.title('Polygon Over US')
plt.show()