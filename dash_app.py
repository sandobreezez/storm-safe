import dash
from dash import Dash

# ... other necessary imports ...
from dash import html, dcc, Input, Output, State, callback, dash_table
import dash_extensions as de
import geopandas as gpd
import pandas as pd
import plotly.express as px
import os
from math import ceil, floor
from datetime import datetime

# Load Cat map
conv_cat_map = pd.read_csv('convective_storm_cat_map.csv')
wint_cat_map = pd.read_csv('winter_storm_cat_map.csv')

# Remove N/A category
conv_cat_map = conv_cat_map[conv_cat_map['event_index'] > 0]
wint_cat_map = wint_cat_map[wint_cat_map['event_index'] > 0]

def load_files_to_dict_list(file_list,directory=''):
    loaded_file_dict = {}
    for file_name in file_list:
        file_path = directory + file_name
        if '.csv' in file_name:
            key = file_name.replace('.csv', '')
            loaded_file_dict[key] = pd.read_csv(file_path)
            if 'STD_ZIP5' in loaded_file_dict[key].columns:
                loaded_file_dict[key]['STD_ZIP5'] = loaded_file_dict[key]['STD_ZIP5'].astype(str).str.zfill(5)
        elif '.geojson' in file_name:
            key = file_name.replace('.geojson', '')
            loaded_file_dict[key] = gpd.read_file(file_path)
    return loaded_file_dict

#print(load_files_to_dict_list(latest_to_load))
static_data_directory = ''
static_data_file_names = ['county_centroids.geojson','zipcode_centroids.geojson']
static_data_dict = load_files_to_dict_list(static_data_file_names)
print(static_data_dict)
static_data_dict['county_centroids'] = static_data_dict['county_centroids'][['NAMELSAD','STUSPS','geometry']]
static_data_dict['zipcode_centroids'] = static_data_dict['zipcode_centroids'][['STD_ZIP5','STUSPS','COUNTYNAME','geometry']]

# county_cols_to_keep = ['NAMELSAD','STUSPS','geometry']
# zip_cols_to_keep = ['STD_ZIP5','STUSPS','COUNTYNAME','geometry']

latest_to_load = ['convective_storm_shapes.geojson','winter_storm_shapes.geojson','convective_storm_county.csv',
                 'winter_storm_county.csv','convective_storm_zipcode.csv','winter_storm_zipcode.csv']
latest_dict = load_files_to_dict_list(latest_to_load)

directory_path = 'data_archive'
# List all files and directories in the specified path
entries = os.listdir(directory_path)
# Filter out directories, keep only files
file_names = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
print(file_names)

# Extracting the desired parts
pull_times = []
for file_name in file_names:
    # Splitting at "issue_" and taking the second part
    part_after_issue = file_name.split('issue_')[1]

    # Splitting at ".geojson" and taking the first part
    desired_part = part_after_issue.split('.')[0]

    pull_times.append(desired_part)


pull_times = sorted(list(set(pull_times)),reverse=True)
pull_times = [int(x) for x in pull_times]

max_issue_time_to_keep = max(pull_times)
## Fix: The below code, should keep last 30 days
## min_issue_time_to_keep = floor(max(pull_times) / 1000000)*1000000

## pull_times = [x for x in pull_times_all if x > min_issue_time_to_keep]
print(pull_times)

archive_dropdown_options = []
for dt in pull_times:
    dt = str(dt)
    # Convert integer to string and parse it into a datetime object
    dt_obj = datetime.strptime(dt, '%Y%m%d%H%M')

    # Format the datetime object into the desired string format for the label
    formatted_date = dt_obj.strftime('%Y-%m-%d UTC %H:%M')

    # Append as a dictionary to the list
    archive_dropdown_options.append({'label': formatted_date, 'value': dt})

archive_to_load = [file_name for file_name in file_names if any(str(time) in file_name for time in pull_times)]

# Dictionary to store the GeoDataFrames
archive_directory = 'data_archive/'
archive_dict = load_files_to_dict_list(archive_to_load,archive_directory)


# archive_dict['convective_storm_county_issue_202401290600'].drop(columns=['geometry','STUSPS']).columns
# for key in archive_dict:
#     if 'zipcode' in key:
#         archive_dict[key].drop(columns=['geometry','STUSPS','COUNTYNAME']).to_csv(archive_directory + key + '.csv',index=False)
#     elif 'county' in key:
#         archive_dict[key].drop(columns=['geometry', 'STUSPS']).to_csv(archive_directory + key + '.csv',index=False)


## test_gdf = gpd.read_file('data_archive/convective_storm_county_issue_202401280600.geojson')


data_dictionary = {}
data_dictionary['Latest'] = latest_dict
data_dictionary['Archive'] = archive_dict
print(static_data_dict['zipcode_centroids'])
print(data_dictionary['Latest']['winter_storm_zipcode'])

pd.merge(static_data_dict['zipcode_centroids'],data_dictionary['Latest']['winter_storm_zipcode'],on='STD_ZIP5',how='left')
# def create_plotly_figure(gdf):
#     # Assuming the geometry column has been converted to 'LAT' and 'LONG'
#     fig = px.scatter_geo(
#         gdf,
#         lat='LAT',
#         lon='LONG',
#         scope='usa',  # Set the scope to the USA
#         # Add more styling and data parameters as needed
#     )
#     fig.update_layout(
#         margin={"r":0,"t":0,"l":0,"b":0},
#         geo=dict(
#             landcolor='rgb(217, 217, 217)',
#         )
#     )
#     return fig

wint_storm_discr_color_map = {
    'LIMITED': '#1E90FF',  # Darker blue
    'MINOR': '#00BFFF',    # Lighter blue
    'MODERATE': '#FFFF00', # Yellow
    'MAJOR': '#FFA500',    # Orange
    'EXTREME': '#FF4500',  # Deeper red
}

conv_storm_discr_color_map = {
    'TSTM': '#1E90FF',  # Darker blue
    'MRGL': '#00BFFF',    # Lighter blue
    'SLGT': '#00FF00',  # Bright green
    'ENH': '#FFFF00',    # Yellow
    'MDT': '#FFA500',    # Orange
    'HIGH': '#FF4500',  # Deeper red
}

# Assuming gdf is your GeoDataFrame sorted by severity (more severe last)
def create_plotly_figure(gdf, color_discrete_map):
    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf.geometry,
        locations=gdf.index,
        color='category',
        color_discrete_map=color_discrete_map,  # Use the discrete color map
        mapbox_style="open-street-map",  # Use OpenStreetMap style
        zoom=3,
        center={"lat": 37.0902, "lon": -95.7129},
        opacity=0.7,
    )

    fig.update_layout(
        title={
            'text': "Storm Severity Map",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='Black',
            orientation='v'
        ),
        height=570
    )
    return fig

def generate_csv(df):
    return df.to_csv(index=False, encoding='utf-8')



# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        # Existing controls
        html.Div([
            html.Label('Granularity', style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='gran_select',
                options=[
                    {'label': 'County', 'value': 'county'},
                    {'label': 'Zip Code', 'value': 'zipcode'}
                ],
                value='county',
                labelStyle={'display': 'block', 'margin': '6px 0'},
                inputStyle={"margin-right": "5px"},
            ),
            html.Br(),
            html.Label('Weather Peril', style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='peril_select',
                options=[
                    {'label': 'Convective Storms', 'value': 'convective_storm'},
                    {'label': 'Winter Storms', 'value': 'winter_storm'}
                ],
                value='winter_storm',
                labelStyle={'display': 'block', 'margin': '6px 0'},
                inputStyle={"margin-right": "5px"},
            ),
            html.Br(),
            html.Label('Impact Category', style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='category_select',
                options=[{'label': 'Test', 'value': 'Test'}],
                labelStyle={'display': 'block', 'margin': '6px 0'},
                inputStyle={"margin-right": "5px"},
            ),
        ], style={'display': 'inline-block', 'verticalAlign': 'top'}),

        # Download button
        html.Div([
            html.Button("Download Data", id="btn_csv"),
            html.Br(),
            html.Label('Issued', style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='latest_or_archive_select',
                options=[
                    {'label': 'Latest', 'value': 'Latest'},
                    {'label': 'Archive', 'value': 'Archive'}
                ],
                value='Latest',
                labelStyle={'display': 'block', 'margin': '6px 0'},
                inputStyle={"margin-right": "5px"},
            ),
            html.Br(),
            # Dropdown menu (initially hidden)
            html.Div([
                dcc.Dropdown(
                    id='archive_dropdown',
                    options=archive_dropdown_options,
                    value=str(max_issue_time_to_keep)
                )
            ], id='archive_dropdown_container', style={'display': 'none', 'width' : '100%'})
        ], style={'display': 'inline-block', 'margin-left': '20px', 'verticalAlign': 'top','width':'300px'}),


    ], style={'display': 'flex'}),

    html.Br(),

    dcc.Store(id='stored-data'),
    html.Div([
        html.Div(id='plot'),
        html.Br(),
        html.Label(id='issue_time', children='Issue Time: ', style={'fontWeight': 'bold'}),
        html.Label(id='start_time', children='      Start Time: ', style={'fontWeight': 'bold'}),
        html.Label(id='end_time', children='      End Time: ', style={'fontWeight': 'bold'})],
        style={'width': '46%', 'display': 'inline-block', 'padding-right': '30px','height': '570px','verticalAlign': 'top'}),  # Placeholder for the plot
    html.Div(id='filtered_data',style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding-left': '30px'}),
    dcc.Download(id="download-data"),
])

@app.callback(
    Output('category_select', 'options'),
    Input('peril_select', 'value')
)
def update_categories(selected_peril):
    if selected_peril == "convective_storm":
        options = [{'label': cat, 'value': i + 1} for i,cat in enumerate(conv_cat_map['category'].tolist())]
    elif selected_peril == "winter_storm":
        options = [{'label': cat, 'value': i + 1} for i,cat in enumerate(wint_cat_map['category'].tolist())]
    else:
        options = []
    return options

@app.callback(
    Output('category_select', 'value'),
    Input('category_select', 'options')
)

## test github
def update_category_selection(category_options):
    if category_options != []:
        return 1
    else:
        return None

@app.callback(
    Output('archive_dropdown_container', 'style'),
    Input('latest_or_archive_select', 'value')
)
def toggle_dropdown(selected_value):
    if selected_value == 'Archive':
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(
    [Output('issue_time', 'children'),
     Output('start_time', 'children'),
     Output('end_time', 'children')],
    [Input('peril_select', 'value'),
     Input('latest_or_archive_select','value'),
     Input('archive_dropdown','value')]
)
def update_issue_time_display(selected_peril,latest_or_archive,selected_archive):
    archive_value = '_issue_' + selected_archive if latest_or_archive == 'Archive' else ''
    print(archive_value)
    gdf_filename = selected_peril + '_shapes' + archive_value
    print(gdf_filename)
    gdf = data_dictionary[latest_or_archive][gdf_filename]
    color_map = conv_storm_discr_color_map if selected_peril == 'convective_storm' else wint_storm_discr_color_map

    if gdf_filename in data_dictionary[latest_or_archive].keys():
        ## if we find the key then use that df
        gdf = data_dictionary[latest_or_archive][gdf_filename]
    else:
        ## if we dont select the latest and 0 it out
        membership_df = data_dictionary['Latest'][membership_filename.split('_issue_')[0]]
        gdf = data_dictionary['Latest'][gdf_filename.split('_issue_')[0]]
        gdf['geometry'] = [Polygon() for _ in range(len(gdf))]

    return 'ISSUE_TIME: ' + str(gdf['ISSUE_TIME'].iloc[0]), '      START_TIME: ' + str(gdf['START_TIME'].iloc[0]), '      END_TIME: ' + str(gdf['END_TIME'].iloc[0])

@app.callback(
    Output('plot', 'children'),
    [Input('peril_select', 'value'),
     Input('category_select', 'value'),
     Input('latest_or_archive_select','value'),
     Input('archive_dropdown','value')]
)

def update_plot(selected_peril, selected_category,latest_or_archive,selected_archive):
    # Use the appropriate GeoDataFrame based on the selected peril
    #if latest_or_archive == 'Latest' | type(selected_archive) != 'NoneType':
    archive_value = '_issue_' + selected_archive if latest_or_archive == 'Archive' else ''
    print(archive_value)
    gdf_filename = selected_peril + '_shapes' + archive_value
    print(gdf_filename)
    gdf = data_dictionary[latest_or_archive][gdf_filename]
    color_map = conv_storm_discr_color_map if selected_peril == 'convective_storm' else wint_storm_discr_color_map

    if gdf_filename in data_dictionary[latest_or_archive].keys():
        ## if we find the key then use that df
        gdf = data_dictionary[latest_or_archive][gdf_filename]
    else:
        ## if we dont select the latest and 0 it out
        membership_df = data_dictionary['Latest'][membership_filename.split('_issue_')[0]]
        gdf = data_dictionary['Latest'][gdf_filename.split('_issue_')[0]]
        gdf['geometry'] = [Polygon() for _ in range(len(gdf))]
    # if selected_peril == "Convective Storms":
    #     gdf = conv_storm_shapes
    #     color_map = conv_storm_discr_color_map
    # elif selected_peril == "Winter Storms":
    #     gdf = wint_storm_shapes
    #     color_map = wint_storm_discr_color_map
    # else:
    #     return "Please select a peril to display the map."

    # Filter the GeoDataFrame as necessary
    # ...

    # Create the plotly figure and return it
    fig = create_plotly_figure(gdf,color_map)
    return dcc.Graph(figure=fig)

@app.callback(
    Output('stored-data', 'data'),
    [Input('peril_select', 'value'),
     Input('gran_select', 'value'),
     Input('category_select', 'value'),
     Input('latest_or_archive_select', 'value'),
     Input('archive_dropdown', 'value')]
)

def update_table(selected_peril, selected_granularity, selected_category,latest_or_archive,selected_archive):
    # Logic to filter data and return as a table
    # You will need to implement the logic to filter the DataFrame based on the inputs
    # and then format it as a table. Dash provides a DataTable component that can be useful here.
    #current_max_index = float(input.category_select()) if input.category_select() is not None else float('inf')

    static_df_key = selected_granularity + '_centroids' ### grab either zipcode or county data
    print(static_df_key)
    static_df = static_data_dict[static_df_key]
    print(static_df)
    archive_value = '_issue_' + selected_archive if latest_or_archive == 'Archive' else ''
    #print(archive_value)
    membership_filename = selected_peril + '_' + selected_granularity + archive_value
    #print(membership_filename)

    if membership_filename in data_dictionary[latest_or_archive].keys():
        ## if we find the key then use that df
        membership_df = data_dictionary[latest_or_archive][membership_filename]
    else:
        ## if we dont select the latest and 0 it out
        membership_df = data_dictionary['Latest'][membership_filename.split('_issue_')[0]]
        membership_df['event_index'] = 0
        membership_df['category'] = 'N/A'


    if selected_granularity == 'zipcode':
        merge_col = 'STD_ZIP5'
    elif selected_granularity == 'county':
        merge_col = 'NAMELSAD'

    merged_df = pd.merge(static_df, membership_df, how='left', on=merge_col)
    merged_df.fillna(0, inplace=True)
    ##color_map = conv_storm_discr_color_map if selected_peril == 'convective_storm' else wint_storm_discr_color_map

    # print('filtered_data2')
    # if selected_peril == "convective_storm":
    #     if selected_granularity == 'county':
    #         print('filtered_data3')
    #         df = conv_county_memb
    #     elif selected_granularity == 'zipcode':
    #         df = conv_zip_memb
    # elif selected_peril == "winter_storm":
    #     print('filtered_data3')
    #     if selected_granularity == 'county':
    #         print('filtered_data3')
    #         df = wint_county_memb
    #     elif selected_granularity == 'zipcode':
    #         df = wint_zip_memb
    filtered_df = merged_df[merged_df["event_index"] >= int(selected_category)].copy()
    print(selected_category)
    print('filtered_data4')
    #print(current_max_index)
    if not filtered_df.empty and 'geometry' in filtered_df.columns:
        # Extract latitude and longitude from the geometry
        filtered_df['LAT'] = filtered_df['geometry'].y
        filtered_df['LONG'] = filtered_df['geometry'].x
    filtered_df = filtered_df.drop(columns=['geometry'], errors='ignore')
    return filtered_df.to_dict('records')

@app.callback(
    Output('filtered_data', 'children'),
    [Input('stored-data', 'data')]
)

def update_display_table(stored_data):
    if stored_data is None:
        columns = []
        stored_data = pd.DataFrame().to_dict('records')
    else:
        columns= [{"name": i, "id": i} for i in stored_data[0].keys()]

    return dash_table.DataTable(
        columns=columns,
        data=stored_data,
        style_table={'overflowX': 'auto'},
        filter_action='native',  # Enable filtering
        sort_action='native', # Enable sorting
        sort_mode='multi',  # Enable multi-column sorting (hold Shift key and click on additional columns)
        page_action='native',
        page_current=0,
        page_size=20,
    )

@app.callback(
    Output("download-data", "data"),
    Input("btn_csv", "n_clicks"),
    State('stored-data', 'data'),  # Retrieve the stored data
    prevent_initial_call=True
)
def download_data(n_clicks, stored_data):
    if n_clicks is None or stored_data is None:
        raise dash.exceptions.PreventUpdate

    # Convert the stored data back to a DataFrame
    df = pd.DataFrame.from_records(stored_data)

    # Generate CSV string from DataFrame
    csv_string = generate_csv(df)
    return dict(content=csv_string, filename="filtered_data.csv")



if __name__ == '__main__':
    app.run_server(debug=True)