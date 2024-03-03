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
from datetime import datetime, timedelta

# Load Cat map
conv_cat_map = pd.read_csv('convective_storm_cat_map.csv')
wint_cat_map = pd.read_csv('winter_storm_cat_map.csv')

# Remove N/A category
conv_cat_map = conv_cat_map[conv_cat_map['event_index'] > 0]
wint_cat_map = wint_cat_map[wint_cat_map['event_index'] > 0]

cat_map_dict = {}
cat_map_dict['convective_storm'] = conv_cat_map
cat_map_dict['winter_storm'] = wint_cat_map

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

static_data_directory = ''
static_data_file_names = ['county_centroids.geojson','zipcode_centroids.geojson']
static_data_dict = load_files_to_dict_list(static_data_file_names)

static_data_dict['county_centroids'] = static_data_dict['county_centroids'][['NAMELSAD','STUSPS','geometry']]
static_data_dict['zipcode_centroids'] = static_data_dict['zipcode_centroids'][['STD_ZIP5','STUSPS','COUNTYNAME','geometry']]


latest_to_load = ['convective_storm_shapes.geojson','winter_storm_shapes.geojson','convective_storm_county.csv',
                 'winter_storm_county.csv','convective_storm_zipcode.csv','winter_storm_zipcode.csv']
latest_dict = load_files_to_dict_list(latest_to_load)

directory_path = 'data_archive'
# List all files and directories in the specified path
entries = os.listdir(directory_path)
# Filter out directories, keep only files
file_names = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]


# Extracting the desired parts
pull_times_all = []
for file_name in file_names:
    # Splitting at "issue_" and taking the second part
    part_after_issue = file_name.split('issue_')[1]

    # Splitting at ".geojson" and taking the first part
    desired_part = part_after_issue.split('.')[0]

    pull_times_all.append(desired_part)


pull_times_all = sorted(list(set(pull_times_all)),reverse=True)
pull_times_all = [int(x) for x in pull_times_all]

def format_date_time_obj(dt):
    return datetime.strptime(str(dt),'%Y%m%d%H%M') # Convert integer to string and parse it into a datetime object

def format_date_time_obj_to_str(dt_obj):
    return dt_obj.strftime('%Y%m%d%H%M')

max_issue_time_to_keep = max(pull_times_all[1:])
## Fix: The below code, should keep last 30 days
## min_issue_time_to_keep = floor(max(pull_times) / 1000000)*1000000
min_issue_time_to_keep = int(format_date_time_obj_to_str(format_date_time_obj(max_issue_time_to_keep) - timedelta(days=7)))
# If we are only running the archive once a day then we could just index [1:7]
pull_times = [dt for dt in pull_times_all[1:] if dt >= min_issue_time_to_keep]
# This would account for failed runs
# In the future we could have a way to just save the latest issue time for any given day
print(pull_times)

pull_times_highlights = [202304011630,202303310100]

def format_date_time_display(dt):
    dt_obj = format_date_time_obj(dt) # Convert integer to string and parse it into a datetime object
    formatted_date = dt_obj.strftime('%Y-%m-%d UTC %H:%M') # Format the datetime object into the desired string format for the label
    return formatted_date

def build_archive_dropdown_dict(dt_list):
    archive_dropdown_options = []
    for dt in dt_list:
        formatted_date = format_date_time_display(dt) # Format date time
        # Append as a dictionary to the list
        archive_dropdown_options.append({'label': formatted_date, 'value': dt})
    return archive_dropdown_options

archive_dropdown_options = build_archive_dropdown_dict(pull_times)
highlight_dropdown_options = build_archive_dropdown_dict(pull_times_highlights)
archive_to_load = [file_name for file_name in file_names if any(str(time) in file_name for time in pull_times)]
highlight_to_load = [file_name for file_name in file_names if any(str(time) in file_name for time in pull_times_highlights)]

# Dictionary to store the GeoDataFrames
archive_directory = 'data_archive/'
archive_dict = load_files_to_dict_list(archive_to_load,archive_directory)
hightlight_dict = load_files_to_dict_list(highlight_to_load,archive_directory)


data_dictionary = {}
data_dictionary['Latest'] = latest_dict
data_dictionary['Archive'] = archive_dict
data_dictionary['Highlights'] = hightlight_dict

pd.merge(static_data_dict['zipcode_centroids'],data_dictionary['Latest']['winter_storm_zipcode'],on='STD_ZIP5',how='left')


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
color_map_dict = {}
color_map_dict['convective_storm'] = conv_storm_discr_color_map
color_map_dict['winter_storm'] = wint_storm_discr_color_map

# Assuming gdf is your GeoDataFrame sorted by severity (more severe last)
def create_plotly_figure(gdf, color_discrete_map,title):
    fig = px.choropleth_mapbox(
        gdf.drop(columns='geometry'),
        geojson=gdf.geometry,
        locations=gdf.index,
        color='category',
        color_discrete_map=color_discrete_map,  # Use the discrete color map
        mapbox_style="open-street-map",  # Use OpenStreetMap style
        zoom=3,
        center={"lat": 37.0902, "lon": -95.7129},
        opacity=0.6,
    )

    fig.update_layout(
        title={
            'text': title,
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
    html.Label(id='app_title', children='Short the Weather: Real-time Moratorium Automation', style={'fontWeight': 'bold','fontSize': '32px','justifyContent': 'center'}),
    html.Br(),
    html.Br(),
    html.Div([
        # Existing controls
        html.Div([
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
            html.Br(),
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
            )
        ], style={'display': 'inline-block', 'verticalAlign': 'top', 'margin-right': '20px'}),

        # Download button
        html.Div([
            html.Label('Issued', style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='latest_or_archive_select',
                options=[
                    {'label': 'Latest', 'value': 'Latest'},
                    {'label': 'Archive', 'value': 'Archive'},
                    {'label': 'Highlights', 'value': 'Highlights'}
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
                    value=[]
                )
            ], id='archive_dropdown_container', style={'display': 'none', 'width' : '100%'})
        ], style={'display': 'inline-block', 'margin-left': '20px', 'verticalAlign': 'top','width':'250px'}),

        html.Div([
            html.Br()
        ],style={'width': '28%', 'display': 'inline-block', 'padding-right': '30px','verticalAlign': 'top'}),
        html.Div([
            html.Div([
                html.Br(), #1
                html.Br(), #2
                html.Br(), #3
                html.Br(), #4
                html.Br(), #5
                html.Br(), #6
                html.Br(), #7
                html.Br(), #8
                html.Br(), #9
                html.Br(), #10
                html.Br(), #11
                html.Br(), #12
                html.Br(), #13
                html.Br(), #14
                html.Br(), #15
                html.Br(), #16
                html.Br(), #17
                html.Br(), #18
                html.Button("Download Data", id="btn_csv")
            ]),
        ],style={'width': '20%', 'display': 'inline-block', 'padding-left': '30px'}),
    ], style={'display': 'flex'}),

    html.Br(),

    dcc.Store(id='stored-data'),
    html.Div([
        html.Div(id='plot'),
        html.Br(),
        html.Div(id='issue_time_display')],
        style={'width': '46%', 'display': 'inline-block', 'padding-right': '30px','height': '570px','verticalAlign': 'top'}),  # Placeholder for the plot
    html.Div(id='filtered_data',style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding-left': '30px'}),
    dcc.Download(id="download-data"),
])

@app.callback(
    Output('category_select', 'options'),
    Input('peril_select', 'value')
)
def update_categories(selected_peril):
    cat_map = cat_map_dict[selected_peril]
    options = [{'label': cat, 'value': i + 1} for i,cat in enumerate(cat_map['category'].tolist())]
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

@app.callback([
    Output('archive_dropdown_container', 'style'),
    Output('archive_dropdown', 'options'),
    Output('archive_dropdown', 'value')],
    [Input('latest_or_archive_select', 'value')]
)
def toggle_dropdown(selected_value):
    if selected_value == 'Archive':
        return {'display': 'block'}, archive_dropdown_options, archive_dropdown_options[0]['value']
    elif selected_value == 'Highlights':
        return {'display': 'block'}, highlight_dropdown_options, highlight_dropdown_options[0]['value']
    else:
        return {'display': 'none'}, [], []

@app.callback(
    Output('issue_time_display', 'children'),
    [Input('peril_select', 'value'),
     Input('latest_or_archive_select','value'),
     Input('archive_dropdown','value')]
)
def update_issue_time_display(selected_peril,latest_or_archive,selected_archive):
    archive_value = '_issue_' + str(selected_archive) if latest_or_archive in ['Archive','Highlights'] else ''
    gdf_shape_filename = selected_peril + '_shapes' + archive_value
    gdf = data_dictionary[latest_or_archive][gdf_shape_filename]


    if gdf_shape_filename in data_dictionary[latest_or_archive].keys():
        gdf = data_dictionary[latest_or_archive][gdf_shape_filename]
        issue_time = format_date_time_display(gdf['ISSUE_TIME'].iloc[0])
        start_time = format_date_time_display(gdf['START_TIME'].iloc[0])
        end_time = format_date_time_display(gdf['END_TIME'].iloc[0])
    else:
        issue_time = 'N/A'
        start_time = 'N/A'
        end_time = 'N/A'
    return html.Div([
        html.Span('ISSUE_TIME: ' + str(issue_time), style={'margin-right': '20px'}),
        html.Span('START_TIME: ' + str(start_time), style={'margin-right': '20px'}),
        html.Span('END_TIME: ' + str(end_time))
    ],style={'fontWeight': 'bold'})

@app.callback(
    Output('plot', 'children'),
    [Input('peril_select', 'value'),
     Input('category_select', 'value'),
     Input('latest_or_archive_select','value'),
     Input('archive_dropdown','value')]
)

def update_plot(selected_peril, selected_category, latest_or_archive, selected_archive):
    archive_value = '_issue_' + str(selected_archive) if latest_or_archive in ['Archive','Highlights'] else ''
    gdf_filename = selected_peril + '_shapes' + archive_value

    color_map = color_map_dict[selected_peril]
    if gdf_filename in data_dictionary[latest_or_archive].keys():
        ## if we find the key then use that df
        gdf = data_dictionary[latest_or_archive][gdf_filename]
    else:
        ## if we dont select the latest and 0 it out
        gdf = data_dictionary['Latest'][gdf_filename.split('_issue_')[0]].copy()
        gdf['geometry'] = [Polygon() for _ in range(len(gdf))]

    title = (selected_peril.replace("_", " ") + " severity index").title() # winter_storm --> Winter Storm Severity Index
    fig = create_plotly_figure(gdf,color_map,title)
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
    static_df_key = selected_granularity + '_centroids' ### grab either zipcode or county data
    static_df = static_data_dict[static_df_key]

    archive_value = '_issue_' + str(selected_archive) if latest_or_archive in ['Archive','Highlights'] else ''
    membership_filename = selected_peril + '_' + selected_granularity + archive_value

    if membership_filename in data_dictionary[latest_or_archive].keys():
        ## if we find the key then use that df
        membership_df = data_dictionary[latest_or_archive][membership_filename]
    else:
        ## if we dont find the key, select the latest and 0 it out
        membership_df = data_dictionary['Latest'][membership_filename.split('_issue_')[0]]
        membership_df['event_index'] = 0
        membership_df['category'] = 'N/A'


    if selected_granularity == 'zipcode':
        merge_col = ['STD_ZIP5','STUSPS']
    elif selected_granularity == 'county':
        merge_col = ['NAMELSAD','STUSPS']

    merged_df = pd.merge(static_df, membership_df, how='left', on=merge_col)
    merged_df.fillna(0, inplace=True)

    filtered_df = merged_df[merged_df["event_index"] >= int(selected_category)].copy()
    if not filtered_df.empty and 'geometry' in filtered_df.columns:
        # Extract latitude and longitude from the geometry
        filtered_df['LAT'] = filtered_df['geometry'].y
        filtered_df['LONG'] = filtered_df['geometry'].x
    filtered_df = filtered_df.drop(columns=['geometry','event_index'], errors='ignore')
    tail_cols_filtered_df = filtered_df.columns
    filtered_df['peril'] = selected_peril
    filtered_df = filtered_df[['peril'] + list(tail_cols_filtered_df)]
    stored_data = {
        "records": filtered_df.to_dict('records'),
        "columns": list(filtered_df.columns)
    }

    return stored_data

@app.callback(
    Output('filtered_data', 'children'),
    [Input('stored-data', 'data')]
)

def update_display_table(stored_data):
    if not stored_data or 'records' not in stored_data or 'columns' not in stored_data:
        return dash_table.DataTable()

    data = stored_data['records']
    columns = [{"name": col, "id": col} for col in stored_data['columns']]

    return dash_table.DataTable(
        columns=columns,
        data=data,
        style_table={'overflowX': 'auto'},
        filter_action='native',
        sort_action='native',
        sort_mode='multi',
        page_action='native',
        page_current=0,
        page_size=20,
    )

@app.callback(
    Output("download-data", "data"),
    [Input("btn_csv", "n_clicks"),
     Input('peril_select', 'value'),
     Input('category_select', 'value'),
     Input('gran_select', 'value'),
     Input('latest_or_archive_select', 'value'),
     Input('archive_dropdown', 'value')],
    State('stored-data', 'data'),  # Retrieve the stored data
    prevent_initial_call=True
)
def download_data(n_clicks, selected_peril, selected_category, selected_granularity, latest_or_archive, selected_archive,stored_data):
    archive_value = '_issue_' + str(selected_archive) if latest_or_archive in ['Archive','Highlights'] else ''
    gdf_shape_filename = selected_peril + '_shapes' + archive_value
    gdf = data_dictionary[latest_or_archive][gdf_shape_filename]

    if gdf_shape_filename in data_dictionary[latest_or_archive].keys():
        gdf = data_dictionary[latest_or_archive][gdf_shape_filename]
        issue_time = format_date_time_display(gdf['ISSUE_TIME'].iloc[0])
    else:
        issue_time = 'N/A'

    if n_clicks is None or stored_data is None:
        raise dash.exceptions.PreventUpdate

    # Convert the stored data back to a DataFrame
    df = pd.DataFrame.from_records(stored_data['records'])

    # Generate CSV string from DataFrame
    csv_string = generate_csv(df)
    cat_map = cat_map_dict[selected_peril]
    selected_category_string = cat_map[cat_map['event_index'] == selected_category]['category'].iloc[0]
    file_name_strings = [selected_peril,selected_category_string,selected_granularity,'issue',issue_time]
    file_name = '_'.join(file_name_strings) + '.csv'

    return dict(content=csv_string, filename=file_name)



if __name__ == '__main__':
    app.run_server(debug=True)