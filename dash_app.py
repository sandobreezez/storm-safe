import dash
from dash import Dash

# ... other necessary imports ...
from dash import html, dcc, Input, Output, callback, dash_table
import geopandas as gpd
import pandas as pd
import plotly.express as px

# Load storm shapes
conv_storm_shapes = gpd.read_file('convective_storm_shapes.geojson')
wint_storm_shapes = gpd.read_file('winter_storm_shapes.geojson')

# Load County Memberships
conv_county_memb = gpd.read_file('convective_storm_county.geojson')
wint_county_memb = gpd.read_file('winter_storm_county.geojson')

# Load Zip Code Memberships
conv_zip_memb = gpd.read_file('convective_storm_zipcode.geojson')
wint_zip_memb = gpd.read_file('winter_storm_zipcode.geojson')

# Load Cat map
conv_cat_map = pd.read_csv('convective_storm_cat_map.csv')
wint_cat_map = pd.read_csv('winter_storm_cat_map.csv')

# Remove N/A category
conv_cat_map = conv_cat_map[conv_cat_map['event_index'] > 0]
wint_cat_map = wint_cat_map[wint_cat_map['event_index'] > 0]

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

# Assuming gdf is your GeoDataFrame sorted by severity (more severe last)
def create_plotly_figure(gdf):
    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf.geometry,
        locations=gdf.index,
        color='category',  # Column denoting the severity
        mapbox_style="carto-positron",
        zoom=3,
        center={"lat": 37.0902, "lon": -95.7129},  # Center of the United States
        opacity=0.5
    )

    #fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    fig.update_layout(
        # Other layout parameters...
        title={
            'text': "Storm Severity Map",  # Your title text
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(
            x=0,  # Position legend on the left side of the plot
            y=1,  # Position legend at the top of the plot
            bgcolor='rgba(255, 255, 255, 0.5)',  # Optional: make legend background semi-transparent
            bordercolor='Black',
            orientation='v'  # Optional: 'v' for vertical, 'h' for horizontal
        )
    )
    return fig



# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        html.Label('Granularity', style={'fontWeight': 'bold'}),
        dcc.RadioItems(
            id='gran_select',
            options=[
                {'label': 'County', 'value': 'County'},
                {'label': 'Zip Code', 'value': 'Zip Code'}
            ],
            value='County',
            labelStyle={'display': 'block', 'margin': '6px 0'},  # Stack labels vertically with margin
            inputStyle={"margin-right": "5px"},  # Spacing between radio button and label
        ),
        html.Br(),
        html.Label('Weather Peril', style={'fontWeight': 'bold'}),
        dcc.RadioItems(
            id='peril_select',
            options=[
                {'label': 'Convective Storms', 'value': 'Convective Storms'},
                {'label': 'Winter Storms', 'value': 'Winter Storms'}
            ],
            value='Convective Storms',
            labelStyle={'display': 'block', 'margin': '6px 0'},  # Stack labels vertically with margin
            inputStyle={"margin-right": "5px"},  # Spacing between radio button and label
        ),
        html.Br(),
        html.Label('Impact Category', style={'fontWeight': 'bold'}),
        dcc.RadioItems(
            id='category_select',
            options=[{'label': 'Test', 'value': 'Test'}],  # Replace with dynamic options
            labelStyle={'display': 'block', 'margin': '6px 0'},  # Stack labels vertically with margin
            inputStyle={"margin-right": "5px"},  # Spacing between radio button and label
        ),
        html.Br(),
        # ... additional components ...
    ], style={
        'width': 'fit-content',
        'display': 'inline-block',
        'minHeight': '400px',  # Adjust minHeight as needed to keep consistent size
    }),
    html.Br(),
    html.Div(id='plot',style={'width': '46%', 'display': 'inline-block', 'padding-right': '30px'}),  # Placeholder for the plot

    html.Div(id='filtered_data',style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding-left': '30px'})  # Placeholder for the table
])

@app.callback(
    Output('category_select', 'options'),
    Input('peril_select', 'value')
)
def update_categories(selected_peril):
    if selected_peril == "Convective Storms":
        options = [{'label': cat, 'value': i + 1} for i,cat in enumerate(conv_cat_map['category'].tolist())]
    elif selected_peril == "Winter Storms":
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
    Output('plot', 'children'),
    [Input('peril_select', 'value'),
     Input('category_select', 'value')]
)
def update_plot(selected_peril, selected_category):
    # Use the appropriate GeoDataFrame based on the selected peril
    if selected_peril == "Convective Storms":
        gdf = conv_storm_shapes
    elif selected_peril == "Winter Storms":
        gdf = wint_storm_shapes
    else:
        return "Please select a peril to display the map."

    # Filter the GeoDataFrame as necessary
    # ...

    # Create the plotly figure and return it
    fig = create_plotly_figure(gdf)
    return dcc.Graph(figure=fig)

@app.callback(
    Output('filtered_data', 'children'),
    [Input('peril_select', 'value'),
     Input('gran_select', 'value'),
     Input('category_select', 'value')]
)

def update_table(selected_peril, selected_granularity, selected_category):
    # Logic to filter data and return as a table
    # You will need to implement the logic to filter the DataFrame based on the inputs
    # and then format it as a table. Dash provides a DataTable component that can be useful here.
    #current_max_index = float(input.category_select()) if input.category_select() is not None else float('inf')
    print('filtered_data2')
    if selected_peril == "Convective Storms":
        if selected_granularity == 'County':
            print('filtered_data3')
            df = conv_county_memb
        elif selected_granularity == 'Zip Code':
            df = conv_zip_memb
    elif selected_peril == "Winter Storms":
        print('filtered_data3')
        if selected_granularity == 'County':
            print('filtered_data3')
            df = wint_county_memb
        elif selected_granularity == 'Zip Code':
            df = wint_zip_memb
    filtered_df = df[df["event_index"] >= int(selected_category)].copy()
    print(selected_category)
    print('filtered_data4')
    #print(current_max_index)
    if not filtered_df.empty and 'geometry' in filtered_df.columns:
        # Extract latitude and longitude from the geometry
        filtered_df['LAT'] = filtered_df['geometry'].y
        filtered_df['LONG'] = filtered_df['geometry'].x
    filtered_df = filtered_df.drop(columns=['geometry'], errors='ignore')
    return dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in filtered_df.columns],
        data=filtered_df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        filter_action='native',  # Enable filtering
        sort_action='native', # Enable sorting
        sort_mode='multi',  # Enable multi-column sorting (hold Shift key and click on additional columns)
        page_action='native',
        page_current=0,
        page_size=20,
    )


if __name__ == '__main__':
    app.run_server(debug=True)