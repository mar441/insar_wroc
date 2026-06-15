import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
from scipy.stats import t
import numpy as np
from geopy.distance import geodesic
import os
import psutil
import dash_bootstrap_components as dbc
from dash import dash_table
import plotly.graph_objects as go  
import plotly.io as pio
import dash_auth

def load_displacement_data(file_path, file_label):
    df = pd.read_csv(file_path)
    df = df.melt(id_vars=['Date'], 
                 var_name='pid', 
                 value_name='displacement')
    df['timestamp'] = pd.to_datetime(df['Date'])
    df.drop(columns=['Date'], inplace=True)
    df['file'] = file_label
    df['pid'] = df['pid'].astype(str)
    return df

def load_anomaly_data(file_path, file_label):
    df = pd.read_csv(file_path)
    df['file'] = file_label
    df['pid'] = df['pid'].astype(str)
    return df
            
geo_data_nysa_ml = pd.read_csv('wroclaw_geo_filtered.csv', delimiter=',')
geo_data_nysa_ml['pid'] = geo_data_nysa_ml['pid'].astype(str).str.strip()

displacement_data_nysa_ml = load_displacement_data('wroclaw_caly_filtered.csv', 'Ascending 175')
displacement_data_nysa_ml['pid'] = displacement_data_nysa_ml['pid'].astype(str).str.strip() 
all_data_nysa_ml = pd.merge(displacement_data_nysa_ml, geo_data_nysa_ml, on='pid', how='left')

prediction_data_nysa_ml = pd.read_csv('predictions_ml_filtered.csv', delimiter=',')
prediction_data_nysa_ml = prediction_data_nysa_ml.melt(var_name='pid', value_name='predicted_displacement')
prediction_data_nysa_ml['label'] = 'ML Nysa Prediction Set'
prediction_data_nysa_ml['step'] = prediction_data_nysa_ml.groupby('pid').cumcount()

anomaly_data_nysa_95_ml = load_anomaly_data('anomaly_95_filtered.csv', 'Anomaly Set 1 ML (95%)')
anomaly_data_nysa_95_ml = anomaly_data_nysa_95_ml.groupby('pid').head(61)

anomaly_data_nysa_99_ml = load_anomaly_data('anomaly_99_filtered.csv', 'Anomaly Set 1 ML (99%)')
anomaly_data_nysa_99_ml = anomaly_data_nysa_99_ml.groupby('pid').head(61)

all_data_nysa_ml.sort_values(by=['pid', 'timestamp'], inplace=True)
all_data_nysa_ml['displacement_diff'] = all_data_nysa_ml.groupby('pid')['displacement'].diff().round(1)
all_data_nysa_ml['time_diff'] = all_data_nysa_ml.groupby('pid')['timestamp'].diff().dt.days.round(1)
all_data_nysa_ml['displacement_speed'] = ((all_data_nysa_ml['displacement_diff'] / all_data_nysa_ml['time_diff']) * 365).round(1)

mean_velocity_data_nysa_ml = all_data_nysa_ml.groupby('pid')['displacement_speed'].mean().round(1).reset_index()
mean_velocity_data_nysa_ml.rename(columns={'displacement_speed': 'mean_velocity'}, inplace=True)
all_data_nysa_ml = pd.merge(all_data_nysa_ml, mean_velocity_data_nysa_ml, on='pid', how='left')

def compute_prefix_sums(data):
    data = data.sort_values(by=['pid', 'step'])
    pivot = data.pivot(index='pid', columns='step', values='predicted_displacement').apply(pd.to_numeric, errors='coerce').fillna(0).round(1)
    pivot.columns = pivot.columns.astype(int) 
    pivot = pivot.sort_index(axis=1)
    for col in pivot.columns[1:]:
        pivot[col] = (pivot[col] + pivot[col-1]).round(1)
    return pivot

nysa_ml_prefix = compute_prefix_sums(prediction_data_nysa_ml)

prefix_data = {
    ('nysa', 'ml'): nysa_ml_prefix,
}

MAX_NYSA = nysa_ml_prefix.columns.max()

def add_obs_step(df):
    df = df.sort_values(by=['pid', 'timestamp'])
    df['obs_step'] = df.groupby('pid').cumcount() + 1
    return df

all_data_nysa_ml = add_obs_step(all_data_nysa_ml)

def compute_prefix_sums_actual(df):
    pivot = df.pivot(index='pid', columns='obs_step', values='displacement').fillna(0).round(1)
    pivot[0] = 0
    pivot = pivot.sort_index(axis=1)
    for col in pivot.columns[1:]:
        pivot[col] = (pivot[col] + pivot[col-1]).round(1)
    return pivot

actual_nysa_ml_prefix = compute_prefix_sums_actual(all_data_nysa_ml)

actual_prefix_data = {
    'nysa': actual_nysa_ml_prefix,
}

MAX_ACTUAL_NYSA = actual_nysa_ml_prefix.columns.max()

TEST_START_DATE = pd.Timestamp('2021-01-01')

orbit_geometry_info = {
    'Ascending 175': {
        'Relative orbit number': '175',
        'View angle': '348.9°',
        'Mean Incidence angle': '33.18°'  
    }}


px.set_mapbox_access_token("pk.eyJ1IjoibnBpZWsiLCJhIjoiY21jN2tvYXE1MTRqeTJrc2NtaTlvNXQyZSJ9.abg-EkfnNKp7bgwEvgRp0w")

app = dash.Dash(__name__, suppress_callback_exceptions=True,external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server  
app.layout = html.Div([

    html.Div([
        html.H3("UPWR InSAR-based time series monitoring, prediction and anomaly detection platform", 
                style={'display': 'inline-block', 'margin-right': '20px'}),
        html.Div([
            html.Button(
                "Info", 
                id="open-help-modal", 
                n_clicks=0, 
                className="help-button", 
                style={'display': 'inline-block', 'marginRight': '10px','backgroundColor': 'transparent',
                       'borderRadius': '10px', 'border': '1px solid #ccc'}
            ),
            html.Button(
                "Legend",
                id="open-legend-modal",
                n_clicks=0,
                className="help-button",
                style={'display': 'inline-block','backgroundColor': 'transparent',
                       'borderRadius': '5px', 'border': '1px solid #ccc'}
            )
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center'}),

    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Instruction")),
            dbc.ModalBody(
                html.Div([
                    html.Iframe(
                        src="https://mar441.github.io/upwrinsarmonitoring_projekt/INSTRUKCJA_OBSLUGI_SERWISU.html",
                        style={"height": "90vh", "width": "100%"}
                    )
                ])
            )
        ],
        id="help-modal",
        is_open=False,
        size="xl",
        style={"overflowY": "auto"}
    ),
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Legend Settings")),
            dbc.ModalBody(
                html.Div([
                    html.Label("Points Transparency"),
                    dcc.Slider(
                        id='point-opacity-slider',
                        min=1, 
                        max=1, 
                        step=0.1,
                        value=1,
                        marks={0: '0', 0.5: '0.5', 1: '1.0'}
                    ),
                    html.Br(),
                    html.Label("Points Size"),
                    dcc.Slider(
                        id='point-size-slider',
                        min=3, 
                        max=20, 
                        step=1,
                        value=7,
                        marks={3: '3', 10: '10', 20: '20'}
                    ),
                    html.Br(),
                    html.Div(
                        [
                            html.Label("Color Scale"),
                            dcc.Dropdown(
                                id='color-scale-dropdown',
                                options=[
                                    {'label': 'Jet', 'value': 'Jet'},
                                    {'label': 'Viridis', 'value': 'Viridis'},
                                    {'label': 'Plasma', 'value': 'Plasma'},
                                    {'label': 'Turbo', 'value': 'Turbo'},
                                    {'label': 'Cividis', 'value': 'Cividis'},
                                ],
                                value='Jet',
                                clearable=False
                            ),
                        ],
                        id='color-scale-dropdown-container'
                    ),
                    html.Div([
                        html.Label("Color Range"),
                        dcc.Dropdown(
                            id='color-range-dropdown',
                            options=[
                                {'label': '-5 to 5', 'value': 'range_5'},
                                {'label': '-10 to 10', 'value': 'range_10'},
                                {'label': '-20 to 20', 'value': 'range_20'},
                                {'label': '-40 to 40', 'value': 'range_40'},
                                {'label': 'Custom',   'value': 'custom'}
                            ],
                            value='range_5',
                            clearable=False
                        ),
                        html.Div([
                            html.Label("Min:", style={'marginRight':'10px'}),
                            dcc.Input(id='custom-min-input', type='number', value=-5, style={'width':'80px'}),
                            html.Label("Max:", style={'margin':'0 10px 0 20px'}),
                            dcc.Input(id='custom-max-input', type='number', value=5, style={'width':'80px'}),
                        ], id='custom-range-container', style={'display':'none','marginTop':'10px'}),
                    ], id='color-range-dropdown-container'),
                    html.Br(),
                ])
            )
        ],
        id="legend-modal",
        is_open=False,  
        backdrop=False,
        content_style={"borderRadius": "12px", "backgroundColor": "white"},
        size="lg",
        style={"overflowY": "auto"}
    ),
    html.Div([
        html.Div([
            html.Label("Map Style"),
            dcc.Dropdown(
                id='map-style-dropdown',
                options=[
                    {'label': 'Satellite', 'value': 'satellite'},
                    {'label': 'Outdoors', 'value': 'outdoors'},
                    {'label': 'Light', 'value': 'light'},
                    {'label': 'Dark', 'value': 'dark'},
                    {'label': 'Streets', 'value': 'streets'}
                ],
                value='satellite',
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '16%', 'padding': '10px'}),

        html.Div([
            html.Label("Visualization Option"),
            dcc.Dropdown(
                id='color-mode-dropdown',
                options=[
                    {'label': 'Orbit Type', 'value': 'orbit'},
                    {'label': 'Displacement Mean Velocity [mm/year]', 'value': 'speed'},
                    {'label': 'Anomaly Type', 'value': 'anomaly_type'},
                    {'label': 'Prediction Velocity', 'value': 'prediction_velocity'},
                    {'label': 'Cumulative Displacement', 'value': 'actual_displacement_velocity'}
                ],
                value='orbit',
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '16%', 'padding': '10px'}),
        html.Div([
            html.Label("Filter by LOS Geometry"),
            dcc.Dropdown(
                id='orbit-filter-dropdown',
                options=[
                    {'label': 'Ascending 175', 'value': 'Ascending 175'},
                ],
                value='Ascending 175',
                multi=True,
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '16%', 'padding': '10px'}),
        html.Div([
            html.Label("Select Area of Interest"),
            dcc.Dropdown(
                id='area-dropdown',
                options=[
                    {'label': 'Wrocław', 'value': 'nysa'},
                ],
                value='nysa',
                clearable=False,
                persistence=True,
                persistence_type='memory',
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '16%', 'padding': '10px'}),
        html.Div([
            html.Label("Enable Distance Calculation"),
            dcc.Dropdown(
                id='distance-calc-dropdown',
                options=[
                    {'label': 'Yes', 'value': 'yes'},
                    {'label': 'No', 'value': 'no'}
                ],
                value='no',
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '16%', 'padding': '10px'}),
    ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),
    html.Div(id='distance-output', style={'font-size': '16px', 'padding': '10px', 'color': 'black'}),
    html.Div(
        id='prediction-method-container',
        children=[
            html.Label("Select Prediction Method"),
            dcc.Dropdown(
                id='prediction-method-dropdown',
                options=[
                    {'label': 'XGBoost', 'value': 'ml'}
                ],
                value='ml',
                clearable=False,
                style={'width': '100%'}
            )
        ],
        style={'display': 'none', 'padding': '10px'}
    ),

    html.Div([
        html.Label("Select Observation Range"),
        html.Div(id='selected-range-dates', style={'fontSize': '14px', 'margin': '10px 0'}),
        dcc.RangeSlider(
            id='dynamic-prediction-range-slider',
            min=1,
            max=60,
            step=1,
            marks={},
            value=[1, 5],
            tooltip={"placement": "bottom", "always_visible": True},
            allowCross=False
        )
    ], id='prediction-slider-container', style={'display': 'none', 'padding': '10px'}),
    dcc.Graph(
        id='map',
        style={'height': '80vh', 'width': '95vw'},
        config={'scrollZoom': True, 'doubleClick': False}
    ),
    dcc.Store(id='selected-points', data={'point_1': None, 'point_2': None}),
    html.Div(
        id='displacement-container',
        children=[
            html.Div([
                html.Label("Select Date Range", style={'fontSize': '18px', 'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                dcc.DatePickerRange(
                    id='date-range-picker',
                    start_date=all_data_nysa_ml['timestamp'].min(),
                    end_date=all_data_nysa_ml['timestamp'].max(),
                    display_format='YYYY-MM-DD',
                    style={
                        'fontSize': '16px'
                    }
                )
            ], style={'display': 'inline-block', 'padding': '15px', 'verticalAlign': 'top'}),
            html.Div([
                html.Label("Set Y-Axis Range (mm)", style={'fontSize': '18px', 'fontWeight': 'bold', 'marginBottom': '8px','display': 'block'}),
                dcc.Input(
                    id='y-axis-min',
                    type='number',
                    placeholder='Min',
                    style={'width': '120px', 'height': '45px', 'fontSize': '16px', 'marginRight': '10px'}
                ),
                dcc.Input(
                    id='y-axis-max',
                    type='number',
                    placeholder='Max',
                    style={'width': '120px', 'height': '45px', 'fontSize': '16px'}
                ),
            ], style={'display': 'inline-block', 'padding': '15px', 'verticalAlign': 'top'}),
            dcc.Graph(id='displacement-graph', style={'height': '50vh', 'width': '95vw'})
        ],
        style={'display': 'none'}
    ),
    html.Div([
        html.H5(style={'marginTop': '10px', 'marginBottom': '10px'}),
        dash_table.DataTable(
            id='point-attributes-table',
            columns=[
                {'name': 'Name', 'id': 'Name'},
                {'name': 'Value', 'id': 'Value'}
            ],
            data=[],
            style_cell={'textAlign': 'left'},
            style_header={'backgroundColor': 'white','fontWeight': 'bold'},
            style_table={'width': '50%', 'margin': 'auto'}
        )
    ], id='point-attributes-container', style={'display': 'none'}),
    html.Div(
            [
                html.Hr(style={"margin": "5px 0"}),
                html.P(
                    [
                        "This work was supported by the Wrocław University of Environmental "
                        "and Life Sciences (Poland) as part of the research project No. ",
                        html.A(
                            "N060/0004/23",
                            href="https://bazawiedzy.upwr.edu.pl/info/projectinternal/UPWR82df6d5513b84da5aab50c936b73f903/",
                            target="_blank",
                            style={
                                "color": "#0066cc",
                                "textDecoration": "underline",
                                "fontWeight": "500",
                            },
                        ),
                        ". For further information or inquiries regarding this project, please contact "
                        "Kamila Pawłuszek-Filipiak (email: ",
                        html.A(
                            "kamila.pawluszek-filipiak@upwr.edu.pl",
                            href="mailto:kamila.pawluszek-filipiak@upwr.edu.pl",
                            style={"color": "#0066cc"},
                        ),
                        ").",
                    ],
                    style={"textAlign": "center", "fontSize": "14px"},
                ),
            ],
            style={"padding": "10px"},
        ),
    ])


@app.callback(
    Output('color-range-dropdown-container', 'style'),
    Input('color-mode-dropdown', 'value')
)
def toggle_color_range_container(selected_mode):
    continuous_modes = ['speed', 'prediction_velocity', 'actual_displacement_velocity']
    if selected_mode in continuous_modes:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(
    Output('custom-range-container', 'style'),
    Input('color-range-dropdown', 'value')
)
def toggle_custom_range(range_choice):
    if range_choice == 'custom':
        return {'display': 'block','marginTop':'10px'}
    else:
        return {'display':'none'}
    
@app.callback(
    Output('color-scale-dropdown-container', 'style'),
    Input('color-mode-dropdown', 'value')
)
def toggle_color_scale_visibility(selected_mode):
    continuous_modes = ['speed', 'prediction_velocity', 'actual_displacement_velocity']

    if selected_mode in continuous_modes:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(
    Output("legend-modal", "is_open"),
    Input("open-legend-modal", "n_clicks"),
    State("legend-modal", "is_open")
)
def toggle_legend_modal(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

@app.callback(
    Output("help-modal", "is_open"),
    Input("open-help-modal", "n_clicks"),
    State("help-modal", "is_open")
)
def toggle_help_modal(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

@app.callback(
    Output('selected-range-dates', 'children'),
    Input('dynamic-prediction-range-slider', 'value'),
    State('area-dropdown', 'value')
)
def display_selected_dates(range_value, selected_area):
    start_val, end_val = range_value

    test_dates = (
        all_data_nysa_ml[
            all_data_nysa_ml['timestamp'] >= TEST_START_DATE
        ]
        .drop_duplicates(subset='timestamp')
        .sort_values('timestamp')
        .reset_index(drop=True)
    )

    date_list = test_dates['timestamp'].tolist()

    date_start = date_list[start_val - 1]
    date_end = date_list[end_val - 1]

    return (
        f"Selected date range: "
        f"{date_start.strftime('%Y-%m-%d')} "
        f"to "
        f"{date_end.strftime('%Y-%m-%d')}"
    )

@app.callback(
    Output('prediction-slider-container', 'style'),
    [Input('color-mode-dropdown', 'value')]
)
def toggle_prediction_slider_visibility(color_mode):
    if color_mode in ['prediction_velocity', 'actual_displacement_velocity']: 
        return {'display': 'block', 'padding': '10px'}
    else:
        return {'display': 'none', 'padding': '10px'}

@app.callback(
    Output('prediction-method-container', 'style'),
    Input('area-dropdown', 'value')
)
def toggle_prediction_method_dropdown(selected_area):
    if selected_area == 'nysa':
        return {'display': 'block', 'padding': '10px'}
    else:
        return {'display': 'none'}

@app.callback(
    [Output('orbit-filter-dropdown', 'options'), 
     Output('orbit-filter-dropdown', 'value'),
     Output('orbit-filter-dropdown', 'disabled')],
    [Input('area-dropdown', 'value')]
)
def update_orbit_filter(selected_area):
    if selected_area == 'nysa':
        return [{'label': 'Ascending 175', 'value': 'Ascending 175'}], 'Ascending 22', True

@app.callback(
    [Output('dynamic-prediction-range-slider', 'max'),
     Output('dynamic-prediction-range-slider', 'marks'),
     Output('dynamic-prediction-range-slider', 'value')],
    [Input('area-dropdown', 'value'),
     Input('color-mode-dropdown', 'value'),
     Input('prediction-method-dropdown', 'value')]
)
def update_slider_max(selected_area, color_mode, prediction_method):

    if color_mode == 'actual_displacement_velocity':
        max_val = {
            'nysa': MAX_ACTUAL_NYSA,
        }.get(selected_area, MAX_ACTUAL_NYSA)
    else:
        max_val = 60  
        
    data_for_area = {
        'nysa': all_data_nysa_ml,
    }.get(selected_area, all_data_nysa_ml)

    timestamps_df = (
        data_for_area[
            data_for_area['timestamp'] >= TEST_START_DATE
        ]
        .drop_duplicates(subset='timestamp')
        .sort_values('timestamp')
        .reset_index(drop=True)
    )

    timestamps_df['slider_step'] = np.arange(
        1,
        len(timestamps_df) + 1
    )

    if max_val > 60:
        N = 40 
    elif max_val > 20:
        N = 15  
    else:
        N = 5  

    marks = {}
    for _, row in timestamps_df.iterrows():
        step_val = row['slider_step']
        if step_val <= max_val:
            if step_val == 1 or step_val == max_val or step_val % N == 0:
                marks[step_val] = row['timestamp'].strftime('%Y-%m-%d')  
            else:
                marks[step_val] = "" 

    default_end = min(5, max_val)
    return max_val, marks, [1, default_end]

@app.callback(
    Output('map', 'figure'),
    [
        Input('map-style-dropdown', 'value'),
        Input('color-mode-dropdown', 'value'),
        Input('orbit-filter-dropdown', 'value'),
        Input('area-dropdown', 'value'),
        Input('dynamic-prediction-range-slider', 'value'),
        Input('prediction-method-dropdown', 'value'),
        Input('point-opacity-slider', 'value'),
        Input('point-size-slider', 'value'),
        Input('color-scale-dropdown', 'value'),  
        Input('color-range-dropdown', 'value'),  
        Input('custom-min-input', 'value'),      
        Input('custom-max-input', 'value')      
    ]
)
def update_map(map_style,color_mode,orbit_filter,selected_area,pred_range,prediction_method,point_opacity,point_size,
               color_scale_selected,range_choice,custom_min,custom_max):
    if selected_area == 'nysa':
        data = all_data_nysa_ml.drop_duplicates(subset=['pid'])
        center_coords = {'lat': data['latitude'].mean(), 'lon': data['longitude'].mean()}
        zoom_level = 11
        orbit_filter = ['Ascending 175']

    if isinstance(orbit_filter, str):
        orbit_filter = [orbit_filter]

    filtered_data = data[data['file'].isin(orbit_filter)].copy()
    filtered_data['mean_velocity'] = filtered_data['mean_velocity'].round(1)

    start_val, end_val = pred_range

    continuous_modes = ['speed', 'prediction_velocity', 'actual_displacement_velocity']

    if color_mode == 'prediction_velocity':
        if selected_area == 'nysa':
            max_steps = MAX_NYSA

        pred_key = (selected_area, prediction_method if selected_area in ['nysa'] else 'ml')
        prefix_pivot = prefix_data[pred_key]

        end_val = min(end_val, max_steps)
        start_val = min(start_val, max_steps)

        numerator = prefix_pivot[end_val] - prefix_pivot[start_val - 1]
        denominator = (end_val - start_val + 1)
        prediction_avg = numerator / denominator

        merged_data = filtered_data.set_index('pid')
        merged_data['prediction_velocity'] = prediction_avg
        merged_data.reset_index(inplace=True)

        if range_choice == 'range_5':
            vmin, vmax = -5, 5
        elif range_choice == 'range_10':
            vmin, vmax = -10, 10
        elif range_choice == 'range_20':
            vmin, vmax = -20, 20
        elif range_choice == 'range_40':
            vmin, vmax = -40, 40
        else:  
            vmin, vmax = custom_min, custom_max

        fig = px.scatter_mapbox(
            merged_data,
            lat='latitude',
            lon='longitude',
            hover_name='pid',
            hover_data={'latitude': True, 'longitude': True, 'height': True},
            color='prediction_velocity',
            color_continuous_scale=color_scale_selected,
            range_color=(vmin, vmax),
            labels={'latitude': 'Latitude','longitude': 'Longitude','height': 'Height'},
            zoom=zoom_level,
            opacity=point_opacity)
        fig.update_layout(legend_title_text='Prediction Velocity Average')

    elif color_mode == 'actual_displacement_velocity':
        if selected_area == 'nysa':
            max_steps = MAX_ACTUAL_NYSA
            prefix_pivot = actual_prefix_data['nysa']

        end_val = min(end_val, max_steps)
        start_val = min(start_val, max_steps)

        numerator = prefix_pivot[end_val] - prefix_pivot[start_val - 1]
        denominator = (end_val - start_val + 1)
        actual_avg = numerator / denominator

        merged_data = filtered_data.set_index('pid')
        merged_data['actual_displacement_velocity'] = actual_avg
        merged_data.reset_index(inplace=True)

        if range_choice == 'range_5':
            vmin, vmax = -5, 5
        elif range_choice == 'range_10':
            vmin, vmax = -10, 10
        elif range_choice == 'range_20':
            vmin, vmax = -20, 20
        elif range_choice == 'range_40':
            vmin, vmax = -40, 40
        else:  
            vmin, vmax = custom_min, custom_max

        fig = px.scatter_mapbox(
            merged_data,
            lat='latitude',
            lon='longitude',
            hover_name='pid',
            hover_data={'latitude': True, 'longitude': True, 'height': True},
            color='actual_displacement_velocity',
            color_continuous_scale=color_scale_selected,
            range_color=(vmin, vmax),
            labels={'latitude': 'Latitude','longitude': 'Longitude','height': 'Height'},
            zoom=zoom_level,
            opacity=point_opacity)
        fig.update_layout(legend_title_text='Actual Displacement Velocity Average')

    elif color_mode == 'anomaly_type':
        if selected_area == 'nysa':
            merged_data = filtered_data.merge(anomaly_data_nysa_99_ml[['pid','is_anomaly']], on='pid', how='left')

        merged_data['is_anomaly'] = merged_data['is_anomaly'].fillna(False).astype(bool)
        merged_data['consecutive_anomalies'] = (
            merged_data.groupby('pid')['is_anomaly'].rolling(3, min_periods=3).sum().reset_index(0, drop=True))
        merged_data['anomaly_3plus'] = merged_data['consecutive_anomalies'] >= 3

        fig = px.scatter_mapbox(
            merged_data,
            lat='latitude', lon='longitude',
            hover_name='pid',
            hover_data={'latitude': True, 'longitude': True, 'height': True},
            labels={'latitude': 'Latitude','longitude': 'Longitude','height': 'Height'},
            color=merged_data['anomaly_3plus'].map({True: 'Anomaly', False: 'No Anomaly'}),
            color_discrete_map={'Anomaly': 'red', 'No Anomaly': 'green'},
            zoom=zoom_level,
            opacity=point_opacity)
        fig.update_layout(legend_title_text='Anomaly Type')

    elif color_mode == 'orbit':
        fig = px.scatter_mapbox(
            filtered_data,
            lat='latitude',
            lon='longitude',
            hover_name='pid',
            hover_data={'latitude': True, 'longitude': True, 'height': True},
            labels={'latitude': 'Latitude','longitude': 'Longitude','height': 'Height'},
            color='file',
            zoom=zoom_level,
            opacity=point_opacity)
        fig.update_layout(legend_title_text='Orbit Type')

    else:
        if range_choice == 'range_5':
            vmin, vmax = -5, 5
        elif range_choice == 'range_10':
            vmin, vmax = -10, 10
        elif range_choice == 'range_20':
            vmin, vmax = -20, 20
        elif range_choice == 'range_40':
            vmin, vmax = -40, 40
        else:
            vmin, vmax = custom_min, custom_max

        fig = px.scatter_mapbox(
            filtered_data,
            lat='latitude',
            lon='longitude',
            hover_name='pid',
            hover_data={'latitude': True, 'longitude': True, 'height': True},
            color='mean_velocity',
            color_continuous_scale=color_scale_selected,
            range_color=(vmin, vmax),
            labels={'latitude': 'Latitude','longitude': 'Longitude','height': 'Height'},
            zoom=zoom_level,
            opacity=point_opacity)
        fig.update_layout(legend_title_text='Mean Velocity')

    fig.update_traces(marker=dict(size=point_size))

    if orbit_filter is not None and len(orbit_filter) > 0:
        annotation_lines = ["Orbit Geometry Info:<br>"]
        for orbit in orbit_filter:
            if orbit in orbit_geometry_info:
                info = orbit_geometry_info[orbit]
                annotation_lines.append(
                    f"<b>{orbit}</b>:<br>"
                    f"Relative orbit number: {info['Relative orbit number']}<br>"
                    f"View angle: {info['View angle']}<br>"
                    f"Mean Incidence angle: {info['Mean Incidence angle']}<br><br>")
                
        if len(annotation_lines) > 1:
            fig.add_annotation(
                text="".join(annotation_lines),
                xref="paper", yref="paper",
                x=1, y=1,
                showarrow=False,
                align="left",
                bordercolor="#cccccc",
                borderwidth=1,
                borderpad=4,
                bgcolor="white",
                opacity=0.8)

    fig.update_layout(
        mapbox_style=map_style,
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=dict(center=center_coords),
        coloraxis_colorbar=dict(title=None))

    return fig

@app.callback(
    Output('selected-points', 'data'),
    [Input('map', 'clickData')],
    [State('selected-points', 'data')]
)
def update_selected_points(clickData, selected_points):
    if clickData is None:
        return selected_points
    
    point_id = clickData['points'][0]['hovertext']
    lat = clickData['points'][0]['lat']
    lon = clickData['points'][0]['lon']
    
    if selected_points['point_1'] is None:
        selected_points['point_1'] = {'pid': point_id, 'lat': lat, 'lon': lon}
    elif selected_points['point_2'] is None:
        selected_points['point_2'] = {'pid': point_id, 'lat': lat, 'lon': lon}
    else:
        selected_points = {'point_1': None, 'point_2': None}

    return selected_points

@app.callback(
    Output('distance-output', 'children'),
    [Input('selected-points', 'data'),
     Input('distance-calc-dropdown', 'value')]
)
def display_distance(selected_points, distance_calc_enabled):
    if distance_calc_enabled == 'no':
        return ""

    point_1 = selected_points['point_1']
    point_2 = selected_points['point_2']
    
    if point_1 is not None and point_2 is not None:
        coords_1 = (point_1['lat'], point_1['lon'])
        coords_2 = (point_2['lat'], point_2['lon'])

        distance_km = geodesic(coords_1, coords_2).kilometers
        
        return html.Div([
            html.H4("Selected Points and Distance"),
            html.Ul([
                html.Li(f"Point 1: {point_1['pid']} (Lat: {point_1['lat']}, Lon: {point_1['lon']})"),
                html.Li(f"Point 2: {point_2['pid']} (Lat: {point_2['lat']}, Lon: {point_2['lon']})"),
                html.Li(f"Distance: {distance_km:.2f} km")
            ], style={'list-style-type': 'none', 'padding': '0', 'margin': '0'})
        ], style={'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px'})
    else:
        return "Select two points on the map to calculate the distance."

@app.callback(
    [Output('date-range-picker', 'start_date'),
     Output('date-range-picker', 'end_date'),
     Output('date-range-picker', 'min_date_allowed'),
     Output('date-range-picker', 'max_date_allowed')],
    [Input('area-dropdown', 'value')]
)
def update_date_picker(selected_area):
    if selected_area == 'nysa':
        start_date = all_data_nysa_ml['timestamp'].min()
        end_date = all_data_nysa_ml['timestamp'].max()

    return start_date, end_date, start_date, end_date

@app.callback(
    [Output('displacement-graph', 'figure'), 
     Output('displacement-container', 'style'),
     Output('point-attributes-table', 'data'),
     Output('point-attributes-container', 'style')],
    [Input('map', 'clickData'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('y-axis-min', 'value'),
     Input('y-axis-max', 'value'),
     Input('area-dropdown', 'value'),
     Input('prediction-method-dropdown', 'value')] 
)
def display_displacement(clickData, start_date, end_date, y_min, y_max, selected_area, prediction_method):
    if clickData is None:
        return {}, {'display': 'none'}, [], {'display': 'none'}

    point_id = clickData['points'][0]['hovertext']
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    if selected_area == 'nysa':
        if prediction_method == 'ml':
            full_data = all_data_nysa_ml[all_data_nysa_ml['pid'] == point_id].copy()
            anomaly_data_95 = anomaly_data_nysa_95_ml
            anomaly_data_99 = anomaly_data_nysa_99_ml
            last_n_data = full_data[full_data['timestamp'] >= TEST_START_DATE].copy()

    last_n_data.set_index('timestamp', inplace=True)
    filtered_data = full_data[(full_data['timestamp'] >= start_date) & (full_data['timestamp'] <= end_date)].copy()

    if not last_n_data.empty and (last_n_data.index.min() <= end_date) and (last_n_data.index.max() >= start_date):
        filtered_last_n_data = last_n_data[(last_n_data.index >= start_date) & (last_n_data.index <= end_date)].copy()
    else:
        filtered_last_n_data = pd.DataFrame()

    filtered_anomalies_95 = anomaly_data_95[anomaly_data_95['pid'] == point_id].copy()
    filtered_anomalies_99 = anomaly_data_99[anomaly_data_99['pid'] == point_id].copy()

    if not filtered_anomalies_95.empty:
        filtered_anomalies_95 = filtered_anomalies_95.tail(len(filtered_last_n_data)).copy()
        if len(filtered_anomalies_95) > 0:
            filtered_anomalies_95['timestamp'] = filtered_last_n_data.index.values[:len(filtered_anomalies_95)]
            filtered_anomalies_95.set_index('timestamp', inplace=True)
            filtered_last_n_data = filtered_last_n_data.join(
                filtered_anomalies_95[['predicted_value', 'upper_bound', 'lower_bound', 'is_anomaly']], 
                how='left'
            )

    if not filtered_anomalies_99.empty:
        filtered_anomalies_99 = filtered_anomalies_99.tail(len(filtered_last_n_data)).copy()
        if len(filtered_anomalies_99) > 0:
            filtered_anomalies_99['timestamp'] = filtered_last_n_data.index.values[:len(filtered_anomalies_99)]
            filtered_anomalies_99.set_index('timestamp', inplace=True)
            filtered_last_n_data = filtered_last_n_data.join(
                filtered_anomalies_99[['upper_bound', 'lower_bound', 'is_anomaly']], 
                how='left', rsuffix='_99'
            )

    fig = px.line(filtered_data, x='timestamp', y='displacement', 
                  markers=True, 
                  labels={'displacement': 'Displacement[mm]'})

    fig.add_scatter(x=filtered_data['timestamp'], y=filtered_data['displacement'], 
                    mode='lines+markers', 
                    name='InSAR measured displacement', 
                    line=dict(color='blue'))

    if not filtered_last_n_data.empty:
        if 'predicted_value' in filtered_last_n_data.columns:
            fig.add_scatter(x=filtered_last_n_data.index, y=filtered_last_n_data['predicted_value'], 
                            mode='lines+markers', 
                            name='Predicted Displacement', 
                            line=dict(color='orange'))

        if 'upper_bound' in filtered_last_n_data.columns:
            fig.add_scatter(x=filtered_last_n_data.index, y=filtered_last_n_data['upper_bound'], 
                            mode='lines', line=dict(color='yellow', dash='dash'),
                            name='Upper Bound p=95')

            fig.add_scatter(x=filtered_last_n_data.index, y=filtered_last_n_data['lower_bound'],
                            mode='lines', line=dict(color='yellow', dash='dash'),
                            fill='tonexty', fillcolor='rgba(255, 252, 127, 0.2)',
                            name='Lower Bound p=95')

        anomalies_95 = filtered_last_n_data[filtered_last_n_data['is_anomaly'] == 1]
        if not anomalies_95.empty:
            fig.add_scatter(x=anomalies_95.index, y=anomalies_95['displacement'], 
                            mode='markers', name='Anomalies p=95', 
                            marker=dict(color='yellow', size=10))

        if 'upper_bound_99' in filtered_last_n_data.columns:
            fig.add_scatter(x=filtered_last_n_data.index, y=filtered_last_n_data['upper_bound_99'], 
                            mode='lines', line=dict(color='red', dash='dash'),
                            name='Upper Bound p=99')

            fig.add_scatter(x=filtered_last_n_data.index, y=filtered_last_n_data['lower_bound_99'],
                            mode='lines', line=dict(color='red', dash='dash'),
                            fill='tonexty', fillcolor='rgba(254, 121, 104, 0.1)',
                            name='Lower Bound p=99')

            anomalies_99 = filtered_last_n_data[filtered_last_n_data['is_anomaly_99'] == 1]
            if not anomalies_99.empty:
                fig.add_scatter(x=anomalies_99.index, y=anomalies_99['displacement'], 
                                mode='markers', name='Anomalies p=99', 
                                marker=dict(color='red', size=10))

    if y_min is not None and y_max is not None:
        fig.update_yaxes(range=[y_min, y_max])

    fig.update_layout(
        xaxis_title='Date', 
        yaxis_title='Displacement LOS[mm]', 
        legend_title="Legend",
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.05)
    )
    
    displacement_data = full_data[full_data['pid'] == point_id]
    attributes = {
        'Point ID': point_id,
        'Mean Velocity [mm/yr]': f"{displacement_data['mean_velocity'].mean():.2f}",
        'Minimum Displacement [mm]': f"{displacement_data['displacement'].min():.2f}",
        'Maximum Displacement [mm]': f"{displacement_data['displacement'].max():.2f}"
    }

    attributes_data = [{'Name': key, 'Value': value} for key, value in attributes.items()]

    return fig, {'display': 'block'}, attributes_data, {'display': 'block'}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
