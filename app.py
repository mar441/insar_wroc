from functools import lru_cache
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc

WROC_TS_PATH   = r"wroclaw_caly_filtered.csv"        
WROC_GEO_PATH  = r"wroclaw_geo_filtered.csv"        
WROC_PRED_PATH = r"predictions_ml_filtered.csv"      
WROC_A95_PATH  = r"anomaly_95_filtered.csv" 
WROC_A99_PATH  = r"anomaly_99_filtered.csv"     

px.set_mapbox_access_token("pk.eyJ1IjoibnBpZWsiLCJhIjoiY21jN2tvYXE1MTRqeTJrc2NtaTlvNXQyZSJ9.abg-EkfnNKp7bgwEvgRp0w")

geo_data_wroc = pd.read_csv(
    WROC_GEO_PATH,
    usecols=["pid", "latitude", "longitude", "height"],
    dtype={
        "pid": "string",
        "latitude": "float32",
        "longitude": "float32",
        "height": "float32",
    },
)
geo_data_wroc["pid"] = geo_data_wroc["pid"].str.strip()

def read_insar_pid(pid: str) -> pd.DataFrame:
    """
    InSAR time series for a single pid from wide CSV: Date + pid columns.
    """
    pid = str(pid).strip()
    df = pd.read_csv(
        WROC_TS_PATH,
        usecols=["Date", pid],
        dtype={pid: "float32"},
    )
    df["timestamp"] = pd.to_datetime(df["Date"])
    df.rename(columns={pid: "displacement"}, inplace=True)
    df = df.drop(columns=["Date"]).dropna(subset=["displacement"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def read_pred_pid(pid: str) -> pd.DataFrame:
    """
    Prediction series for a single pid from wide CSV: Date + pid columns.
    """
    pid = str(pid).strip()
    df = pd.read_csv(
        WROC_PRED_PATH,
        usecols=["Date", pid],
        dtype={pid: "float32"},
    )
    df["timestamp"] = pd.to_datetime(df["Date"])
    df.rename(columns={pid: "predicted_value"}, inplace=True)
    df = df.drop(columns=["Date"]).dropna(subset=["predicted_value"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


ANOM_USECOLS = ["pid", "lower_bound", "upper_bound", "actual_value", "predicted_value", "is_anomaly"]

ANOM_DTYPES = {
    "pid": "string",
    "lower_bound": "float32",
    "upper_bound": "float32",
    "actual_value": "float32",
    "predicted_value": "float32",
    "is_anomaly": "string",   
}

def _normalize_is_anomaly(s: pd.Series) -> pd.Series:
    """
    Convert many encodings to int 0/1:
    True/False, 'True'/'False', 1/0, yes/no etc.
    """
    s = s.astype(str).str.strip().str.lower()
    return s.isin(["true", "1", "t", "yes", "y"]).astype("int8")


def read_anom_pid_chunked(path: str, pid: str, chunksize: int = 200_000) -> pd.DataFrame:
    """
    Read anomaly rows for one pid from long CSV by chunking.
    Keeps RAM low.
    """
    pid = str(pid).strip()
    out = []
    for ch in pd.read_csv(path, chunksize=chunksize, usecols=ANOM_USECOLS, dtype=ANOM_DTYPES):
        ch["pid"] = ch["pid"].str.strip()
        part = ch[ch["pid"] == pid]
        if not part.empty:
            part = part.copy()
            part["is_anomaly"] = _normalize_is_anomaly(part["is_anomaly"])
            out.append(part)

    if not out:
        return pd.DataFrame()

    df = pd.concat(out, ignore_index=True)
    df["pid"] = df["pid"].astype(str).str.strip()
    df["is_anomaly"] = _normalize_is_anomaly(df["is_anomaly"])
    df = df.reset_index(drop=True)
    return df

@lru_cache(maxsize=512)
def cached_insar(pid: str) -> pd.DataFrame:
    return read_insar_pid(pid)

@lru_cache(maxsize=512)
def cached_pred(pid: str) -> pd.DataFrame:
    return read_pred_pid(pid)

@lru_cache(maxsize=512)
def cached_a95(pid: str) -> pd.DataFrame:
    return read_anom_pid_chunked(WROC_A95_PATH, pid)

@lru_cache(maxsize=512)
def cached_a99(pid: str) -> pd.DataFrame:
    return read_anom_pid_chunked(WROC_A99_PATH, pid)

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
server = app.server


app.layout = html.Div(
    [
        html.H3("UPWR InSAR-based time series monitoring, prediction and anomaly detection platform"),

        dcc.Graph(
            id="map",
            style={"height": "80vh", "width": "95vw"},
            config={"scrollZoom": True, "doubleClick": False},
        ),

        html.Div(
            id="displacement-container",
            children=[
                dcc.Graph(id="displacement-graph", style={"height": "50vh", "width": "95vw"})
            ],
            style={"display": "none"},
        ),

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
    ]
)

@app.callback(
    Output("map", "figure"),
    Input("map", "id"),
)
def update_map(_):
    map_style = "satellite"
    data = geo_data_wroc.dropna(subset=["latitude", "longitude"]).copy()

    center_coords = {
        "lat": float(data["latitude"].mean()),
        "lon": float(data["longitude"].mean()),
    }

    fig = px.scatter_mapbox(
        data,
        lat="latitude",
        lon="longitude",
        hover_name="pid",
        hover_data={"latitude": True, "longitude": True, "height": True},
        zoom=13,
        opacity=0.9,
    )
    fig.update_traces(marker=dict(size=7, color="blue"))

    fig.update_layout(
        mapbox_style=map_style,
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=dict(center=center_coords),
    )
    return fig


@app.callback(
    [Output("displacement-graph", "figure"), Output("displacement-container", "style")],
    Input("map", "clickData"),
)
def display_displacement(clickData):
    if clickData is None:
        return {}, {"display": "none"}

    point_id = clickData["points"][0]["hovertext"].strip()

    insar = cached_insar(point_id)
    if insar is None or insar.empty:
        return {}, {"display": "none"}

    pred = cached_pred(point_id)

    a95 = cached_a95(point_id)
    a99 = cached_a99(point_id)

    if pred is not None and not pred.empty:
        pred = pred.sort_values("timestamp").reset_index(drop=True)

    if pred is not None and not pred.empty and a95 is not None and not a95.empty:
        n = min(len(pred), len(a95))
        pred_tail = pred.tail(n).reset_index(drop=True)
        a95 = a95.tail(n).reset_index(drop=True)
        a95["timestamp"] = pred_tail["timestamp"].values

    if pred is not None and not pred.empty and a99 is not None and not a99.empty:
        n = min(len(pred), len(a99))
        pred_tail = pred.tail(n).reset_index(drop=True)
        a99 = a99.tail(n).reset_index(drop=True)
        a99["timestamp"] = pred_tail["timestamp"].values

    fig = px.line(
        insar,
        x="timestamp",
        y="displacement",
        markers=True,
        labels={"displacement": "Displacement[mm]"},
    )
    fig.add_scatter(
        x=insar["timestamp"],
        y=insar["displacement"],
        mode="lines+markers",
        name="InSAR measured displacement",
        line=dict(color="blue"),
    )

    if pred is not None and not pred.empty:
        fig.add_scatter(
            x=pred["timestamp"],
            y=pred["predicted_value"],
            mode="lines+markers",
            name="Predicted Displacement",
            line=dict(color="orange"),
        )

    if a95 is not None and not a95.empty and "timestamp" in a95.columns:
        fig.add_scatter(
            x=a95["timestamp"],
            y=a95["upper_bound"],
            mode="lines",
            line=dict(color="yellow", dash="dash"),
            name="Upper Bound p=95",
        )
        fig.add_scatter(
            x=a95["timestamp"],
            y=a95["lower_bound"],
            mode="lines",
            line=dict(color="yellow", dash="dash"),
            fill="tonexty",
            fillcolor="rgba(255, 252, 127, 0.2)",
            name="Lower Bound p=95",
        )

        an95 = a95[a95["is_anomaly"] == 1]
        if not an95.empty:
            fig.add_scatter(
                x=an95["timestamp"],
                y=an95["actual_value"],
                mode="markers",
                name="Anomalies p=95",
                marker=dict(color="yellow", size=10),
            )

    if a99 is not None and not a99.empty and "timestamp" in a99.columns:
        fig.add_scatter(
            x=a99["timestamp"],
            y=a99["upper_bound"],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Upper Bound p=99",
        )
        fig.add_scatter(
            x=a99["timestamp"],
            y=a99["lower_bound"],
            mode="lines",
            line=dict(color="red", dash="dash"),
            fill="tonexty",
            fillcolor="rgba(254, 121, 104, 0.1)",
            name="Lower Bound p=99",
        )

        an99 = a99[a99["is_anomaly"] == 1]
        if not an99.empty:
            fig.add_scatter(
                x=an99["timestamp"],
                y=an99["actual_value"],
                mode="markers",
                name="Anomalies p=99",
                marker=dict(color="red", size=10),
            )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Displacement LOS[mm]",
        legend_title="Legend",
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.05),
    )

    return fig, {"display": "block"}


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8060))
    app.run_server(host="0.0.0.0", port=port, debug=False)
