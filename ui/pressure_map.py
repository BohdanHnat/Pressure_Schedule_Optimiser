"""
ui/pressure_map.py — EPANET network on OpenStreetMap tiles.

Only CONSUMER nodes (non-zero base demand) are shown on the map.
Discrete 3-zone colouring: RED / GREEN / YELLOW / RED.
Pipe connections drawn as thin lines matching the EPANET model.

OOP design note:
  NetworkGeometry replaces the original _get_network_geometry() function which
  returned a plain dictionary with 11 keys. The dataclass makes the structure
  explicit, gives attribute access (geom.consumer_x vs geom["consumer_x"]),
  and co-locates the extraction logic with the data it produces via a
  @classmethod factory method.

  The Plotly figure construction in _render_mapbox() and _render_scatter()
  cannot be simplified — every go.Scattermapbox() call is a direct specification
  of what Plotly renders. There is no shorter way to produce an interactive
  geographic map with coloured markers using Plotly's API.
"""
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass, field

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

_BG     = "#0d1117"
_RED    = "#f85149"
_GREEN  = "#3fb950"
_YELLOW = "#d29922"


def _pressure_color(p: float) -> str:
    if p < C.P_MIN_CONSUMER:
        return _RED
    elif p <= 3.7:
        return _GREEN
    elif p <= C.P_MAX_CONSUMER:
        return _YELLOW
    else:
        return _RED


@dataclass
class NetworkGeometry:
    """
    Holds all geometric data extracted from a WNTR water network model.

    Constructed via NetworkGeometry.from_water_network(wn) rather than
    directly, since extraction requires iterating the WNTR model object.
    """
    consumer_x: list = field(default_factory=list)
    consumer_y: list = field(default_factory=list)
    consumer_names: list = field(default_factory=list)
    edge_x: list = field(default_factory=list)
    edge_y: list = field(default_factory=list)
    station_xy: tuple = None
    pump_midpoints: list = field(default_factory=list)
    reservoir_points: list = field(default_factory=list)
    valve_points: list = field(default_factory=list)
    n_consumer: int = 0
    n_junctions: int = 0

    @classmethod
    def from_water_network(cls, wn) -> "NetworkGeometry":
        """Extract all geometry from a WNTR water network object."""
        geom = cls()

        # All nodes — needed for pipe endpoint lookup
        all_coords = {}
        for jname in wn.junction_name_list:
            all_coords[jname] = wn.get_node(jname).coordinates
        for rname in wn.reservoir_name_list:
            all_coords[rname] = wn.get_node(rname).coordinates
        for tname in wn.tank_name_list:
            all_coords[tname] = wn.get_node(tname).coordinates

        # Consumer nodes only — non-zero base demand
        for jname in wn.junction_name_list:
            j = wn.get_node(jname)
            try:
                has_demand = any(d.base_value > 0 for d in j.demand_timeseries_list)
            except (AttributeError, TypeError):
                has_demand = getattr(j, 'base_demand', 0) > 0
            if has_demand:
                coords = j.coordinates
                geom.consumer_x.append(coords[0])
                geom.consumer_y.append(coords[1])
                geom.consumer_names.append(jname)

        # Pipe edges — thin lines connecting nodes as in the EPANET model
        for pname in wn.pipe_name_list:
            pipe = wn.get_link(pname)
            s_name = pipe.start_node_name
            e_name = pipe.end_node_name
            if s_name in all_coords and e_name in all_coords:
                s = all_coords[s_name]
                e = all_coords[e_name]
                geom.edge_x += [s[0], e[0], None]
                geom.edge_y += [s[1], e[1], None]

        # Pump midpoints
        for pid in C.PUMP_IDS:
            if pid in wn.pump_name_list:
                pump = wn.get_link(pid)
                s = wn.get_node(pump.start_node_name).coordinates
                e = wn.get_node(pump.end_node_name).coordinates
                geom.pump_midpoints.append(((s[0]+e[0])/2, (s[1]+e[1])/2, pid))

        # Station position = slightly above the topmost pump centroid
        if geom.pump_midpoints:
            avg_x = sum(p[0] for p in geom.pump_midpoints) / len(geom.pump_midpoints)
            max_y = max(p[1] for p in geom.pump_midpoints)
            geom.station_xy = (avg_x, max_y + 0.00012)
        elif wn.reservoir_name_list:
            r = wn.get_node(wn.reservoir_name_list[0]).coordinates
            geom.station_xy = (r[0], r[1])

        # Reservoir and valve positions for labelling
        for rname in wn.reservoir_name_list:
            c = wn.get_node(rname).coordinates
            geom.reservoir_points.append((c[0], c[1], rname))

        for vname in wn.valve_name_list:
            v = wn.get_link(vname)
            s = wn.get_node(v.start_node_name).coordinates
            e = wn.get_node(v.end_node_name).coordinates
            geom.valve_points.append(((s[0]+e[0])/2, (s[1]+e[1])/2, vname))

        geom.n_consumer  = len(geom.consumer_names)
        geom.n_junctions = len(wn.junction_name_list)
        return geom


def render_pressure_map(results) -> None:
    wn                = results.wn
    node_pressures_ga = results.node_pressures_ga
    ga_schedule       = results.ga_schedule

    geom = NetworkGeometry.from_water_network(wn)

    # NOTE: Streamlit — st.slider returns the current value on every rerun.
    # The slider widget is Streamlit-specific; no plain-Python equivalent.
    hour = st.slider(
        "Select hour:",
        min_value=0, max_value=23, value=8, step=1,
        format="%02d:00", key="pmap_hour",
        label_visibility="collapsed",
    )

    p_set = float(ga_schedule[hour])

    # Horizontal info bar: outlet pressure + colour legend
    st.markdown(
        f"<div class='map-info-bar'>"
        f"<div class='map-info-bar-pressure'>"
        f"<span class='map-info-bar-label'>Station Outlet Pressure</span>"
        f"<span class='map-info-bar-hour'>{hour:02d}:00</span>"
        f"<span class='map-info-bar-value'>{p_set:.2f} bar</span>"
        f"</div>"
        f"<div class='map-info-bar-legend'>"
        f"<div class='legend-h-item'>"
        f"<span class='legend-h-dot dot-red'></span>"
        f"<span class='legend-h-text'>Critical Low&ensp;&lt;&ensp;"
        f"<span style='font-size:17px'>2.5</span></span>"
        f"</div>"
        f"<div class='legend-h-item'>"
        f"<span class='legend-h-dot dot-green'></span>"
        f"<span class='legend-h-text'>Acceptable&ensp;"
        f"<span style='font-size:17px'>2.5</span>&ensp;&ndash;&ensp;"
        f"<span style='font-size:17px'>3.7</span></span>"
        f"</div>"
        f"<div class='legend-h-item'>"
        f"<span class='legend-h-dot dot-yellow'></span>"
        f"<span class='legend-h-text'>Caution&ensp;"
        f"<span style='font-size:17px'>3.7</span>&ensp;&ndash;&ensp;"
        f"<span style='font-size:17px'>6.0</span></span>"
        f"</div>"
        f"<div class='legend-h-item'>"
        f"<span class='legend-h-dot dot-red'></span>"
        f"<span class='legend-h-text'>Critical High&ensp;&gt;&ensp;"
        f"<span style='font-size:17px'>6.0</span></span>"
        f"</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Build pressure arrays for consumer nodes only
    hour_data   = node_pressures_ga.get(hour, {})
    node_p      = [hour_data.get(n, 0.0) for n in geom.consumer_names]
    node_colors = [_pressure_color(p) for p in node_p]
    hover_texts = [f"{n}: {p:.2f} bar" for n, p in zip(geom.consumer_names, node_p)]

    # Heuristic: coordinates in the range (20–45 lon, 40–55 lat) are geographic
    # (covers Ukraine and surrounding region). Outside this range, use plain scatter.
    lons = geom.consumer_x
    lats = geom.consumer_y
    avg_lon = sum(lons) / len(lons) if lons else 0
    avg_lat = sum(lats) / len(lats) if lats else 0
    is_geographic = (20 < avg_lon < 45) and (40 < avg_lat < 55)

    if is_geographic:
        _render_mapbox(geom, node_colors, hover_texts, lons, lats, avg_lon, avg_lat)
    else:
        _render_scatter(geom, node_colors, hover_texts)


def _render_mapbox(geom: NetworkGeometry, node_colors, hover_texts,
                   lons, lats, avg_lon, avg_lat):
    """
    Render on OpenStreetMap tiles using Plotly Scattermapbox.

    NOTE: The go.Figure() / add_trace() / update_layout() calls below are the
    Plotly library's API for building an interactive geographic map. There is no
    shorter way to produce this output using Plotly — each parameter specifies
    exactly one visual property of the rendered figure.
    """
    fig = go.Figure()

    # Pipe edges — rebuild from edge_x/edge_y (which use None as segment separator)
    edge_lons, edge_lats = [], []
    for i in range(0, len(geom.edge_x), 3):
        if i + 1 < len(geom.edge_x):
            edge_lons += [geom.edge_x[i], geom.edge_x[i+1], None]
            edge_lats += [geom.edge_y[i], geom.edge_y[i+1], None]

    fig.add_trace(go.Scattermapbox(
        lon=edge_lons, lat=edge_lats,
        mode="lines",
        line=dict(color="rgba(40,80,160,0.7)", width=2.5),
        hoverinfo="skip", showlegend=False,
    ))

    # Consumer nodes — coloured by pressure zone
    fig.add_trace(go.Scattermapbox(
        lon=lons, lat=lats,
        mode="markers",
        marker=dict(size=11, color=node_colors),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    # Pumping station marker
    if geom.station_xy:
        sx, sy = geom.station_xy
        fig.add_trace(go.Scattermapbox(
            lon=[sx], lat=[sy],
            mode="markers+text",
            marker=dict(size=22, color="rgba(40,80,160,0.9)"),
            text=["Pumping Station"],
            textfont=dict(color="#58a6ff", size=11),
            textposition="top center",
            hovertemplate="<b>Pumping Station</b><extra></extra>",
            showlegend=False,
        ))

    # Pump labels
    for mx, my, pid in geom.pump_midpoints:
        fig.add_trace(go.Scattermapbox(
            lon=[mx], lat=[my],
            mode="markers+text",
            marker=dict(size=12, color="#3fb950", symbol="circle"),
            text=[pid], textfont=dict(color="#3fb950", size=10),
            textposition="top center",
            hovertemplate=f"<b>{pid}</b><br>Pump<extra></extra>",
            showlegend=False,
        ))

    # Reservoir labels
    for rx, ry, rname in geom.reservoir_points:
        fig.add_trace(go.Scattermapbox(
            lon=[rx], lat=[ry],
            mode="markers+text",
            marker=dict(size=14, color="#6e7681"),
            text=[rname], textfont=dict(color="#6e7681", size=9),
            textposition="top center",
            hovertemplate=f"<b>{rname}</b><br>Reservoir<extra></extra>",
            showlegend=False,
        ))

    # Valve labels
    for vx, vy, vname in geom.valve_points:
        fig.add_trace(go.Scattermapbox(
            lon=[vx], lat=[vy],
            mode="markers+text",
            marker=dict(size=10, color="#6e7681"),
            text=[vname], textfont=dict(color="#6e7681", size=9),
            textposition="top center",
            hovertemplate=f"<b>{vname}</b><br>Valve<extra></extra>",
            showlegend=False,
        ))

    lon_spread = max(lons) - min(lons)
    lat_spread = max(lats) - min(lats)
    spread = max(lon_spread, lat_spread)
    zoom = 16 if spread < 0.005 else 15.5 if spread < 0.01 else 15 if spread < 0.02 else 14

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lon=avg_lon, lat=avg_lat),
            zoom=zoom,
        ),
        height=650,
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(family="monospace", size=13),
    )

    st.plotly_chart(fig, width='stretch', config={
        "scrollZoom": True,
        "modeBarButtonsToRemove": [
            "toImage", "pan2d", "select2d", "lasso2d",
            "zoomIn2d", "zoomOut2d", "autoScale2d",
            "zoomInMapbox", "zoomOutMapbox",
        ],
        "displayModeBar": True,
    })


def _render_scatter(geom: NetworkGeometry, node_colors, hover_texts):
    """
    Fallback renderer using plain Plotly Scatter (local/non-geographic coordinates).

    NOTE: Same Plotly API constraints apply here as in _render_mapbox — the
    go.Scatter() calls are direct Plotly specifications with no simpler alternative.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=geom.edge_x, y=geom.edge_y,
        mode="lines", line=dict(color="rgba(150,180,220,0.4)", width=1),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=geom.consumer_x, y=geom.consumer_y,
        mode="markers",
        marker=dict(size=11, color=node_colors,
                    line=dict(width=0.5, color="#0d1117")),
        text=hover_texts, hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    if geom.station_xy:
        sx, sy = geom.station_xy
        fig.add_trace(go.Scatter(
            x=[sx], y=[sy], mode="markers+text",
            marker=dict(size=26, color="rgba(40,80,160,0.9)", symbol="circle",
                        line=dict(width=2, color="#58a6ff")),
            text=["Pumping Station"],
            textfont=dict(color="#58a6ff", size=10, family="monospace"),
            textposition="top center",
            hovertemplate="<b>Pumping Station</b><extra></extra>",
            showlegend=False,
        ))

    for rx, ry, rname in geom.reservoir_points:
        fig.add_trace(go.Scatter(
            x=[rx], y=[ry], mode="markers+text",
            marker=dict(size=16, color="#6e7681", symbol="circle",
                        line=dict(width=1, color="#484f58")),
            text=[rname], textfont=dict(color="#8b949e", size=9, family="monospace"),
            textposition="top center",
            hovertemplate=f"<b>{rname}</b><br>Reservoir<extra></extra>",
            showlegend=False,
        ))

    for vx, vy, vname in geom.valve_points:
        fig.add_trace(go.Scatter(
            x=[vx], y=[vy], mode="markers+text",
            marker=dict(size=12, color="#6e7681", symbol="diamond",
                        line=dict(width=1, color="#484f58")),
            text=[vname], textfont=dict(color="#8b949e", size=9, family="monospace"),
            textposition="top center",
            hovertemplate=f"<b>{vname}</b><br>Valve<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        height=650, margin=dict(l=10, r=10, t=10, b=20),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                   scaleanchor="x", scaleratio=1),
        font=dict(family="monospace", color="#e6edf3", size=13),
    )
    st.plotly_chart(fig, width='stretch', config={
        "scrollZoom": True,
        "modeBarButtonsToRemove": [
            "toImage", "pan2d", "select2d", "lasso2d",
            "zoomIn2d", "zoomOut2d", "autoScale2d",
        ],
        "displayModeBar": True,
    })
