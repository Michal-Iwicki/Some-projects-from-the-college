import streamlit as st
import pandas as pd
import time
from cassandra_utils import *
from utils import *
from datetime import datetime
import numpy as np
import pydeck as pdk
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt

# st_autorefresh(interval=10000, key="dataframerefresh")


st.set_page_config(layout="wide", page_title="✈️ Live Flight Dashboard")
st.title("✈️ Live Flight Dashboard")


# OLD CODE TO LOAD FROM CASSANDRA DIRECTLY
#
# @st.cache_data()
# def load_cassandra_data():
#     rows = session.execute(
#         f"SELECT * FROM {TABLE_ACTUAL}"
#     )
#     data = [row._asdict() for row in rows]
#     df = pd.DataFrame(data)
    
#     # Sort by timestamp if available
#     for col in ["simulation_timestamp", "event_time", "timestamp", "minute", "datetime"]:
#         if col in df.columns:
#             try:
#                 df[col] = pd.to_datetime(df[col], errors='coerce')
#                 df = df.sort_values(col, ascending=False)
#                 break
#             except:
#                 continue
    
#     return df
#


# Load current data from Cassandra
df = load_actual_data()

df["planned_dep_timestamp"] = pd.to_datetime(
    df["planned_dep_timestamp"].astype("int64") // 1_000_000,
    unit="s",
    origin="unix",
    errors="coerce"
)


# Auto-refresh
if st.button("🔄 Refresh Now"):
    st.cache_data.clear()
    st.rerun()

# --------------------------------------------------
# DASHBOARD TABS
# --------------------------------------------------
if df.empty:
    st.warning("No data available from Cassandra")
else:
    # Prepare data for dashboard
    flight_df = prepare_flight_data(df.copy())
    departured_df = prepare_departed_flights(df.copy(), n_timestamps=10)
    weather_data = extract_weather_data_with_changes(df)
    seismic_df = extract_seismic_data(df)
    # --------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------

    # Load aggregated data
    monthly_df = load_monthly_data()
    weekly_df = load_weekly_data()
    
    # Create tabs
    tab1, tab3, tab2, tab4, tab5, tab6 = st.tabs(["🛫 Departure Board", "🌍 Environment", "✈️ Find My Flight", "📈 Monthly Trends", "📅 Weekly Trends", "🔗 Relationships"])
    
    
    # ---------------------- TAB 1: Departure Board ----------------------
    with tab1:
        # ---------- Header ----------
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📊 Flight Overview")
        with col2:
            latest_time = flight_df["timestamp"].max()
            if hasattr(latest_time, "strftime"):
                st.metric("🕒 Current Airport Time", latest_time.strftime("%H:%M"))
            else:
                st.metric("🕒 Current Airport Time", str(latest_time))

        # ---------- Prepare board data ----------
        display_cols = [
            "Flight_Num",
            "Dest",
            "Airline Name",
            "Planned_Dep_Timestamp",
            "Status",
        ]

        # Add sorting flag
        if not flight_df.empty:
            flight_df["Is_Departed"] = False
        if not departured_df.empty:
            departured_df["Is_Departed"] = True

        # Keep only required columns
        flight_disp = flight_df[[c for c in display_cols + ["Is_Departed"] if c in flight_df.columns]]
        departed_disp = departured_df[[c for c in display_cols + ["Is_Departed"] if c in departured_df.columns]]

        # Merge
        board_df = pd.concat([departed_disp, flight_disp], ignore_index=True)

        # Remove invalid rows
        board_df = board_df.dropna(subset=["Flight_Num"])

        # Sort:
        # 1️⃣ Departed first
        # 2️⃣ Then by planned departure time
        board_df = board_df.sort_values(
            by=["Is_Departed", "Planned_Dep_Timestamp"],
            ascending=[False, True],
        )

        # ---------- Render board ----------
        if not board_df.empty:
            blink = int(datetime.now().second) % 2 == 0

            # Column headers
            header_cols = st.columns([1, 1, 2, 2, 2])
            header_cols[0].markdown("### 🕒 Time")
            header_cols[1].markdown("### ✈️ Flight")
            header_cols[2].markdown("### 📍 Destination")
            header_cols[3].markdown("### 🏢 Airline")
            header_cols[4].markdown("### 📌 Status")

            st.markdown(
                "<hr style='margin:5px 0 20px 0; border:0; border-top:1px solid #ccc;'>",
                unsafe_allow_html=True,
            )

            # Rows
            for _, flight in board_df.iterrows():
                cols = st.columns([1, 1, 2, 2, 2])

                # Time
                cols[0].markdown(
                    f"**{format_ts(flight['Planned_Dep_Timestamp'])}**"
                    if pd.notna(flight.get("Planned_Dep_Timestamp"))
                    else "**--:--**"
                )

                # Flight number
                cols[1].markdown(f"**{flight.get('Flight_Num', 'N/A')}**")

                # Destination
                cols[2].markdown(f"**{flight.get('Dest', 'Unknown')}**")

                # Airline
                cols[3].markdown(f"**{flight.get('Airline Name', 'Unknown Airline')}**")

                # Status
                status = flight.get("Status", "Unknown")
                style = status_style(status, blink)
                cols[4].markdown(
                    f"<div style='background-color:{style['background-color']};"
                    f"padding:5px;border-radius:3px;text-align:center;font-weight:bold'>"
                    f"{status}</div>",
                    unsafe_allow_html=True,
                )

        else:
            st.info("No flight data available for the departure board.")
    
# ---------------------- TAB 2: Find My Flight ----------------------
    with tab2:
        st.subheader("🔍 Find Your Flight")
        
        # Create a more visually appealing search section
        with st.container():
            st.markdown("### Enter Flight Details")
            
            col_search1, col_search2 = st.columns([3, 1])
            
            with col_search1:
                if 'Flight_Num' in flight_df.columns:
                    flight_options = flight_df['Flight_Num'].dropna().unique()
                    flight_num = st.selectbox(
                        "Flight Number:",
                        options=flight_options,
                    )
                else:
                    flight_num = st.text_input(
                        "Flight Number:",
                        placeholder="e.g., AA123, DL456"
                    ).strip().upper()
            
            with col_search2:
                st.markdown(
                    "<div style='height:29px'></div>",
                    unsafe_allow_html=True
                )
                search_button = st.button(
                    "🔍 Search Flight",
                    use_container_width=True,
                    type="primary"
                )
        
        if search_button and flight_num:
            try:
                # Try to find flight
                if 'Flight_Num' in flight_df.columns:
                    mask = flight_df['Flight_Num'].astype(str).str.contains(str(flight_num), case=False, na=False)
                    result_df = flight_df[mask]
                    
                    if not result_df.empty:
                        flight = result_df.iloc[0]
                        
                        # Create a visually appealing header
                        st.markdown(f"## ✈️ Flight {flight['Flight_Num']}")
                        
                        # Determine if flight is delayed
                        is_delayed = False
                        if 'Status' in flight:
                            is_delayed = 'DELAY' in str(flight['Status']).upper()
                        elif 'Delay' in flight:
                            is_delayed = float(flight['Delay'] or 0) > 0
                        
                        # Status indicator with color
                        status_color = "🟢"  # Green
                        if is_delayed:
                            status_color = "🔴"  # Red
                        elif 'Status' in flight and 'CANCELLED' in str(flight['Status']).upper():
                            status_color = "⚫"  # Black
                        
                        # Basic info in cards
                        # st.info(f"{status_color} **Current Status:** {flight.get('Status', 'N/A')}")
                        
                        # Create columns for info display
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                label="Airline",
                                value=flight.get('Airline Name', 'N/A'),
                                help="Operating airline"
                            )
                            planned_dep = flight.get('Planned_Dep_Timestamp', 'N/A')
                            if planned_dep != 'N/A':
                                st.metric(
                                    label="Planned Departure",
                                    value=format_ts(planned_dep),
                                    delta=None
                                )
                        
                        with col2:
                            st.metric(
                                label="Destination",
                                value=flight.get('Dest', 'N/A'),
                                help="Arrival airport"
                            )
                            status = flight.get('Status', 'N/A')
                            status_icon = "🟢"
                            if status.find("Delayed") != -1:
                                status = "Delayed"
                                status_icon = "🔴"

                            st.metric(
                                label="Status",
                                value=f"{status_icon} {status}",
                                help="Flight status information",
                            )
                        
                        with col3:
                            # Show prediction if flight is not yet delayed
                            if not is_delayed and 'probability' in flight:
                                pred_value = float(flight['probability'])
                                st.markdown(
                                    "<div style='height:30px'></div>",
                                    unsafe_allow_html=True
                                )
                                
                                # Determine risk level
                                if pred_value < 0.3:
                                    risk_level = "Low"
                                    risk_color = "green"
                                    emoji = "✅"
                                    advice = "On-time likely"
                                elif pred_value < 0.6:
                                    risk_level = "Moderate"
                                    risk_color = "orange"
                                    emoji = "⚠️"
                                    advice = "Minor delay possible"
                                else:
                                    risk_level = "High"
                                    risk_color = "red"
                                    emoji = "🚨"
                                    advice = "Delay likely"
                                
                                # Create custom metric with color coding
                                st.markdown(f"""
                                <div style="border-left: 4px solid {risk_color}; padding-left: 10px;">
                                    <h4 style="margin: 0; color: {risk_color}; font-size: 34px;">{emoji} Delay Risk: {risk_level}</h4>
                                    <p style="margin: 5px 0; font-size: 20px;">Probability: {pred_value:.1%}</p>
                                    <p style="margin: 0; font-size: 16px; color: #666;">{advice}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif is_delayed and 'Delay' in flight:
                                st.metric(
                                    label="Current Delay",
                                    value=f"{int(flight['Delay'])} min",
                                    help="Actual delay time"
                                )
                            else:
                                st.metric(
                                    label="Status",
                                    value=flight.get('Status', 'N/A')
                                )
                        
                        st.divider()
                        
                        # Flight map section
                        if 'Dest' in flight and flight['Dest'] != 'N/A':
                            st.markdown("### 🌍 Flight Route")
                            
                            # Generate consistent coordinates for destination
                            seed = hash(str(flight['Dest'])) % (2**32)
                            
                            # Use real airport coordinates if available, otherwise generate realistic ones
                            airport_coordinates = {
                                # Existing
                                'JFK': (40.6413, -73.7781),
                                'LHR': (51.4700, -0.4543),
                                'CDG': (49.0097, 2.5479),
                                'DXB': (25.2532, 55.3657),
                                'HND': (35.5494, 139.7798),
                                'SYD': (-33.9399, 151.1753),

                                # Major US hubs
                                'ATL': (33.6407, -84.4277),
                                'DFW': (32.8998, -97.0403),
                                'ORD': (41.9742, -87.9073),
                                'DEN': (39.8561, -104.6737),
                                'LAX': (33.9416, -118.4085),
                                'SFO': (37.6213, -122.3790),
                                'SEA': (47.4502, -122.3088),
                                'MIA': (25.7959, -80.2871),
                                'LAS': (36.0840, -115.1537),

                                # New York area
                                'EWR': (40.6895, -74.1745),
                                'LGA': (40.7769, -73.8740),

                                # Texas
                                'IAH': (29.9902, -95.3368),
                                'AUS': (30.1975, -97.6664),
                                'SAT': (29.5337, -98.4698),

                                # California
                                'SAN': (32.7338, -117.1933),
                                'SJC': (37.3639, -121.9289),
                                'OAK': (37.7126, -122.2197),
                                'SMF': (38.6954, -121.5908),

                                # Midwest
                                'MSP': (44.8848, -93.2223),
                                'DTW': (42.2162, -83.3554),
                                'STL': (38.7500, -90.3700),
                                'CLE': (41.4117, -81.8498),

                                # Southeast
                                'CLT': (35.2144, -80.9473),
                                'BNA': (36.1245, -86.6782),
                                'TPA': (27.9755, -82.5332),
                                'MCO': (28.4312, -81.3081),

                                # Northeast
                                'BOS': (42.3656, -71.0096),
                                'PHL': (39.8744, -75.2424),
                                'IAD': (38.9531, -77.4565),
                                'DCA': (38.8512, -77.0402),

                                # Mountain / Southwest
                                'PHX': (33.4373, -112.0078),
                                'SLC': (40.7899, -111.9791),
                                'ABQ': (35.0402, -106.6092),

                                # Pacific / Hawaii
                                'HNL': (21.3187, -157.9225),
                            }
                            
                            # Try to get real coordinates, otherwise generate from seed
                            dest_code = str(flight['Dest']).strip().upper()
                            if dest_code in airport_coordinates:
                                dest_lat, dest_lon = airport_coordinates[dest_code]
                            else:
                                # Generate deterministic pseudo-random coordinates
                                # US airports generally between lat 25-49, lon -125 to -65
                                # International varies more
                                import random
                                random.seed(seed)
                                if dest_code[-1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                                    # Likely US airport
                                    dest_lat = random.uniform(25.0, 49.0)
                                    dest_lon = random.uniform(-125.0, -65.0)
                                else:
                                    # International
                                    dest_lat = random.uniform(-55.0, 70.0)
                                    dest_lon = random.uniform(-180.0, 180.0)
                            
                            LA_LAT, LA_LON = 33.9416, -118.4085
                            
                            # Create dataframes for pydeck
                            airports_df = pd.DataFrame([
                                {"name": "LAX (Los Angeles)", "lat": LA_LAT, "lon": LA_LON},
                                {"name": f"{dest_code} ({flight['Dest']})", "lat": dest_lat, "lon": dest_lon}
                            ])
                            
                            arc_df = pd.DataFrame([{
                                "from_lat": LA_LAT,
                                "from_lon": LA_LON,
                                "to_lat": dest_lat,
                                "to_lon": dest_lon
                            }])
                            
                            # Color based on delay status
                            if is_delayed:
                                arc_color = [255, 50, 50]  # Red for delayed
                            elif 'prediction' in flight and float(flight['prediction']) > 0.5:
                                arc_color = [255, 140, 0]  # Orange for at-risk
                            else:
                                arc_color = [30, 144, 255]  # Blue for normal
                            
                            # Airport layer
                            layer_airports = pdk.Layer(
                                "ScatterplotLayer",
                                data=airports_df,
                                get_position=["lon", "lat"],  # Note: [lon, lat] order!
                                get_radius=200000,
                                get_fill_color=[220, 20, 60],  # Crimson red
                                get_line_color=[0, 0, 0],
                                get_line_width=2,
                                pickable=True,
                                radius_min_pixels=6,
                                radius_max_pixels=10,
                            )
                            
                            # Arc/route layer
                            layer_flight = pdk.Layer(
                                "ArcLayer",
                                data=arc_df,
                                get_source_position=["from_lon", "from_lat"],  # [lon, lat] order!
                                get_target_position=["to_lon", "to_lat"],  # [lon, lat] order!
                                get_source_color=arc_color,
                                get_target_color=arc_color,
                                get_width=3,
                                get_tilt=15,
                                pickable=False,
                            )
                            
                            # Calculate center and zoom
                            center_lat = (LA_LAT + dest_lat) / 2
                            center_lon = (LA_LON + dest_lon) / 2
                            
                            # Calculate appropriate zoom based on distance
                            import math
                            lat_diff = abs(LA_LAT - dest_lat)
                            lon_diff = abs(LA_LON - dest_lon)
                            distance = math.sqrt(lat_diff**2 + lon_diff**2)
                            
                            if distance > 30:  # Long distance (international)
                                zoom = 1.8
                                pitch = 20
                            elif distance > 10:  # Medium distance (cross-country)
                                zoom = 2.5
                                pitch = 30
                            else:  # Short distance
                                zoom = 3.5
                                pitch = 40
                            
                            view_state = pdk.ViewState(
                                latitude=center_lat,
                                longitude=center_lon,
                                zoom=zoom,
                                pitch=pitch,
                                bearing=0
                            )
                            
                            # Create tooltip
                            tooltip = {
                                "html": "<b>{name}</b>",
                                "style": {
                                    "backgroundColor": "white",
                                    "color": "black",
                                    "fontFamily": "Arial",
                                    "fontSize": "12px",
                                    "padding": "5px"
                                }
                            }
                            
                            # Create deck
                            deck = pdk.Deck(
                                layers=[layer_airports, layer_flight],
                                initial_view_state=view_state,
                                tooltip=tooltip
                            )
                            
                            # Display the map
                            st.pydeck_chart(deck)
                            
                            # Map legend
                            st.markdown("---")
                            legend_col1, legend_col2, legend_col3, legend_col4 = st.columns(4)
                            
                            with legend_col1:
                                st.markdown("🔴 **Departure:** LAX")
                            with legend_col2:
                                st.markdown("🔴 **Arrival:** " + dest_code)
                            with legend_col3:
                                if is_delayed:
                                    st.markdown("🔴 **Route:** Currently Delayed")
                                elif 'prediction' in flight and float(flight['prediction']) > 0.5:
                                    st.markdown("🟠 **Route:** High Delay Risk")
                                else:
                                    st.markdown("🔵 **Route:** Normal Operation")
                    else:
                        st.error("❌ Flight not found")
                        st.markdown("""
                        **Suggestions:**
                        - Check the flight number format (e.g., AA123)
                        - Ensure flight operates from LAX
                        - Verify the flight hasn't been cancelled
                        """)
                else:
                    st.warning("⚠️ Flight data not available in the system")
                    
            except Exception as e:
                st.error(f"🚫 Error searching for flight: {str(e)}")
                st.info("Please try again with a different flight number")
        
        elif not flight_num:
            # Show instructions when no flight is entered
            with st.container():
                st.markdown("""
                ### How to Find Your Flight
                
                1. **Enter your flight number** (e.g., AA123, DL456)
                2. Click **"Search Flight"** to get detailed information
                3. View real-time status, delay predictions, and flight route
                
                **Features you'll see:**
                - ✅ Real-time flight status
                - 📊 Delay probability prediction
                - 🌍 Interactive flight route map
                - ⏰ Actual vs planned departure times
                """)
                
                # Quick statistics if available
                if 'Flight_Num' in flight_df.columns:
                    st.divider()
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    with stats_col1:
                        st.metric("Total Flights", len(flight_df))
                    with stats_col2:
                        if 'Delay' in flight_df.columns:
                            delayed = flight_df['Delay'].astype(float) > 0
                            st.metric("Delayed Flights", f"{delayed.sum():,}")
                    with stats_col3:
                        if 'prediction' in flight_df.columns:
                            high_risk = (flight_df['prediction'].astype(float) > 0.5).sum()
                            st.metric("At Risk", f"{high_risk:,}")
    
    # ---------------------- TAB 3: Metrics ----------------------
    # with tab3:
    #     st.subheader("📊 Flight Metrics")
        
    #     if not flight_df.empty and 'Status' in flight_df.columns:
    #         total_flights = len(flight_df)
    #         delayed = flight_df['Status'].str.contains("Delayed").sum()
    #         cancelled = (flight_df['Status'] == "Cancelled").sum()
    #         departed = (flight_df['Status'] == "Departured").sum()
    #         on_time = (flight_df['Status'] == "On-time").sum()
            
    #         if 'Delay' in flight_df.columns:
    #             max_delay = flight_df['Delay'].max()
    #             avg_delay = flight_df['Delay'].mean()
    #         else:
    #             max_delay = 0
    #             avg_delay = 0
            
    #         col1, col2, col3 = st.columns(3)
    #         col1.metric("Total Flights", total_flights)
    #         col2.metric("On Time", on_time)
    #         col3.metric("Departed", departed)
            
    #         col4, col5, col6 = st.columns(3)
    #         col4.metric("Delayed Flights", delayed)
    #         col5.metric("Cancelled Flights", cancelled)
    #         col6.metric("Max Delay (min)", int(max_delay))
            
    #         st.markdown(f"**Average Delay:** {avg_delay:.1f} min")
    #     else:
    #         st.info("Flight metrics not available in current data")
    
    # ---------------------- TAB 4: Environment ----------------------
    with tab3:
        st.subheader("🌤️ Weather & 🌍 Seismic Activity")
        
        # Weather Section
        st.markdown("### 🌡️ Current Weather at Airport")
        
        if weather_data:
            col1, col2, col3 = st.columns(3)
            
            # Temperature with change if available
            if 'temp' in weather_data:
                current_temp = weather_data['temp']
                delta_temp = None
                
                # Check if we have temperature change data
                if 'temp_change' in weather_data:
                    delta_temp = weather_data['temp_change']
                
                # Check if we have previous temperature in session state
                elif 'last_temperature' in st.session_state:
                    delta_temp = current_temp - st.session_state.last_temperature
                    st.session_state.last_temperature = current_temp
                else:
                    st.session_state.last_temperature = current_temp
                
                # Display with appropriate delta
                if delta_temp is not None and abs(delta_temp) > 0.01:
                    col1.metric("Temperature (°C)", f"{current_temp:.1f}", 
                            f"{delta_temp:+.1f}°C")
                else:
                    col1.metric("Temperature (°C)", f"{current_temp:.1f}")
            
            # Humidity (unchanged)
            if 'humidity' in weather_data:
                dew = weather_data.get('dew', 'N/A')
                delta_humidity = None
                
                # Optional: Show humidity change if available
                if 'humidity_change' in weather_data:
                    delta_humidity = f"{weather_data['humidity_change']:+.0f}%"
                
                if delta_humidity:
                    col2.metric("Humidity (%)", f"{abs(weather_data['humidity']):.0f}", 
                            delta_humidity)
                else:
                    col2.metric("Humidity (%)", f"{abs(weather_data['humidity']):.0f}")
            
            # Wind (unchanged)
            if 'windspeed' in weather_data:
                gust = weather_data.get('windgust', weather_data['windspeed'])
                delta_wind = None
                
                # Optional: Show wind change if available
                if 'windspeed_change' in weather_data:
                    delta_wind = f"{weather_data['windspeed_change']:+.1f} km/h"
                
                if delta_wind:
                    col3.metric("Wind (km/h)", f"{abs(weather_data['windspeed']):.1f}", 
                            delta_wind)
                else:
                    col3.metric("Wind (km/h)", f"{abs(weather_data['windspeed']):.1f}")
            
            # Second row (optional changes could be added similarly)
            col4, col5, col6 = st.columns(3)
            if 'precip' in weather_data:
                col4.metric("Precipitation (mm)", f"{abs(weather_data['precip']):.1f}")
            
            if 'cloudcover' in weather_data:
                col5.metric("Cloud Cover (%)", f"{abs(weather_data['cloudcover']):.0f}")
            
            if 'visibility' in weather_data:
                col6.metric("Visibility (km)", f"{abs(weather_data['visibility']):.1f}")
        else:
            st.info("No weather data available")
        
        # Seismic Activity Section
        st.markdown("### 🌍 Seismic Activity")
        
        if not seismic_df.empty:

            seismic_df = seismic_df.copy()
            seismic_df['datetime'] = pd.to_datetime(seismic_df['datetime'], errors='coerce')

            cutoff = df['timestamp'].max()
            recent_quakes = seismic_df[seismic_df['datetime'] <= cutoff]

            if not recent_quakes.empty:

                latest_ts = df['timestamp'].max()

                def recency_color(event_time):
                    if pd.isna(event_time):
                        return [200, 200, 200, 150]

                    delta_hours = (latest_ts - event_time).total_seconds() / 3600

                    if delta_hours < 1:
                        return [255, 0, 0, 200]
                    elif delta_hours < 3:
                        return [255, 165, 0, 180]
                    else:
                        return [200, 200, 200, 150]

                recent_quakes['color'] = recent_quakes['datetime'].apply(recency_color)

                # ✅ PRECOMPUTE RADIUS (CRITICAL)
                recent_quakes['radius'] = (recent_quakes['mag'] * 500).clip(lower=15000)

                st.markdown("#### Recent Seismic Events (Last 24 hours)")

                seismic_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=recent_quakes,
                    get_position='[longitude, latitude]',
                    get_radius='radius',        # ✅ column name ONLY
                    get_fill_color='color',     # ✅ column name ONLY
                    pickable=True,
                    auto_highlight=True
                )

                center_lat = recent_quakes.iloc[0]['latitude']
                center_lon = recent_quakes.iloc[0]['longitude']

                seismic_view = pdk.ViewState(
                    latitude=center_lat,
                    longitude=center_lon,
                    zoom=6
                )

                st.pydeck_chart(
                    pdk.Deck(
                        layers=[seismic_layer],
                        initial_view_state=seismic_view,
                        tooltip={
                            "html": (
                                "<b>Place:</b> {place}<br/>"
                                "<b>Magnitude:</b> {mag}<br/>"
                                "<b>Depth:</b> {depth} km<br/>"
                                "<b>Time:</b> {datetime}"
                            ),
                            "style": {"color": "white"}
                        }
                    )
                )

                display_cols = ['datetime', 'place', 'mag', 'depth', 'status']
                display_df = recent_quakes[display_cols].sort_values(
                    by='datetime', ascending=False
                )

                # st.dataframe(display_df, use_container_width=True)

            else:
                st.info("No seismic activity in the last 24 hours.")

        else:
            st.info("No seismic data available.")

    
    # # ---------------------- TAB 5: Raw Data ----------------------
    # with tab5:
    #     st.subheader("📋 Raw Cassandra Data")
    #     st.dataframe(seismic_df, use_container_width=True, hide_index=True)
        
    #     # Show column information
    #     st.subheader("📊 Column Information")
    #     col_info = pd.DataFrame({
    #         'Column': df.columns,
    #         'Type': [str(df[col].dtype) for col in df.columns],
    #         'Non-Null Count': [df[col].count() for col in df.columns],
    #         'Unique Values': [df[col].nunique() for col in df.columns]
    #     })
    #     st.dataframe(col_info, use_container_width=True)
        
        
        
### historic data



# --------------------------------------------------
# TABS FOR DIFFERENT VIEWS
# --------------------------------------------------


    with tab4:
        if not monthly_df.empty:
            st.header("Monthly Aggregated Trends")
            
            # Date range selector for monthly data
            min_date = monthly_df['month_start'].min()
            max_date = monthly_df['month_start'].max()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="monthly_start"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="monthly_end"
                )
            
            # Filter data based on date selection
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)
            filtered_monthly = monthly_df[
                (monthly_df['month_start'] >= start_date) & 
                (monthly_df['month_start'] <= end_date)
            ]
            
            # Create a readable month-year label
            filtered_monthly['month_year'] = filtered_monthly['month_start'].dt.strftime('%b %Y')
            
            # Category selector
            category = st.selectbox(
                "Select data category",
                ["Seismic", "Flights", "Weather"],
                key="monthly_category"
            )
            
            if category == "Seismic":
                # Seismic metrics
                st.subheader("Seismic Activity")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.line(
                        filtered_monthly,
                        x='month_year',
                        y='seismic_total_events',
                        title='Total Seismic Events per Month',
                        markers=True
                    )
                    fig.update_layout(xaxis_title="Month", yaxis_title="Total Events")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    fig2 = px.line(
                        filtered_monthly,
                        x='month_year',
                        y='seismic_avg_magnitude',
                        title='Average Magnitude per Month',
                        markers=True
                    )
                    fig2.update_layout(xaxis_title="Month", yaxis_title="Average Magnitude")
                    st.plotly_chart(fig2, use_container_width=True)
                
                with col2:
                    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
                    fig3.add_trace(
                        go.Scatter(
                            x=filtered_monthly['month_year'],
                            y=filtered_monthly['seismic_max_magnitude'],
                            name='Max Magnitude',
                            mode='lines+markers',
                            line=dict(color='red')
                        ),
                        secondary_y=False
                    )
                    fig3.add_trace(
                        go.Scatter(
                            x=filtered_monthly['month_year'],
                            y=filtered_monthly['seismic_avg_depth'],
                            name='Average Depth',
                            mode='lines+markers',
                            line=dict(color='blue')
                        ),
                        secondary_y=True
                    )
                    fig3.update_layout(
                        title_text='Max Magnitude and Average Depth',
                        xaxis_title="Month"
                    )
                    fig3.update_yaxes(title_text="Max Magnitude", secondary_y=False)
                    fig3.update_yaxes(title_text="Average Depth (km)", secondary_y=True)
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Significant events
                    fig4 = px.bar(
                        filtered_monthly,
                        x='month_year',
                        y='seismic_significant_events',
                        title='Significant Seismic Events',
                        color='seismic_significant_events',
                        color_continuous_scale='reds'
                    )
                    fig4.update_layout(xaxis_title="Month", yaxis_title="Significant Events")
                    st.plotly_chart(fig4, use_container_width=True)
            
            elif category == "Flights":
                # Flight metrics
                st.subheader("Flight Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.line(
                        filtered_monthly,
                        x='month_year',
                        y='flights_total',
                        title='Total Flights per Month',
                        markers=True
                    )
                    fig.update_layout(xaxis_title="Month", yaxis_title="Total Flights")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    fig2 = px.line(
                        filtered_monthly,
                        x='month_year',
                        y='flights_avg_delay',
                        title='Average Flight Delay (minutes)',
                        markers=True,
                        line_shape='spline'
                    )
                    fig2.update_layout(xaxis_title="Month", yaxis_title="Average Delay (min)")
                    st.plotly_chart(fig2, use_container_width=True)
                
                with col2:
                    # Cancellation and on-time rates
                    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
                    fig3.add_trace(
                        go.Scatter(
                            x=filtered_monthly['month_year'],
                            y=filtered_monthly['flights_cancellation_rate'] * 100,
                            name='Cancellation Rate %',
                            mode='lines+markers',
                            line=dict(color='red')
                        ),
                        secondary_y=False
                    )
                    fig3.add_trace(
                        go.Scatter(
                            x=filtered_monthly['month_year'],
                            y=filtered_monthly['flights_on_time_rate'] * 100,
                            name='On-Time Rate %',
                            mode='lines+markers',
                            line=dict(color='green')
                        ),
                        secondary_y=True
                    )
                    fig3.update_layout(
                        title_text='Cancellation and On-Time Rates',
                        xaxis_title="Month"
                    )
                    fig3.update_yaxes(title_text="Cancellation Rate %", secondary_y=False)
                    fig3.update_yaxes(title_text="On-Time Rate %", secondary_y=True)
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Top carrier share
                    fig4 = px.bar(
                        filtered_monthly,
                        x='month_year',
                        y='flights_top_carrier_share',
                        title='Top Carrier Market Share',
                        color='flights_top_carrier',
                        text='flights_top_carrier'
                    )
                    fig4.update_layout(xaxis_title="Month", yaxis_title="Market Share")
                    fig4.update_traces(textposition='outside')
                    st.plotly_chart(fig4, use_container_width=True)
            
            elif category == "Weather":
                # Weather metrics
                st.subheader("Weather Conditions")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Temperature trends
                    fig = make_subplots(specs=[[{"secondary_y": False}]])
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_monthly['month_year'],
                            y=filtered_monthly['weather_avg_temp'],
                            name='Avg Temp',
                            mode='lines+markers',
                            line=dict(color='orange')
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_monthly['month_year'],
                            y=filtered_monthly['weather_max_temp'],
                            name='Max Temp',
                            mode='lines+markers',
                            line=dict(color='red', dash='dash')
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_monthly['month_year'],
                            y=filtered_monthly['weather_min_temp'],
                            name='Min Temp',
                            mode='lines+markers',
                            line=dict(color='blue', dash='dash')
                        )
                    )
                    fig.update_layout(
                        title_text='Temperature Trends (°C)',
                        xaxis_title="Month",
                        yaxis_title="Temperature (°C)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Precipitation
                    fig2 = px.bar(
                        filtered_monthly,
                        x='month_year',
                        y='weather_total_precipitation',
                        title='Total Precipitation',
                        color='weather_rainy_days',
                        color_continuous_scale='blues'
                    )
                    fig2.update_layout(xaxis_title="Month", yaxis_title="Precipitation (mm)")
                    st.plotly_chart(fig2, use_container_width=True)
                
                with col2:
                    # Humidity and windspeed
                    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
                    fig3.add_trace(
                        go.Scatter(
                            x=filtered_monthly['month_year'],
                            y=filtered_monthly['weather_avg_humidity'],
                            name='Avg Humidity %',
                            mode='lines+markers',
                            line=dict(color='green')
                        ),
                        secondary_y=False
                    )
                    fig3.add_trace(
                        go.Scatter(
                            x=filtered_monthly['month_year'],
                            y=filtered_monthly['weather_avg_windspeed'],
                            name='Avg Windspeed',
                            mode='lines+markers',
                            line=dict(color='purple')
                        ),
                        secondary_y=True
                    )
                    fig3.update_layout(
                        title_text='Humidity and Windspeed',
                        xaxis_title="Month"
                    )
                    fig3.update_yaxes(title_text="Humidity %", secondary_y=False)
                    fig3.update_yaxes(title_text="Windspeed", secondary_y=True)
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Pressure
                    fig4 = px.line(
                        filtered_monthly,
                        x='month_year',
                        y='weather_avg_pressure',
                        title='Average Pressure',
                        markers=True,
                        line_shape='spline'
                    )
                    fig4.update_layout(xaxis_title="Month", yaxis_title="Pressure (hPa)")
                    st.plotly_chart(fig4, use_container_width=True)
            
            # Show data table
            with st.expander("View Monthly Data"):
                st.dataframe(
                    filtered_monthly.drop(columns=['month_year']),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning("No monthly data available")

    with tab5:
        if not weekly_df.empty:
            st.header("Weekly Aggregated Trends")
            
            # Date range selector for weekly data
            min_date = weekly_df['week_start'].min()
            max_date = weekly_df['week_start'].max()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="weekly_start"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="weekly_end"
                )
            
            # Filter data
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)
            filtered_weekly = weekly_df[
                (weekly_df['week_start'] >= start_date) & 
                (weekly_df['week_start'] <= end_date)
            ]
            
            # Create readable week label
            filtered_weekly['week_label'] = filtered_weekly['week_start'].dt.strftime('%Y-W%W')
            
            # Category selector
            category = st.selectbox(
                "Select data category",
                ["Seismic", "Flights", "Weather"],
                key="weekly_category"
            )
            
            if category == "Seismic":
                # Weekly seismic visualizations
                st.subheader("Weekly Seismic Activity")
                
                fig = px.line(
                    filtered_weekly,
                    x='week_label',
                    y='seismic_total_events',
                    title='Seismic Events per Week',
                    markers=True
                )
                fig.update_layout(xaxis_title="Week", yaxis_title="Total Events")
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig2 = px.scatter(
                        filtered_weekly,
                        x='seismic_avg_magnitude',
                        y='seismic_avg_depth',
                        size='seismic_total_events',
                        color='seismic_max_magnitude',
                        title='Magnitude vs Depth (Size = Total Events)',
                        hover_data=['week_start']
                    )
                    fig2.update_layout(xaxis_title="Average Magnitude", yaxis_title="Average Depth")
                    st.plotly_chart(fig2, use_container_width=True)
                
                with col2:
                    fig3 = px.bar(
                        filtered_weekly,
                        x='week_label',
                        y=['seismic_weekday_events', 'seismic_weekend_events'],
                        title='Weekday vs Weekend Events',
                        barmode='group'
                    )
                    fig3.update_layout(xaxis_title="Week", yaxis_title="Events Count")
                    st.plotly_chart(fig3, use_container_width=True)
            
            elif category == "Flights":
                # Weekly flight visualizations
                st.subheader("Weekly Flight Statistics")
                
                fig = px.line(
                    filtered_weekly,
                    x='week_label',
                    y='flights_total',
                    title='Total Flights per Week',
                    markers=True
                )
                fig.update_layout(xaxis_title="Week", yaxis_title="Total Flights")
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig2 = px.scatter(
                        filtered_weekly,
                        x='flights_avg_delay',
                        y='flights_on_time_rate',
                        size='flights_total',
                        color='flights_cancellation_rate',
                        title='Delay vs On-Time Rate',
                        hover_data=['week_start']
                    )
                    fig2.update_layout(xaxis_title="Average Delay", yaxis_title="On-Time Rate")
                    st.plotly_chart(fig2, use_container_width=True)
                
                with col2:
                    fig3 = px.line(
                        filtered_weekly,
                        x='week_label',
                        y=['flights_avg_delay', 'flights_median_delay'],
                        title='Average vs Median Delay',
                        markers=True
                    )
                    fig3.update_layout(xaxis_title="Week", yaxis_title="Delay (minutes)")
                    st.plotly_chart(fig3, use_container_width=True)
            
            elif category == "Weather":
                # Weekly weather visualizations
                st.subheader("Weekly Weather Conditions")
                
                fig = px.line(
                    filtered_weekly,
                    x='week_label',
                    y='weather_avg_temp',
                    title='Average Temperature per Week',
                    markers=True
                )
                fig.update_layout(xaxis_title="Week", yaxis_title="Temperature (°C)")
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig2 = px.scatter(
                        filtered_weekly,
                        x='weather_avg_humidity',
                        y='weather_total_precipitation',
                        size='weather_rainy_days',
                        color='weather_avg_temp',
                        title='Humidity vs Precipitation',
                        hover_data=['week_start']
                    )
                    fig2.update_layout(xaxis_title="Average Humidity", yaxis_title="Total Precipitation")
                    st.plotly_chart(fig2, use_container_width=True)
                
                with col2:
                    fig3 = px.line(
                        filtered_weekly,
                        x='week_label',
                        y=['weather_avg_windspeed', 'weather_max_windspeed'],
                        title='Average vs Max Windspeed',
                        markers=True
                    )
                    fig3.update_layout(xaxis_title="Week", yaxis_title="Windspeed")
                    st.plotly_chart(fig3, use_container_width=True)
            
            # Show data table
            with st.expander("View Weekly Data"):
                st.dataframe(
                    filtered_weekly.drop(columns=['week_label']),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning("No weekly data available")

    with tab6:
        st.header("Relationships Analysis")
        
        # Select data period
        analysis_period = st.radio(
            "Select data period for analysis",
            ["Monthly Data", "Weekly Data"],
            horizontal=True
        )
        
        if analysis_period == "Monthly Data" and not monthly_df.empty:
            df = monthly_df.copy()
            time_col = 'month_start'
            time_label = 'month_year'
            df[time_label] = df[time_col].dt.strftime('%b %Y')
        elif analysis_period == "Weekly Data" and not weekly_df.empty:
            df = weekly_df.copy()
            time_col = 'week_start'
            time_label = 'week_label'
            df[time_label] = df[time_col].dt.strftime('%Y-W%W')
        else:
            st.warning(f"No {analysis_period.lower()} available for analysis")
            st.stop()
        
        # Relationship type selector
        relationship_type = st.selectbox(
            "Select relationship to explore",
            [
                "Seismic vs Flights Comparison",
                "Weather Impact on Flights", 
                "Time Series Trends",
                "Cross-Category Correlation"
            ]
        )
        
        if relationship_type == "Seismic vs Flights Comparison":
            st.subheader("Seismic Events vs Flight Volume Comparison")
            
            # Create a 2x2 grid of visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Normalized comparison bar chart
                if 'seismic_total_events' in df.columns and 'flights_total' in df.columns:
                    # Normalize data for comparison
                    seismic_norm = (df['seismic_total_events'] - df['seismic_total_events'].min()) / \
                                (df['seismic_total_events'].max() - df['seismic_total_events'].min() + 1e-10)
                    flights_norm = (df['flights_total'] - df['flights_total'].min()) / \
                                (df['flights_total'].max() - df['flights_total'].min() + 1e-10)
                    
                    # Create grouped bar chart
                    fig = go.Figure()
                    
                    # Add seismic bars
                    fig.add_trace(go.Bar(
                        name='Seismic Events (Normalized)',
                        x=df[time_label],
                        y=seismic_norm,
                        marker_color='red',
                        opacity=0.7
                    ))
                    
                    # Add flight bars
                    fig.add_trace(go.Bar(
                        name='Flights (Normalized)',
                        x=df[time_label],
                        y=flights_norm,
                        marker_color='blue',
                        opacity=0.7
                    ))
                    
                    fig.update_layout(
                        title='Normalized Comparison: Seismic Events vs Flight Volume',
                        xaxis_title=analysis_period.replace(" Data", ""),
                        yaxis_title='Normalized Value',
                        barmode='group',
                        xaxis_tickangle=45
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Scatter plot of seismic vs flights
                if 'seismic_total_events' in df.columns and 'flights_total' in df.columns:
                    fig = px.scatter(
                        df,
                        x='seismic_total_events',
                        y='flights_total',
                        title='Seismic Events vs Flight Volume',
                        trendline='ols',
                        hover_data=[time_col],
                        color=df[time_col].dt.month if analysis_period == "Monthly Data" else None,
                        size='seismic_total_events'
                    )
                    fig.update_layout(
                        xaxis_title='Seismic Events',
                        yaxis_title='Total Flights'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Additional visualizations in second row
            col3, col4 = st.columns(2)
            
            with col3:
                # Seismic magnitude vs flight delays
                if all(col in df.columns for col in ['seismic_avg_magnitude', 'flights_avg_delay']):
                    fig = px.scatter(
                        df,
                        x='seismic_avg_magnitude',
                        y='flights_avg_delay',
                        title='Seismic Magnitude vs Flight Delays',
                        trendline='ols',
                        hover_data=[time_col, 'seismic_total_events', 'flights_total'],
                        color='seismic_total_events',
                        size='flights_total'
                    )
                    fig.update_layout(
                        xaxis_title='Average Seismic Magnitude',
                        yaxis_title='Average Flight Delay (minutes)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                # Seismic depth vs flight cancellation rate
                if all(col in df.columns for col in ['seismic_avg_depth', 'flights_cancellation_rate']):
                    fig = px.scatter(
                        df,
                        x='seismic_avg_depth',
                        y=df['flights_cancellation_rate'] * 100,
                        title='Seismic Depth vs Flight Cancellation Rate',
                        trendline='ols',
                        hover_data=[time_col, 'seismic_total_events'],
                        color='seismic_total_events',
                        size='seismic_total_events'
                    )
                    fig.update_layout(
                        xaxis_title='Average Seismic Depth (km)',
                        yaxis_title='Flight Cancellation Rate (%)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        elif relationship_type == "Weather Impact on Flights":
            st.subheader("Weather Impact Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Precipitation vs Flight Cancellations
                if all(col in df.columns for col in ['weather_total_precipitation', 'flights_cancellation_rate']):
                    fig = px.scatter(
                        df,
                        x='weather_total_precipitation',
                        y=df['flights_cancellation_rate'] * 100,
                        title='Precipitation vs Flight Cancellations',
                        trendline='ols',
                        hover_data=[time_col],
                        text=df[time_label],
                        size='weather_rainy_days' if 'weather_rainy_days' in df.columns else None,
                        color='weather_avg_temp' if 'weather_avg_temp' in df.columns else None
                    )
                    fig.update_traces(textposition='top center')
                    fig.update_layout(
                        xaxis_title='Total Precipitation',
                        yaxis_title='Cancellation Rate (%)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Temperature vs Flight Delays
                if all(col in df.columns for col in ['weather_avg_temp', 'flights_avg_delay']):
                    fig = px.scatter(
                        df,
                        x='weather_avg_temp',
                        y='flights_avg_delay',
                        title='Temperature vs Flight Delays',
                        trendline='ols',
                        hover_data=[time_col],
                        size='weather_max_windspeed' if 'weather_max_windspeed' in df.columns else None,
                        color='weather_avg_humidity' if 'weather_avg_humidity' in df.columns else None
                    )
                    fig.update_layout(
                        xaxis_title='Average Temperature (°C)',
                        yaxis_title='Average Flight Delay (minutes)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Windspeed vs Flight Cancellations
                if all(col in df.columns for col in ['weather_avg_windspeed', 'flights_cancellation_rate']):
                    fig = px.scatter(
                        df,
                        x='weather_avg_windspeed',
                        y=df['flights_cancellation_rate'] * 100,
                        title='Windspeed vs Flight Cancellations',
                        trendline='ols',
                        hover_data=[time_col],
                        size='weather_max_windspeed' if 'weather_max_windspeed' in df.columns else None
                    )
                    fig.update_layout(
                        xaxis_title='Average Windspeed',
                        yaxis_title='Cancellation Rate (%)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                # Humidity vs On-Time Rate
                if all(col in df.columns for col in ['weather_avg_humidity', 'flights_on_time_rate']):
                    fig = px.scatter(
                        df,
                        x='weather_avg_humidity',
                        y=df['flights_on_time_rate'] * 100,
                        title='Humidity vs On-Time Rate',
                        trendline='ols',
                        hover_data=[time_col],
                        color='weather_avg_temp' if 'weather_avg_temp' in df.columns else None
                    )
                    fig.update_layout(
                        xaxis_title='Average Humidity (%)',
                        yaxis_title='On-Time Rate (%)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        elif relationship_type == "Time Series Trends":
            st.subheader("Time Series Analysis of Key Metrics")
            
            # Create 3x2 grid of time series plots
            # Row 1: Seismic Events and Flight Delays
            col1, col2 = st.columns(2)
            
            with col1:
                # Seismic Events Over Time
                if 'seismic_total_events' in df.columns:
                    fig = px.line(
                        df,
                        x=time_label,
                        y='seismic_total_events',
                        title='Seismic Events Over Time',
                        markers=True,
                        line_shape='spline'
                    )
                    fig.update_layout(
                        xaxis_title=analysis_period.replace(" Data", ""),
                        yaxis_title='Number of Events',
                        xaxis_tickangle=45
                    )
                    # Add trend line
                    if len(df) > 2:
                        z = np.polyfit(range(len(df)), df['seismic_total_events'].fillna(0), 1)
                        p = np.poly1d(z)
                        fig.add_trace(go.Scatter(
                            x=df[time_label],
                            y=p(range(len(df))),
                            mode='lines',
                            name=f'Trend: {z[0]:.3f}x + {z[1]:.2f}',
                            line=dict(color='red', dash='dash')
                        ))
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Flight Delays Over Time
                if 'flights_avg_delay' in df.columns:
                    # Create bar chart with color coding
                    colors = []
                    for delay in df['flights_avg_delay']:
                        if pd.isna(delay):
                            colors.append('gray')
                        elif delay > 30:
                            colors.append('red')
                        elif delay > 15:
                            colors.append('orange')
                        else:
                            colors.append('green')
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=df[time_label],
                            y=df['flights_avg_delay'],
                            marker_color=colors,
                            text=df['flights_avg_delay'].round(1),
                            textposition='outside'
                        )
                    ])
                    
                    fig.update_layout(
                        title='Average Flight Delays Over Time',
                        xaxis_title=analysis_period.replace(" Data", ""),
                        yaxis_title='Average Delay (minutes)',
                        xaxis_tickangle=45,
                        showlegend=False
                    )
                    
                    # Add horizontal line for on-time reference
                    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Row 2: Temperature and Cancellation Rate
            col3, col4 = st.columns(2)
            
            with col3:
                # Temperature Trends
                if 'weather_avg_temp' in df.columns:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=df[time_label],
                        y=df['weather_avg_temp'],
                        mode='lines+markers',
                        name='Avg Temp',
                        line=dict(color='orange', width=2)
                    ))
                    
                    if 'weather_min_temp' in df.columns and 'weather_max_temp' in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df[time_label],
                            y=df['weather_min_temp'],
                            mode='lines',
                            name='Min Temp',
                            line=dict(color='blue', width=1, dash='dash'),
                            opacity=0.5
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=df[time_label],
                            y=df['weather_max_temp'],
                            mode='lines',
                            name='Max Temp',
                            line=dict(color='red', width=1, dash='dash'),
                            opacity=0.5
                        ))
                    
                    fig.update_layout(
                        title='Temperature Trends',
                        xaxis_title=analysis_period.replace(" Data", ""),
                        yaxis_title='Temperature (°C)',
                        xaxis_tickangle=45
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                # Cancellation Rate Over Time
                if 'flights_cancellation_rate' in df.columns:
                    cancellation_rate = df['flights_cancellation_rate'] * 100
                    
                    fig = px.line(
                        df,
                        x=time_label,
                        y=cancellation_rate,
                        title='Flight Cancellation Rate Over Time',
                        markers=True,
                        line_shape='spline'
                    )
                    
                    # Add average line
                    avg_rate = cancellation_rate.mean()
                    fig.add_hline(
                        y=avg_rate,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text=f'Average: {avg_rate:.1f}%',
                        annotation_position="bottom right"
                    )
                    
                    fig.update_layout(
                        xaxis_title=analysis_period.replace(" Data", ""),
                        yaxis_title='Cancellation Rate (%)',
                        xaxis_tickangle=45
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Row 3: Recent weeks analysis
            st.subheader(f"Recent {analysis_period.replace(' Data', 's')} Analysis")
            
            # Show last 12 periods
            recent_df = df.tail(12).copy()
            
            col5, col6 = st.columns(2)
            
            with col5:
                # Recent seismic events
                if 'seismic_total_events' in recent_df.columns:
                    fig = px.bar(
                        recent_df,
                        x=time_label,
                        y='seismic_total_events',
                        title=f'Recent {analysis_period.replace(" Data", "")} Seismic Events',
                        color='seismic_total_events',
                        color_continuous_scale='reds'
                    )
                    fig.update_layout(
                        xaxis_title=analysis_period.replace(" Data", ""),
                        yaxis_title='Number of Events',
                        xaxis_tickangle=45
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col6:
                # Flight volume vs delays
                if all(col in recent_df.columns for col in ['flights_total', 'flights_avg_delay']):
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig.add_trace(
                        go.Bar(
                            x=recent_df[time_label],
                            y=recent_df['flights_total'],
                            name='Flight Volume',
                            marker_color='blue',
                            opacity=0.6
                        ),
                        secondary_y=False
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=recent_df[time_label],
                            y=recent_df['flights_avg_delay'],
                            name='Avg Delay',
                            mode='lines+markers',
                            line=dict(color='red', width=2)
                        ),
                        secondary_y=True
                    )
                    
                    fig.update_layout(
                        title=f'Recent {analysis_period.replace(" Data", "")}: Flight Volume vs Delays',
                        xaxis_title=analysis_period.replace(" Data", ""),
                        xaxis_tickangle=45
                    )
                    
                    fig.update_yaxes(
                        title_text="Number of Flights",
                        secondary_y=False
                    )
                    fig.update_yaxes(
                        title_text="Average Delay (minutes)",
                        secondary_y=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        elif relationship_type == "Cross-Category Correlation":
            st.subheader("Cross-Category Correlation Analysis")
            
            # Select variables for correlation
            col1, col2, col3 = st.columns(3)
            
            with col1:
                seismic_var = st.selectbox(
                    "Seismic Variable",
                    [col for col in df.columns if col.startswith('seismic_')],
                    key="corr_seismic"
                )
            
            with col2:
                flight_var = st.selectbox(
                    "Flight Variable",
                    [col for col in df.columns if col.startswith('flights_')],
                    key="corr_flight"
                )
            
            with col3:
                weather_var = st.selectbox(
                    "Weather Variable",
                    [col for col in df.columns if col.startswith('weather_')],
                    key="corr_weather"
                )
            
            # Calculate correlations
            corr_matrix = df[[seismic_var, flight_var, weather_var]].corr()
            
            # Display correlation matrix
            st.write("### Correlation Matrix")
            st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1).format("{:.3f}"))
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto='.3f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1,
                title=f"Correlation Heatmap: {analysis_period}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            
            # Create scatter matrix
            st.write("### Scatter Matrix")
            fig = px.scatter_matrix(
                df,
                dimensions=[seismic_var, flight_var, weather_var],
                color=df[time_col].dt.month if analysis_period == "Monthly Data" else None,
                hover_data=[time_col],
                title=f"Scatter Matrix: {analysis_period}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # EXAMPLE OF POSSIBLE INSIGHTS SECTION
        # with st.expander("Correlation Insights"):
        #     st.write("""
        #     **How to interpret correlations:**
        #     - **Strong positive correlation (0.7 to 1.0):** Variables increase together
        #     - **Moderate positive correlation (0.3 to 0.7):** Some tendency to increase together
        #     - **Weak correlation (-0.3 to 0.3):** Little to no relationship
        #     - **Moderate negative correlation (-0.7 to -0.3):** Some tendency to move oppositely
        #     - **Strong negative correlation (-1.0 to -0.7):** Variables decrease together
            
        #     **Note:** Correlation does not imply causation. Always consider external factors.
        #     """)