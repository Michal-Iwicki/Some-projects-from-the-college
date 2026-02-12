import pandas as pd
import numpy as np


def random_lat_lon(seed_value):
    rng = np.random.default_rng(seed=seed_value)
    lat = rng.uniform(-60, 70)
    lon = rng.uniform(-180, 180)
    return lat, lon

def format_ts(ts):
    if pd.isna(ts):
        return "--:--"
    return ts.strftime('%H:%M')

def get_status(row):
    # Check if columns exist before accessing
    if 'Cancelled' in row and row.get('Cancelled', 0) == 1:
        return "Cancelled"
    elif 'Actual_Dep_Timestamp' in row and pd.notna(row.get('Actual_Dep_Timestamp')) and row['Actual_Dep_Timestamp'] <= pd.Timestamp.now():
        return "Departured"
    elif 'Delay' in row and row['Delay'] > 0:
        return f"Delayed ({int(row['Delay'])} min)"
    else:
        return "On-time"

def status_style(status, blink=False):
    if "Delayed" in status:
        color = "#FF6961"
    elif status == "Cancelled":
        color = "#D3D3D3"
    elif status == "On-time":
        color = "#77DD77"
    elif status == "Departed":
        color = "#FFD700"
    else:
        color = "#FFFFFF"
    return {"background-color": color, "padding": "5px", "border-radius": "3px", "text-align": "center"}

def prepare_flight_data(df):
    """Prepare flight data for dashboard display"""
    if "timestamp" in df.columns:
        latest_ts = df["timestamp"].max()
        df = df[df["timestamp"] == latest_ts]
    # Convert column names if needed (Cassandra to expected format)
    column_mapping = {
        'flight_number': 'Flight_Num',
        'flight_num': 'Flight_Num',
        'airline': 'Airline Name',
        'airline_name': 'Airline Name',
        'destination': 'Dest',
        'dest': 'Dest',
        'planned_departure': 'Planned_Dep_Timestamp',
        'planned_dep_timestamp': 'Planned_Dep_Timestamp',
        'actual_departure': 'Actual_Dep_Timestamp',
        'actual_dep_timestamp': 'Actual_Dep_Timestamp',
        'estimated_delay': 'Estimated_Delay',
        'delay': 'Estimated_Delay',
        'cancelled': 'Cancelled'
    }
    
    # Rename columns if they exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Convert timestamps if they exist
    timestamp_cols = ['Planned_Dep_Timestamp', 'Actual_Dep_Timestamp']
    for col in timestamp_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
    
    # Calculate delay if not present
    if 'Estimated_Delay' in df.columns and 'Delay' not in df.columns:
        df['Delay'] = df['Estimated_Delay'].fillna(0)
    elif 'Delay' not in df.columns:
        df['Delay'] = 0
    
    # Determine status
    if 'Status' not in df.columns:
        df['Status'] = df.apply(get_status, axis=1)
    
    # Sort by departure time if available
    if 'Planned_Dep_Timestamp' in df.columns:
        df = df.sort_values(by='Planned_Dep_Timestamp')
    
    return df

def prepare_departed_flights(df, n_timestamps=10):
    """
    Get flights that were present in previous timestamps
    but are missing in the latest timestamp (i.e. departed).
    """
    if "timestamp" not in df.columns:
        return pd.DataFrame()

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Get last N timestamps
    timestamps = sorted(df["timestamp"].dropna().unique())
    if len(timestamps) < 2:
        return pd.DataFrame()

    last_ts = timestamps[-1]
    prev_ts = timestamps[-n_timestamps:-1]

    # Flights in latest snapshot
    latest_df = df[df["timestamp"] == last_ts]
    latest_flights = set(latest_df.get("flight_number", latest_df.get("flight_num", [])))

    # Flights in previous snapshots
    prev_df = df[df["timestamp"].isin(prev_ts)]

    flight_col = (
        "flight_number"
        if "flight_number" in prev_df.columns
        else "flight_num"
        if "flight_num" in prev_df.columns
        else None
    )

    if flight_col is None:
        return pd.DataFrame()

    # Keep only flights NOT present in latest snapshot
    departed_df = prev_df[~prev_df[flight_col].isin(latest_flights)]

    # Keep the most recent occurrence per flight
    departed_df = (
        departed_df.sort_values("timestamp")
        .groupby(flight_col, as_index=False)
        .last()
    )

    # Reuse your existing preparation logic
    departed_df = prepare_flight_data(departed_df)

    # Force status
    departed_df["Status"] = "Departed"

    # Optional: sort most recent departures first
    if "Actual_Dep_Timestamp" in departed_df.columns:
        departed_df = departed_df.sort_values(
            by="Actual_Dep_Timestamp", ascending=False
        )

    return departed_df

def extract_weather_data(df):
    """Extract weather data from the combined dataset"""
    weather_cols = ['temp', 'feelslike', 'humidity', 'dew', 'windspeed', 
                   'windgust', 'precip', 'cloudcover', 'visibility', 'datetime']
    
    df_sorted = df.sort_values("timestamp")
    
    weather_data = {}
    for col in weather_cols:
        if col in df_sorted.columns:
            # Take the latest non-null value
            non_null = df_sorted[col].dropna()
            if not non_null.empty:
                weather_data[col] = non_null.iloc[-1]
    
    return weather_data

def extract_weather_data_with_changes(df):
    """
    Extract weather data for the latest timestamp and calculate changes
    relative to the previous timestamp.
    """
    weather_cols = [
        'temp', 'feelslike', 'humidity', 'dew', 'windspeed', 
        'windgust', 'precip', 'cloudcover', 'visibility'
    ]

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Sort ascending by timestamp
    df_sorted = df.sort_values('timestamp')
    
    # Get unique timestamps (most recent two)
    unique_ts = df_sorted['timestamp'].dropna().unique()
    if len(unique_ts) == 0:
        return {}
    
    latest_ts = unique_ts[-1]
    prev_ts = unique_ts[-2] if len(unique_ts) > 1 else None

    latest_row = df_sorted[df_sorted['timestamp'] == latest_ts].iloc[-1]
    prev_row = df_sorted[df_sorted['timestamp'] == prev_ts].iloc[-1] if prev_ts is not None else None

    weather_data = {}

    for col in weather_cols:
        if col in df_sorted.columns:
            weather_data[col] = latest_row[col]

            if prev_row is not None and pd.notna(prev_row[col]) and pd.notna(latest_row[col]):
                weather_data[f'{col}_change'] = latest_row[col] - prev_row[col]

    # Include the timestamp of the current reading
    weather_data['current_datetime'] = latest_row['timestamp']

    return weather_data


def extract_seismic_data(df):
    """Extract seismic data from the combined dataset"""
    seismic_cols = ['place', 'mag', 'depth', 'longitude', 'latitude', 
                   'datetime', 'status']
    
    seismic_data = []
    for _, row in df.iterrows():
        if all(col in row and pd.notna(row[col]) for col in ['latitude', 'longitude', 'mag']):
            event = {
                'place': row.get('place', 'Unknown'),
                'mag': float(row.get('mag', 0)),
                'depth': float(row.get('depth', 0)),
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'datetime': row.get('datetime', pd.Timestamp.now()),
                'status': row.get('status', 'automatic')
            }
            seismic_data.append(event)
            
    # FOR TESTING PURPOSES ONLY:
    # data = {
    #     "time": ["2024-09-11 14:09:00"],
    #     "id": ["ci40727399"],
    #     "mag": [1.25],
    #     "place": ["4 km SSE of Highland Park, CA"],
    #     "updated": ["2024-09-09 20:56:38"],
    #     "status": ["reviewed"],
    #     "sig": [24.0],
    #     "net": ["ci"],
    #     "code": ["40727399"],
    #     "nst": [26.0],
    #     "dmin": [0.03059],
    #     "rms": [0.18],
    #     "gap": [52.0],
    #     "magType": ["ml"],
    #     "type": ["earthquake"],
    #     "title": ["M 1.3 - 4 km SSE of Highland Park, CA"],
    #     "longitude": [-118.1838333],
    #     "latitude": [34.0875],
    #     "depth": [9.25],
    #     "datetime": [pd.Timestamp("2024-09-11 14:09:00")]
    # }

    # seismic_df = pd.DataFrame(data)
    # return seismic_df
    return pd.DataFrame(seismic_data) if seismic_data else pd.DataFrame()