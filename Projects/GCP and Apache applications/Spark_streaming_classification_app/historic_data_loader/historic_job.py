"""
Convert all timestamps to pandas datetime

Weather: join on the nearest hourly weather record

Seismic: aggregate earthquakes into time windows before each flight
(e.g., number of quakes in last 6 hours, max magnitude)

Merge everything into a single flight-level table

Target variable = Delay or Cancelled
"""
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from data.simulate_flights import BASE_DIR

pd.set_option('display.max_columns', None)

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_FILE = f"{BASE_DIR}\\processed_dates.txt"
BASE_DIR = BASE_DIR.parent


def earthquakes_to_dataframe(data: dict) -> pd.DataFrame:
    rows = []

    for feature in data.get("features", []):
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [None, None, None])

        row = {
            # ID
            "id": feature.get("id"),

            # Properties
            "mag": props.get("mag"),
            "place": props.get("place"),
            "time": props.get("time"),
            "updated": props.get("updated"),
            "tz": props.get("tz"),
            "url": props.get("url"),
            "detail": props.get("detail"),
            "felt": props.get("felt"),
            "cdi": props.get("cdi"),
            "mmi": props.get("mmi"),
            "alert": props.get("alert"),
            "status": props.get("status"),
            "tsunami": props.get("tsunami"),
            "sig": props.get("sig"),
            "net": props.get("net"),
            "code": props.get("code"),
            "ids": props.get("ids"),
            "sources": props.get("sources"),
            "types": props.get("types"),
            "nst": props.get("nst"),
            "dmin": props.get("dmin"),
            "rms": props.get("rms"),
            "gap": props.get("gap"),
            "magType": props.get("magType"),
            "type": props.get("type"),
            "title": props.get("title"),

            # Geometry (schema mapping)
            "longitude": coords[0],
            "latitude": coords[1],
            "depth": coords[2],
        }

        rows.append(row)

    return pd.DataFrame(rows)


def fetch_earthquakes(
        starttime: str,
        endtime: str,
        latitude: float = 34.0522,
        longitude: float = -118.2437,
        maxradiuskm: int = 100
):
    """
    Fetch earthquake data from the USGS API.

    starttime, endtime: ISO strings like '2024-09-01T00:00:00'
    """
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    params = {
        "format": "geojson",
        "starttime": starttime,
        "endtime": endtime,
        "latitude": latitude,
        "longitude": longitude,
        "maxradiuskm": maxradiuskm,
        "eventtype": "earthquake",
        "orderby": "time",
    }

    response = requests.get(url, params=params)
    data = response.json()

    df = earthquakes_to_dataframe(data)

    return df


def weather_to_dataframe(data: dict) -> pd.DataFrame:
    rows = []

    for day in data.get("days", []):
        for hour in day.get("hours", []):
            row = {
                "datetime": hour.get("datetime"),
                "datetimeEpoch": hour.get("datetimeEpoch"),
                "temp": hour.get("temp"),
                "feelslike": hour.get("feelslike"),
                "humidity": hour.get("humidity"),
                "dew": hour.get("dew"),
                "precip": hour.get("precip"),
                "windgust": hour.get("windgust"),
                "windspeed": hour.get("windspeed"),
                "winddir": hour.get("winddir"),
                "pressure": hour.get("pressure"),
                "cloudcover": hour.get("cloudcover"),
                "visibility": hour.get("visibility"),
                "uvindex": hour.get("uvindex"),
                "conditions": hour.get("conditions"),
                "icon": hour.get("icon"),
                "source": hour.get("source"),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def fetch_weather(
        start_date: str,
        end_date: str,
        location: str = "KLAX",
        api_key: str = "35UKBH4MZ8E34ABXX5872T7AJ"
) -> pd.DataFrame:
    """
    Fetch hourly weather data day-by-day to avoid cost overruns.
    """

    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    all_rows = []

    current = start
    while current < end:
        day_str = current.strftime("%Y-%m-%d")

        url = (
            "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/"
            f"timeline/{location}/{day_str}"
        )

        params = {
            "unitGroup": "metric",
            "contentType": "json",
            "key": api_key,
            "include": "hours",
        }

        response = requests.get(url, params=params)

        if response.status_code == 429:
            raise RuntimeError(
                "Visual Crossing API quota exceeded. "
                "Wait until tomorrow or reduce request volume."
            )

        response.raise_for_status()

        data = response.json()

        df_day = weather_to_dataframe(data)
        all_rows.append(df_day)

        current += timedelta(days=1)

    return pd.concat(all_rows, ignore_index=True)


def fetch_flights(start_date: str, end_date: str):
    df = pd.read_csv(f"{BASE_DIR}\\data\\flights_test.csv")

    # Parse timestamps
    df["Minute"] = pd.to_datetime(df["Minute"])
    df["flight_time"] = pd.to_datetime(df["Actual_Dep_Timestamp"], unit="s")

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    df = df[
        (df["Minute"] >= start_dt) &
        (df["Minute"] < end_dt)
        ].copy()

    # Targets
    df["delay_minutes"] = df["Delay"]
    df["cancelled"] = df["Cancelled"]

    return df


def seismic_features(flight_time, seismic_df, window_hours=6):
    start = flight_time - pd.Timedelta(hours=window_hours)

    recent = seismic_df[
        (seismic_df["quake_time"] >= start) &
        (seismic_df["quake_time"] <= flight_time)
        ]

    return pd.Series({
        "quake_count_6h": len(recent),
        "max_mag_6h": recent["mag"].max() if not recent.empty else 0,
        "mean_depth_6h": recent["depth"].mean() if not recent.empty else 0
    })


def load_processed_dates():
    """Load processed date ranges from file."""
    if not os.path.exists(PROCESSED_FILE):
        # Create an empty file
        with open(PROCESSED_FILE, "w") as f:
            pass
        return set()

    with open(PROCESSED_FILE, "r") as f:
        lines = f.read().strip().splitlines()

    return set(line.strip() for line in lines)


def save_processed_date(start_date, end_date):
    """Append a processed date range to the tracking file."""
    with open(PROCESSED_FILE, "a") as f:
        f.write(f"{start_date},{end_date}\n")


def generate_weekly_queries(
        start_date="2024-09-08",
        end_date="2024-12-31"
):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Weekly boundaries (7-day frequency)
    dates = pd.date_range(start=start, end=end, freq="7D")

    queries = []
    for i in range(len(dates)):
        week_start = dates[i]
        # Ensure the last week ends exactly at the end_date
        if i + 1 < len(dates):
            week_end = dates[i + 1]
        else:
            week_end = end + pd.Timedelta(days=1)

        queries.append((
            week_start.strftime("%Y-%m-%d"),
            week_end.strftime("%Y-%m-%d")
        ))

    return queries


def main():
    os.makedirs(f"{BASE_DIR}\\data_historic", exist_ok=True)

    processed = load_processed_dates()
    queries = generate_weekly_queries(
        start_date="2024-09-08",
        end_date="2024-12-31"
    )

    for start_date, end_date in queries:
        key = f"{start_date},{end_date}"

        # Skip if already processed
        if key in processed:
            print(f"Skipping {start_date} → {end_date} (already processed)")
            continue
        print(f"\nProcessing {start_date} → {end_date}")

        # -------------------- Seismic --------------------
        seismic = fetch_earthquakes(
            f"{start_date}T00:00:00",
            f"{end_date}T00:00:00"
        )
        seismic["quake_time"] = pd.to_datetime(seismic["time"], unit="ms")

        # -------------------- Weather --------------------
        weather = fetch_weather(start_date, end_date)
        weather["weather_time"] = pd.to_datetime(
            weather["datetimeEpoch"], unit="s"
        )

        # -------------------- Flights --------------------
        flights = fetch_flights(start_date, end_date)

        # -------------------- Merge flights + weather --------------------
        flights = flights.sort_values("flight_time")
        weather = weather.sort_values("weather_time")

        flights_weather = pd.merge_asof(
            flights,
            weather,
            left_on="flight_time",
            right_on="weather_time",
            direction="backward"
        )

        # -------------------- Seismic features --------------------
        seismic_features_df = flights_weather["flight_time"].apply(
            lambda t: seismic_features(t, seismic, window_hours=6)
        )

        final_df = pd.concat(
            [
                flights_weather.reset_index(drop=True),
                seismic_features_df.reset_index(drop=True),
            ],
            axis=1
        )

        # -------------------- Save CSV --------------------
        filename = f"{BASE_DIR}\\data_historic\\snapshot_{start_date}_to_{end_date}.csv"
        final_df.to_csv(filename, index=False)

        print(f"Saved {filename} ({len(final_df)} rows)")

        # Mark as processed
        save_processed_date(start_date, end_date)


if __name__ == "__main__":
    main()
