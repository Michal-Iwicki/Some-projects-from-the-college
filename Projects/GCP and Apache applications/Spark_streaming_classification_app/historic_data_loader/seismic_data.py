import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

pd.set_option('display.max_columns', None)


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

    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['updated'] = pd.to_datetime(df['updated'], unit='ms')

    df = (
        df
        .sort_values('mag', ascending=False)
        .groupby(df['time'].dt.floor('min'), as_index=False)
        .first()
    )

    df['time'] = pd.to_datetime(df['time'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
    df['updated'] = pd.to_datetime(df['updated'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')

    # dropping not needed columns
    df = df.drop(columns=['url','tz','tsunami', 'detail', 'ids', 'sources', 'types'])

    n = df['id'].notna().sum()
    df = df.loc[:, df.notna().sum() == n]

    return df

def main():
    starttime = "2024-10-01"
    endtime = "2024-11-30"

    output_dir = Path("data_historic/seismic")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = fetch_earthquakes(starttime=starttime, endtime=endtime)

    df['time'] = pd.to_datetime(df['time']).dt.floor('min')
    df['updated'] = pd.to_datetime(df['updated'])

    start_dt = pd.to_datetime(starttime)
    end_dt = pd.to_datetime(endtime) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)

    full_minutes = pd.date_range(start=start_dt, end=end_dt, freq='min')

    df = df.set_index('time').reindex(full_minutes)

    missing_mask = df['id'].isna()

    df.loc[missing_mask, 'updated'] = df.index[missing_mask]

    df = df.reset_index().rename(columns={'index': 'time'})

    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['updated'] = pd.to_datetime(df['updated']).dt.strftime('%Y-%m-%d %H:%M:%S')

    df['datetime'] = pd.to_datetime(df['time'])

    for (day, hour), group in df.groupby(
        [df['datetime'].dt.date, df['datetime'].dt.hour]
    ):
        filename = f"seismic_{day}_{hour:02d}.csv"
        filepath = output_dir / filename

        group.drop(columns='datetime').to_csv(filepath, index=False)
    print(f"Saved seismic data to {output_dir}")
    

if __name__ == "__main__":
    main()