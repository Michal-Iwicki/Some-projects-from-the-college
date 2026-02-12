import time
import os
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

BASE_DIR = Path(__file__).resolve().parent

def simulate_estimated_delay_backward(
    minutes_to_departure,
    final_delay,
    reveal_bias,
    noise_rng
):
    # No delay shown if no final delay
    if final_delay <= 0:
        return 0.0

    # >6h out → nothing
    if minutes_to_departure > 360:
        return 0.0

    # Progress: 6h out -> 0, departure -> 1
    progress = 1 - (minutes_to_departure / 360)
    progress = np.clip(progress, 0, 1)

    # Small delays revealed late, big ones earlier
    if final_delay <= 15:
        base_power = 4.0
    elif final_delay <= 30:
        base_power = 2.5
    else:
        base_power = 1.5

    reveal_power = np.clip(base_power + reveal_bias, 1.2, 5.0)

    revealed_fraction = progress ** reveal_power
    expected = final_delay * revealed_fraction

    # Fading noise (cannot break constraints)
    noise_scale = (1 - progress) * min(3, final_delay * 0.1)
    noise = noise_rng.normal(0, noise_scale)

    estimate = expected + noise

    # Past planned departure → must reflect elapsed delay
    if minutes_to_departure < 0:
        estimate = max(estimate, abs(minutes_to_departure))

    # Hard bounds
    return float(np.clip(estimate, 0, final_delay))


def save_daily_snapshots(df, output_dir=f"{BASE_DIR}\\flights"):
    os.makedirs(output_dir, exist_ok=True)

    df = df.copy()
    df["Planned_DT"] = pd.to_datetime(df["Planned_Dep_Timestamp"], unit="s")
    df["Actual_DT"] = pd.to_datetime(df["Actual_Dep_Timestamp"], unit="s", errors="coerce")

    while True:
        now = datetime.now() - relativedelta(years=1, hours=12)
        today = now.date()

        snapshot = df[df["Planned_DT"].dt.date == today].copy()
        snapshot["Estimated_Delay"] = None

        for idx, row in snapshot.iterrows():
            planned = row["Planned_DT"]
            actual = row["Actual_DT"]

            if pd.isna(actual) or actual > now:
                snapshot.at[idx, "Actual_Dep"] = None
                snapshot.at[idx, "Actual_Dep_Timestamp"] = None
                snapshot.at[idx, "Delay"] = None
                snapshot.at[idx, "Cancelled"] = None

                minutes_to_departure = (planned - now).total_seconds() / 60

                seed = abs(hash((row["Flight_Num"], planned.date()))) % (2**32)
                rng = np.random.default_rng(seed)

                final_delay = row["Delay"]  # ground truth

                # Some flights reveal earlier / later
                reveal_bias = rng.normal(0, 0.4)

                # Stable noise across snapshots
                noise_rng = np.random.default_rng(seed + 1)

                est_delay = simulate_estimated_delay_backward(
                    minutes_to_departure=minutes_to_departure,
                    final_delay=final_delay,
                    reveal_bias=reveal_bias,
                    noise_rng=noise_rng
                )

                snapshot.at[idx, "Estimated_Delay"] = round(est_delay, 0)

            else:
                snapshot.at[idx, "Estimated_Delay"] = row["Delay"]

        filename = now.strftime("%Y-%m-%d_%H-%M") + ".csv"
        filepath = os.path.join(output_dir, filename)

        snapshot.drop(columns=["Planned_DT", "Actual_DT"], inplace=True)
        snapshot.to_csv(filepath, index=False)

        print(f"Saved snapshot: {filepath}")
        time.sleep(60)


def main():
    df = pd.read_csv(f"{BASE_DIR}\\flights_test.csv")
    save_daily_snapshots(df)


if __name__ == '__main__':
    main()