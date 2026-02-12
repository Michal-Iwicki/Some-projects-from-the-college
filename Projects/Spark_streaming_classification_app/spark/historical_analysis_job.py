import os

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import input_file_name

# ===================================================
# CONFIG
# ===================================================

APP_NAME = "Historical Analysis"

HDFS_PATH = "hdfs://hdfs-namenode:8020/data/preprocessed/"

CASSANDRA_HOST = os.getenv("CASSANDRA_HOST", "cassandra")

# ===================================================
# Spark session
# ===================================================

spark = (
    SparkSession.builder
    .appName(APP_NAME)
    .config("spark.cassandra.connection.host", CASSANDRA_HOST)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("INFO")

print("======================================")
print("[INFO] Historical Analysis job started")
print("======================================")

# ===================================================
# Read data from HDFS (Avro)
# ===================================================

print(f"[INFO] Reading data from HDFS: {HDFS_PATH}")

df_raw = (
    spark.read
    .format("avro")
    .option("mergeSchema", "false")
    .load(HDFS_PATH)
)

df_raw = df_raw.withColumn("_source_file", input_file_name())

print(f"[INFO] Rows loaded: {df_raw.count()}")
df_raw.printSchema()

# Parse timestamps
df = (
    df_raw
    .withColumn("Minute_ts", F.to_timestamp("Minute"))
    .withColumn("Planned_Dep_Timestamp_ts", F.to_timestamp("Planned_Dep_Timestamp"))
    .withColumn("Actual_Dep_Timestamp_ts", F.to_timestamp("Actual_Dep_Timestamp"))
    .withColumn("board_ts", F.to_timestamp("timestamp"))  # board refresh time
)

# -------------------------------------------------------------------
# 2. Reconstruct logical tables with deduplication
# -------------------------------------------------------------------

# ---------- Flights ----------
# Keep only rows that look like real flights
flights = df.filter(F.col("Flight_Num").isNotNull())

# Deduplicate: for each flight, keep the latest board snapshot
w_flights = Window.partitionBy(
    "Flight_Num", "Actual_Dep_Timestamp_ts"
).orderBy(F.col("board_ts").desc())

flights_dedup = (
    flights
    .withColumn("rn", F.row_number().over(w_flights))
    .filter(F.col("rn") == 1)
    .drop("rn")
)

# Use Minute_ts as primary datetime
flights_dedup = flights_dedup.withColumn(
    "datetime",
    F.coalesce("Minute_ts", "Actual_Dep_Timestamp_ts")
)

flights_dedup = (
    flights_dedup
    .withColumn("date", F.to_date("datetime"))
    .withColumn("year", F.year("datetime"))
    .withColumn("month", F.month("datetime"))
    .withColumn("month_start", F.trunc("datetime", "month"))
    .withColumn(
        "week_start",
        F.date_sub(F.to_date("datetime"), F.dayofweek("datetime") - 2)  # Monday as start
    )
    .withColumn("hour", F.hour("datetime"))
    .withColumn("day_of_week", F.date_format("datetime", "EEEE"))
)

# ---------- Seismic ----------
# mag > 0 => earthquake event
seismic = df.filter(F.col("mag").isNotNull() & (F.col("mag") > 0) & (F.col("time_since_last_event_sec") == 0))

# Deduplicate seismic events across board refreshes
# Use (board_ts, mag, longitude, latitude) as event identity
w_seis = Window.partitionBy(
    "board_ts", "mag", "longitude", "latitude"
).orderBy(F.col("board_ts").desc())

seismic_dedup = (
    seismic
    .withColumn("rn", F.row_number().over(w_seis))
    .filter(F.col("rn") == 1)
    .drop("rn")
)

# Use board_ts as event time
seismic_dedup = (
    seismic_dedup
    .withColumn("time", F.col("board_ts"))
    .withColumn("date", F.to_date("time"))
    .withColumn("datetime", F.col("time"))
    .withColumn("year", F.year("time"))
    .withColumn("month", F.month("time"))
    .withColumn("month_start", F.trunc("time", "month"))
    .withColumn(
        "week_start",
        F.date_sub(F.to_date("time"), F.dayofweek("time") - 2)
    )
    .withColumn("hour", F.hour("time"))
    .withColumn("day_of_week", F.date_format("time", "EEEE"))
)

# ---------- Weather ----------
# Assume one weather record per board_ts; deduplicate by board_ts
weather = df.filter(F.col("temp").isNotNull())

w_weather = Window.partitionBy("board_ts").orderBy(F.col("board_ts").desc())

weather_dedup = (
    weather
    .withColumn("rn", F.row_number().over(w_weather))
    .filter(F.col("rn") == 1)
    .drop("rn")
)

weather_dedup = (
    weather_dedup
    .withColumn("datetime", F.col("board_ts"))
    .withColumn("date", F.to_date("datetime"))
    .withColumn("year", F.year("datetime"))
    .withColumn("month", F.month("datetime"))
    .withColumn("month_start", F.trunc("datetime", "month"))
    .withColumn(
        "week_start",
        F.date_sub(F.to_date("datetime"), F.dayofweek("datetime") - 2)
    )
    .withColumn("hour", F.hour("datetime"))
)

# -------------------------------------------------------------------
# 3. Monthly aggregations (Spark version of create_monthly_aggregations)
# -------------------------------------------------------------------

# Collect all months present in any of the three tables
months_seis = seismic_dedup.select("month_start").distinct()
months_flt = flights_dedup.select("month_start").distinct()
months_wth = weather_dedup.select("month_start").distinct()

all_months = (
    months_seis.union(months_flt).union(months_wth)
    .distinct()
    .withColumn("year", F.year("month_start"))
    .withColumn("month", F.month("month_start"))
    .withColumn("month_name", F.date_format("month_start", "MMMM"))
)

# ----- Seismic monthly -----
seis_monthly = (
    seismic_dedup
    .groupBy("month_start")
    .agg(
        F.count("*").alias("seismic_total_events"),
        F.avg("mag").alias("seismic_avg_magnitude"),
        F.max("mag").alias("seismic_max_magnitude"),
        F.min("mag").alias("seismic_min_magnitude"),
        F.stddev("mag").alias("seismic_std_magnitude"),
        F.expr("percentile_approx(mag, 0.5)").alias("seismic_median_magnitude"),
        # F.avg("depth").alias("seismic_avg_depth"),
        # F.max("depth").alias("seismic_max_depth"),
        F.sum(F.when(F.col("mag") >= 3.0, 1).otherwise(0)).alias("seismic_significant_events")
    )
    .withColumn("seismic_events_per_day", F.col("seismic_total_events") / F.lit(30.0))
)

# ----- Flights monthly -----
flights_monthly = (
    flights_dedup
    .groupBy("month_start")
    .agg(
        F.count("*").alias("flights_total"),
        F.avg("Delay").alias("flights_avg_delay"),
        F.max("Delay").alias("flights_max_delay"),
        F.min("Delay").alias("flights_min_delay"),
        F.stddev("Delay").alias("flights_std_delay"),
        F.expr("percentile_approx(Delay, 0.5)").alias("flights_median_delay"),
        F.avg(F.col("Cancelled").cast("double")).alias("flights_cancellation_rate"),
        F.sum("Cancelled").alias("flights_cancelled_total"),
        F.avg(F.when(F.col("Delay") <= 0, 1.0).otherwise(0.0)).alias("flights_on_time_rate"),
        F.avg(F.when(F.col("Delay") > 60, 1.0).otherwise(0.0)).alias("flights_severe_delay_rate"),
    )
    .withColumn("flights_per_day", F.col("flights_total") / F.lit(30.0))
)

# Top carrier and route monthly
if "Carrier" in flights_dedup.columns:
    carrier_monthly = (
        flights_dedup
        .groupBy("month_start", "Carrier")
        .agg(F.count("*").alias("cnt"))
    )
    w_carrier = Window.partitionBy("month_start").orderBy(F.col("cnt").desc())
    carrier_monthly_top = (
        carrier_monthly
        .withColumn("rn", F.row_number().over(w_carrier))
        .filter(F.col("rn") == 1)
        .select(
            "month_start",
            F.col("Carrier").alias("flights_top_carrier"),
            (F.col("cnt") / F.sum("cnt").over(Window.partitionBy("month_start"))).alias("flights_top_carrier_share")
        )
    )
else:
    carrier_monthly_top = spark.createDataFrame([], all_months.schema)

if set(["Origin", "Dest"]).issubset(flights_dedup.columns):
    flights_routes = flights_dedup.withColumn("route", F.concat_ws("-", "Origin", "Dest"))
    route_monthly = (
        flights_routes
        .groupBy("month_start", "route")
        .agg(F.count("*").alias("cnt"))
    )
    w_route = Window.partitionBy("month_start").orderBy(F.col("cnt").desc())
    route_monthly_top = (
        route_monthly
        .withColumn("rn", F.row_number().over(w_route))
        .filter(F.col("rn") == 1)
        .select(
            "month_start",
            F.col("route").alias("flights_top_route")
        )
    )
else:
    route_monthly_top = spark.createDataFrame([], all_months.schema)

# ----- Weather monthly -----
weather_monthly = (
    weather_dedup
    .groupBy("month_start")
    .agg(
        F.avg("temp").alias("weather_avg_temp"),
        F.max("temp").alias("weather_max_temp"),
        F.min("temp").alias("weather_min_temp"),
        F.stddev("temp").alias("weather_std_temp"),
        F.avg("humidity").alias("weather_avg_humidity"),
        F.avg("windspeed").alias("weather_avg_windspeed"),
        F.max("windspeed").alias("weather_max_windspeed"),
        F.avg("pressure").alias("weather_avg_pressure"),
        F.sum("precip").alias("weather_total_precipitation"),
        F.countDistinct(F.when(F.col("precip") > 0, F.col("date"))).alias("weather_rainy_days")
    )
)

if "conditions" in weather_dedup.columns:
    cond_monthly = (
        weather_dedup
        .groupBy("month_start", "conditions")
        .agg(F.count("*").alias("cnt"))
    )
    w_cond = Window.partitionBy("month_start").orderBy(F.col("cnt").desc())
    cond_monthly_top = (
        cond_monthly
        .withColumn("rn", F.row_number().over(w_cond))
        .filter(F.col("rn") == 1)
        .select(
            "month_start",
            F.col("conditions").alias("weather_common_condition"),
            (F.col("cnt") / F.sum("cnt").over(Window.partitionBy("month_start"))).alias("weather_common_condition_pct")
        )
    )
else:
    cond_monthly_top = spark.createDataFrame([], all_months.schema)


# ----- Join all monthly pieces -----
def drop_date_cols(df):
    return df.drop(
        *[c for c in df.columns if c in ("year", "month", "month_name")]
    )


seis_monthly_clean = drop_date_cols(seis_monthly)
flights_monthly_clean = drop_date_cols(flights_monthly)
carrier_monthly_top_clean = drop_date_cols(carrier_monthly_top)
route_monthly_top_clean = drop_date_cols(route_monthly_top)
weather_monthly_clean = drop_date_cols(weather_monthly)
cond_monthly_top_clean = drop_date_cols(cond_monthly_top)

monthly_agg = (
    all_months
    .join(seis_monthly_clean, "month_start", "left")
    .join(flights_monthly_clean, "month_start", "left")
    .join(carrier_monthly_top_clean, "month_start", "left")
    .join(route_monthly_top_clean, "month_start", "left")
    .join(weather_monthly_clean, "month_start", "left")
    .join(cond_monthly_top_clean, "month_start", "left")
    .orderBy("month_start")
)

# -------------------------------------------------------------------
# 4. Weekly aggregations (Spark version of create_weekly_aggregations)
# -------------------------------------------------------------------

weeks_seis = seismic_dedup.select("week_start").distinct()
weeks_flt = flights_dedup.select("week_start").distinct()
weeks_wth = weather_dedup.select("week_start").distinct()

all_weeks = (
    weeks_seis.union(weeks_flt).union(weeks_wth)
    .distinct()
)

# ----- Seismic weekly -----
seis_weekly = (
    seismic_dedup
    .groupBy("week_start")
    .agg(
        F.count("*").alias("seismic_total_events"),
        F.avg("mag").alias("seismic_avg_magnitude"),
        F.max("mag").alias("seismic_max_magnitude"),
        F.min("mag").alias("seismic_min_magnitude"),
        F.expr("percentile_approx(mag, 0.5)").alias("seismic_median_magnitude"),
        (F.count("*") / F.lit(7.0)).alias("seismic_events_per_day")
    )
)

# Peak hour & weekday/weekend counts:
seis_weekly_extra = (
    seismic_dedup
    .withColumn("is_weekend", F.col("day_of_week").isin("Saturday", "Sunday"))
    .groupBy("week_start")
    .agg(
        F.sum(F.when(F.col("is_weekend"), 1).otherwise(0)).alias("seismic_weekend_events"),
        F.sum(F.when(~F.col("is_weekend"), 1).otherwise(0)).alias("seismic_weekday_events")
    )
)

# Peak hour, separate aggregation:
seis_peak_hour = (
    seismic_dedup
    .groupBy("week_start", "hour")
    .agg(F.count("*").alias("cnt"))
)
w_peak = Window.partitionBy("week_start").orderBy(F.col("cnt").desc())
seis_peak_hour_top = (
    seis_peak_hour
    .withColumn("rn", F.row_number().over(w_peak))
    .filter(F.col("rn") == 1)
    .select("week_start", F.col("hour").alias("seismic_peak_hour"))
)

seis_weekly = (
    all_weeks
    .join(seis_weekly, "week_start", "left")
    .join(seis_weekly_extra, "week_start", "left")
    .join(seis_peak_hour_top, "week_start", "left")
)

# ----- Flights weekly -----
flights_weekly_base = (
    flights_dedup
    .groupBy("week_start")
    .agg(
        F.count("*").alias("flights_total"),
        F.avg("Delay").alias("flights_avg_delay"),
        F.max("Delay").alias("flights_max_delay"),
        F.min("Delay").alias("flights_min_delay"),
        F.expr("percentile_approx(Delay, 0.5)").alias("flights_median_delay"),
        F.avg(F.col("Cancelled").cast("double")).alias("flights_cancellation_rate"),
        F.sum("Cancelled").alias("flights_cancelled_total"),
        F.avg(F.when(F.col("Delay") <= 0, 1.0).otherwise(0.0)).alias("flights_on_time_rate"),
        F.avg(F.when(F.col("Delay") > 60, 1.0).otherwise(0.0)).alias("flights_severe_delay_rate"),
        (F.count("*") / F.lit(7.0)).alias("flights_per_day")
    )
)

# Peak hour
flights_peak_hour = (
    flights_dedup
    .groupBy("week_start", "hour")
    .agg(F.count("*").alias("cnt"))
)
w_flt_peak = Window.partitionBy("week_start").orderBy(F.col("cnt").desc())
flights_peak_hour_top = (
    flights_peak_hour
    .withColumn("rn", F.row_number().over(w_flt_peak))
    .filter(F.col("rn") == 1)
    .select("week_start", F.col("hour").alias("flights_peak_hour"))
)

# Top carrier & route weekly (optional)
if "Carrier" in flights_dedup.columns:
    carrier_weekly = (
        flights_dedup
        .groupBy("week_start", "Carrier")
        .agg(F.count("*").alias("cnt"))
    )
    w_cw = Window.partitionBy("week_start").orderBy(F.col("cnt").desc())
    carrier_weekly_top = (
        carrier_weekly
        .withColumn("rn", F.row_number().over(w_cw))
        .filter(F.col("rn") == 1)
        .select("week_start", F.col("Carrier").alias("flights_top_carrier"))
    )
else:
    carrier_weekly_top = spark.createDataFrame([], all_weeks.schema)

if set(["Origin", "Dest"]).issubset(flights_dedup.columns):
    flights_routes_w = flights_dedup.withColumn("route", F.concat_ws("-", "Origin", "Dest"))
    route_weekly = (
        flights_routes_w
        .groupBy("week_start", "route")
        .agg(F.count("*").alias("cnt"))
    )
    w_rw = Window.partitionBy("week_start").orderBy(F.col("cnt").desc())
    route_weekly_top = (
        route_weekly
        .withColumn("rn", F.row_number().over(w_rw))
        .filter(F.col("rn") == 1)
        .select("week_start", F.col("route").alias("flights_top_route"))
    )
else:
    route_weekly_top = spark.createDataFrame([], all_weeks.schema)

flights_weekly = (
    all_weeks
    .join(flights_weekly_base, "week_start", "left")
    .join(flights_peak_hour_top, "week_start", "left")
    .join(carrier_weekly_top, "week_start", "left")
    .join(route_weekly_top, "week_start", "left")
)

# ----- Weather weekly -----
weather_weekly = (
    weather_dedup
    .groupBy("week_start")
    .agg(
        F.avg("temp").alias("weather_avg_temp"),
        F.max("temp").alias("weather_max_temp"),
        F.min("temp").alias("weather_min_temp"),
        F.avg("humidity").alias("weather_avg_humidity"),
        F.avg("windspeed").alias("weather_avg_windspeed"),
        F.max("windspeed").alias("weather_max_windspeed"),
        F.avg("pressure").alias("weather_avg_pressure"),
        F.sum("precip").alias("weather_total_precipitation"),
        F.countDistinct(F.when(F.col("precip") > 0, F.col("date"))).alias("weather_rainy_days")
    )
)

if "conditions" in weather_dedup.columns:
    cond_weekly = (
        weather_dedup
        .groupBy("week_start", "conditions")
        .agg(F.count("*").alias("cnt"))
    )
    w_cw2 = Window.partitionBy("week_start").orderBy(F.col("cnt").desc())
    cond_weekly_top = (
        cond_weekly
        .withColumn("rn", F.row_number().over(w_cw2))
        .filter(F.col("rn") == 1)
        .select("week_start", F.col("conditions").alias("weather_common_condition"))
    )
else:
    cond_weekly_top = spark.createDataFrame([], all_weeks.schema)

weekly_agg = (
    all_weeks
    .join(seis_weekly, "week_start", "left")
    .join(flights_weekly, "week_start", "left")
    .join(weather_weekly, "week_start", "left")
    .join(cond_weekly_top, "week_start", "left")
    .orderBy("week_start")
)


# -------------------------------------------------------------------
# 5. Save to Cassandra
# -------------------------------------------------------------------

# Write monthly aggregations to Cassandra
(
    monthly_agg
    .write
    .format("org.apache.spark.sql.cassandra")
    .mode("append")
    .options(table="monthly_aggregations", keyspace="analytics")
    .save()
)

# Write weekly aggregations to Cassandra
(
    weekly_agg
    .write
    .format("org.apache.spark.sql.cassandra")
    .mode("append")
    .options(table="weekly_aggregations", keyspace="analytics")
    .save()
)


print("[INFO] Write to Cassandra finished successfully")

print("======================================")
print("[INFO] Historical Analysis job finished")
print("======================================")

spark.stop()
