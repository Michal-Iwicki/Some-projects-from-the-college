import os
from pyspark.sql import functions as F
import uuid
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

MODEL_PATH = "file:/models/GBT"

print(f"[INFO] Loading GBT model from {MODEL_PATH}")
model = PipelineModel.load(MODEL_PATH)
# ==================================================
# CONFIG
# ==================================================
LAX_LAT = 33.9416
LAX_LON = -118.4085
EARTH_RADIUS_KM = 6371.0
APP_NAME = "flight_weather_seismic_stream"

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
CHECKPOINT = "/tmp/checkpoints/flight_weather_seismic"

CASSANDRA_HOST = os.getenv("CASSANDRA_HOST", "cassandra")
KEYSPACE = "kafka_stream"
TABLE = "flight_weather_seismic"

WATERMARK = "2 minutes"

# ==================================================
# SPARK
# ==================================================

spark = (
    SparkSession.builder
    .appName(APP_NAME)
    .config("spark.cassandra.connection.host", CASSANDRA_HOST)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("INFO")

# ==================================================
# SCHEMAS
# ==================================================

flight_schema = StructType([
    StructField("Carrier", StringType(), True),
    StructField("Airline Name", StringType(), True),
    StructField("Flight_Num", IntegerType(), True),
    StructField("Origin", StringType(), True),
    StructField("Dest", StringType(), True),
    StructField("Dep_Time", IntegerType(), True),
    StructField("Actual_Dep", IntegerType(), True),
    StructField("Delay", IntegerType(), True),
    StructField("Cancelled", IntegerType(), True),
    StructField("Planned_Dep_Timestamp", LongType(), True),
    StructField("Actual_Dep_Timestamp", LongType(), True),
    StructField("Minute", StringType(), True),
    StructField("Estimated_Delay", DoubleType(), True),
    StructField("Simulation_Timestamp", StringType(), True)
])


weather_schema = StructType([
    StructField("datetime", StringType(), True),
    StructField("datetimeEpoch", LongType(), True),
    StructField("temp", DoubleType(), True),
    StructField("feelslike", DoubleType(), True),
    StructField("humidity", DoubleType(), True),
    StructField("dew", DoubleType(), True),
    StructField("precip", DoubleType(), True),
    StructField("windgust", DoubleType(), True),
    StructField("windspeed", DoubleType(), True),
    StructField("winddir", DoubleType(), True),
    StructField("pressure", DoubleType(), True),
    StructField("cloudcover", DoubleType(), True),
    StructField("visibility", DoubleType(), True),
    StructField("uvindex", DoubleType(), True),
    StructField("conditions", StringType(), True),
    StructField("icon", StringType(), True),
    StructField("source", StringType(), True),
    StructField("weather_time", StringType(), True)
])
seismic_schema = StructType([
    StructField("time", StringType(), True),
    StructField("id", StringType(), True),
    StructField("mag", DoubleType(), True),
    StructField("place", StringType(), True),
    StructField("updated", StringType(), True),
    StructField("felt", IntegerType(), True),
    StructField("cdi", DoubleType(), True),
    StructField("mmi", DoubleType(), True),
    StructField("alert", StringType(), True),
    StructField("status", StringType(), True),
    StructField("sig", IntegerType(), True),
    StructField("net", StringType(), True),
    StructField("code", StringType(), True),
    StructField("nst", IntegerType(), True),
    StructField("dmin", DoubleType(), True),
    StructField("rms", DoubleType(), True),
    StructField("gap", DoubleType(), True),
    StructField("magType", StringType(), True),
    StructField("type", StringType(), True),
    StructField("title", StringType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("depth", DoubleType(), True),
    StructField("datetime", StringType(), True)
])



# ==================================================
# KAFKA READER (BEZ event_time)
# ==================================================

def read_kafka(topic, schema, ts_col):
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", topic)
        .load()
        .selectExpr("CAST(value AS STRING)")
        .select(F.from_json("value", schema).alias("d"))
        .select("d.*")
        .withColumn(ts_col, F.to_timestamp(ts_col))
        .withWatermark(ts_col, "2 minutes")
    )


# ==================================================
# STREAMS
# ==================================================
f = read_kafka("flights", flight_schema, "Simulation_Timestamp")
w = read_kafka("weather", weather_schema, "weather_time")
s = read_kafka("seismic", seismic_schema, "datetime")


# ==================================================
# STREAM–STREAM JOINS (RÓWNOŚĆ NA GOTOWYCH KOLUMNACH)
# ==================================================
fw = f.join(
    w,
    f.Simulation_Timestamp == w.weather_time,
    "left"
).drop("weather_time")

fws = fw.join(
    s,
    fw.Simulation_Timestamp == s.datetime,
    "left"
).drop("datetime")



# ==================================================
# FEATURE ENGINEERING
# ==================================================

df = fws.withColumnRenamed("Simulation_Timestamp", "timestamp")

# ---------- SEISMIC: NULL -> 0 ----------
seismic_num_cols = [
    "mag", "sig", "nst", "dmin", "rms", "gap", "longitude", "latitude"
]

for c in seismic_num_cols:
    if c in df.columns:
        df = df.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

df = (
    df
    .withColumn("type_earthquake", F.when(F.col("type") == "earthquake", 1).otherwise(0))
    .withColumn(
        "type_other",
        F.when((F.col("type").isNotNull()) & (F.col("type") != "earthquake"), 1).otherwise(0)
    )
)

# ---------- AIRLINE ONE-HOT ----------
airlines = [
    "Alaska Airlines", "Allegiant Air", "American Airlines", "Delta",
    "Envoy Air", "Frontier Airlines", "Hawaiian Airlines",
    "Horizon Air", "Jet Blue", "SkyWest Airlines",
    "Southwest", "Spirit Airlines", "United Airlines"
]

for a in airlines:
    df = df.withColumn(
        f"Airline Name_{a}",
        F.when(F.col("`Airline Name`") == a, 1).otherwise(0)
    )

df = df.withColumn(
    "Airline Name_other",
    F.when(
        (~F.col("`Airline Name`").isin(airlines)) & F.col("`Airline Name`").isNotNull(),
        1
    ).otherwise(0)
)

# ---------- WEATHER ONE-HOT ----------
df = (
    df
    .withColumn("conditions_Clear", F.when(F.col("conditions") == "Clear", 1).otherwise(0))
    .withColumn("conditions_Overcast", F.when(F.col("conditions") == "Overcast", 1).otherwise(0))
    .withColumn(
        "conditions_Partially cloudy",
        F.when(F.col("conditions") == "Partially cloudy", 1).otherwise(0)
    )
    .withColumn(
        "conditions_other",
        F.when(
            (~F.col("conditions").isin(["Clear", "Overcast", "Partially cloudy"]))
            & F.col("conditions").isNotNull(),
            1
        ).otherwise(0)
    )
)

# ---------- TIME FEATURES ----------
df = df.withColumn(
    "Time_Till_Dep_Sec",
    (F.col("Planned_Dep_Timestamp") - F.col("timestamp").cast("long")).cast("double")
)

df = df.withColumn("time_since_last_event_sec", F.lit(0.0))

# ---------- DISTANCE FROM LAX ----------
df = df.withColumn("distance_from_LAX_km", F.lit(0.0))


# ---------- WEATHER NULL -> 0 ----------
weather_num_cols = [
    "temp", "humidity", "precip", "windgust", "windspeed",
    "winddir", "pressure", "cloudcover", "visibility"
]

for c in weather_num_cols:
    if c in df.columns:
        df = df.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))


# ==================================================
# FINAL COLUMN ORDER (MUST MATCH CASSANDRA)
# ==================================================
final_cols = [
    # ======================
    # PRIMARY KEY
    # ======================
    "flight_num",
    "planned_dep_timestamp",
    "timestamp",

    # ======================
    # FLIGHT (UI)
    # ======================
    "carrier",
    "airline_name",
    "origin",
    "dest",
    "dep_time",
    "estimated_delay",

    # ======================
    # WEATHER (UI)
    # ======================
    "temp",
    "humidity",
    "precip",
    "windspeed",
    "pressure",
    "visibility",
    "conditions",

    # ======================
    # SEISMIC (UI – summary)
    # ======================
    "mag",
    "sig",
    "distance_from_lax_km",

    # ======================
    # ML OUTPUT
    # ======================
    "prediction",
    "probability"
]

for c in df.columns:
    new_name = c.lower().replace(" ", "_")
    df = df.withColumnRenamed(c, new_name)

# ==================================================
# WRITE TO CASSANDRA
# ==================================================

def write_to_cassandra(batch_df, batch_id):
    scored = model.transform(batch_df)
    scored = scored.withColumn(
            "probability",
            vector_to_array("probability")[1]
        )

    (
        scored.select(*final_cols)
        .write
        .format("org.apache.spark.sql.cassandra")
        .options(
            keyspace=KEYSPACE,
            table=TABLE
        )
        .mode("append")
        .save()
    )

(
    df.writeStream
    .foreachBatch(write_to_cassandra)
    .option("checkpointLocation", CHECKPOINT)
    .outputMode("append")
    .start()
    .awaitTermination()
)
