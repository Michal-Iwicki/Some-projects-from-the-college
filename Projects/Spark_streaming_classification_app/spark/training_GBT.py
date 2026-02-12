from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, to_timestamp, rand, unix_timestamp
)
from pyspark.sql.types import NumericType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os
from pyspark.sql import functions as F

SEED = 42

# ===================================================
# Spark session
# ===================================================
spark = (
    SparkSession.builder
    .appName("FlightDelayGBT-IncrementalSampling")
    .getOrCreate()
)

# ===================================================
# Read Avro
# ===================================================
path = "hdfs://hdfs-namenode:8020/data/preprocessed/"

df = (
    spark.read
    .format("avro")
    .option("mergeSchema", "false")
    .load(path)
)

# ===================================================
# Timestamp normalization
# ===================================================
df = (
    df
    .withColumn("planned_dep_timestamp", to_timestamp(col("Planned_Dep_Timestamp")))
    .withColumn("actual_dep_timestamp", to_timestamp(col("Actual_Dep_Timestamp")))
    .withColumn("timestamp", to_timestamp(col("timestamp")))
)

# ===================================================
# Fill NULLs (numeric only)
# ===================================================
numeric_cols = [
    f.name for f in df.schema.fields
    if isinstance(f.dataType, NumericType)
]
df = df.fillna(0, subset=numeric_cols)

# ===================================================
# Label
# ===================================================
df = df.withColumn("label", when(col("Delay") > 0, 1).otherwise(0))

# ===================================================
# Normalize column names
# ===================================================
df = df.select([
    F.col(c).alias(c.lower().replace(" ", "_"))
    for c in df.columns
])

# ===================================================
# Feature columns
# ===================================================
airlines = [
    "alaska_airlines", "allegiant_air", "american_airlines",
    "delta", "envoy_air", "frontier_airlines", "hawaiian_airlines",
    "horizon_air", "jet_blue", "skywest_airlines",
    "southwest", "spirit_airlines", "united_airlines"
]

feature_cols = [
    *[f"airline_name_{a}" for a in airlines],
    "airline_name_other",
    "time_till_dep_sec",
    "mag", "sig", "nst", "dmin", "rms", "gap",
    "longitude", "latitude", "distance_from_lax_km",
    "type_earthquake", "type_other", "time_since_last_event_sec",
    "temp", "humidity", "precip", "windgust", "windspeed", "winddir",
    "pressure", "cloudcover", "visibility",
    "conditions_clear", "conditions_overcast",
    "conditions_partially_cloudy", "conditions_other"
]

missing = set(feature_cols + ["label", "timestamp"]) - set(df.columns)
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

# ===================================================
# Base dataset
# ===================================================
data = df.select(*feature_cols, "label", "timestamp")
data = data.withColumn("ts_long", unix_timestamp("timestamp")).cache()
data.count()

# ===================================================
# Chronological split (80 / 20)
# ===================================================
cutoff_long = data.approxQuantile("ts_long", [0.8], 0.001)[0]

train_full = data.filter(col("ts_long") <= cutoff_long).cache()
test_full  = data.filter(col("ts_long") > cutoff_long)

# ===================================================
# TEST: random 5% (FAST, ONCE)
# ===================================================
test_df = (
    test_full
    .sample(False, 0.05, seed=SEED)
    .drop("ts_long", "timestamp")
    .cache()
)
test_size = test_df.count()

test_dist = (
    test_df
    .groupBy("label")
    .count()
    .collect()
)
# ===================================================
# TRAIN: random order ONCE
# ===================================================
train_full = (
    train_full
    .withColumn("rnd", rand(SEED))
    .drop("ts_long")
    .cache()
)

# Split by label ONCE
train_pos = train_full.filter(col("label") == 1).cache()
train_neg = train_full.filter(col("label") == 0).cache()
train_pos.count()
train_neg.count()

# ===================================================
# Vector assembler
# ===================================================
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

# ===================================================
# Model
# ===================================================
gbt = GBTClassifier(
    labelCol="label",
    featuresCol="features",
    maxIter=80,
    maxDepth=6,
    stepSize=0.05,
    subsamplingRate=0.8,
    seed=SEED
)

pipeline = Pipeline(stages=[assembler, gbt])

evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    metricName="areaUnderROC"
)

# ===================================================
# Experiments (INCREMENTAL)
# ===================================================
TRAIN_FRACTIONS = [0.01, 0.02, 0.05, 0.10]

report_dir = "/reports"
model_dir = "/models"
os.makedirs(report_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

for frac in TRAIN_FRACTIONS:
    print(f"\n=== TRAIN FRACTION {int(frac*100)}% ===")

    pos_part = train_pos.filter(col("rnd") <= frac)
    neg_part = train_neg.filter(col("rnd") <= frac)

    pos_cnt = pos_part.count()
    neg_cnt = neg_part.count()

    ratio = pos_cnt / neg_cnt
    neg_down = neg_part.sample(False, ratio, seed=SEED)

    train_bal = (
        pos_part.unionByName(neg_down)
        .drop("timestamp", "rnd")
        .cache()
    )
    train_bal.count()

    model = pipeline.fit(train_bal)
    preds = model.transform(test_df)

    auc = evaluator.evaluate(preds)
    model_path = f"file:{model_dir}/GBT_train_{int(frac*100)}pct"
    model.write().overwrite().save(model_path)
    # ===============================
    # Report
    # ===============================
    train_size = train_bal.count()

    train_dist = (
        train_bal
        .groupBy("label")
        .count()
        .collect()
    )

    cm = (
        preds
        .groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
        .collect()
    )

    report_path = f"{report_dir}/GBT_train_{int(frac*100)}pct.txt"
    with open(report_path, "w") as f:
        f.write("Model: GBT\n")
        f.write(f"AUC: {auc:.4f}\n\n")

        f.write("=== DATASET SIZES ===\n")
        f.write(f"Train size: {train_size}\n")
        f.write(f"Test size: {test_size}\n\n")

        f.write("=== LABEL DISTRIBUTION ===\n")
        f.write("Train:\n")
        for r in sorted(train_dist, key=lambda x: -x["label"]):
            f.write(
                f"  label={r['label']}: {r['count']} "
                f"({r['count']/train_size:.3f})\n"
            )

        f.write("Test:\n")
        for r in sorted(test_dist, key=lambda x: -x["label"]):
            f.write(
                f"  label={r['label']}: {r['count']} "
                f"({r['count']/test_size:.3f})\n"
            )

        f.write("\n=== CONFUSION MATRIX (label, prediction) ===\n")
        for r in cm:
            f.write(
                f"label={int(r['label'])}, "
                f"pred={int(r['prediction'])} → {r['count']}\n"
            )
spark.stop()
