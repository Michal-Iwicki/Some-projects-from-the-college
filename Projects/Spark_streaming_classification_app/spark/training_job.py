from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, to_timestamp, rand, input_file_name
)
from pyspark.sql.types import NumericType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os
from pyspark.sql.functions import col
from pyspark.sql.window import Window
from pyspark.sql.functions import percent_rank
from pyspark.sql.functions import count
from pyspark.sql import functions as F


# ===================================================
# Spark session
# ===================================================
spark = (
    SparkSession.builder
    .appName("FlightDelayBinaryClassification-Downsampled")
    .getOrCreate()
)

# ===================================================
# Read Avro (MERGE SCHEMA!)
# ===================================================
path = "hdfs://hdfs-namenode:8020/data/preprocessed/"

df = (
    spark.read
    .format("avro")
    .option("mergeSchema", "false")  
    .load(path)
)

# ===================================================
# Add source file column (CRUCIAL)
# ===================================================
df = df.withColumn("_source_file", input_file_name())

# ===================================================
# Timestamp normalization
# ===================================================

df = (
    df
    .withColumn("Planned_Dep_Timestamp", to_timestamp(col("Planned_Dep_Timestamp")))
    .withColumn("Actual_Dep_Timestamp", to_timestamp(col("Actual_Dep_Timestamp")))
    .withColumn("timestamp", to_timestamp(col("timestamp")))
)

# ===================================================
# Fill NULLs (NUMERIC ONLY)
# ===================================================
numeric_cols = [
    f.name for f in df.schema.fields
    if isinstance(f.dataType, NumericType)
]

df = df.fillna(0, subset=numeric_cols)
# ===================================================
# Label
# ===================================================
df = df.withColumn(
    "label",
    when(col("Delay") > 0, 1).otherwise(0)
)

# ===================================================
# 🔑 5% RANDOM SAMPLE FROM EACH FILE
# ===================================================
df = (
    df
    .withColumn("_file", input_file_name())
    .withColumn("_r", rand(seed=42))
    .where(col("_r") <= 0.05)
    .drop("_r", "_file")
)


# ===================================================
# Drop helper column
# ===================================================
df = df.drop("_source_file")

# ===================================================
# Feature selection
# ===================================================
# --------------------------------------------------
# NORMALIZACJA KOLUMN (NAJPIERW)
# --------------------------------------------------
df = df.select([
    F.col(c).alias(c.lower().replace(" ", "_"))
    for c in df.columns
])

# --------------------------------------------------
# AIRLINES — POPRAWIONE
# --------------------------------------------------
airlines = [
    "alaska_airlines",
    "allegiant_air",
    "american_airlines",
    "delta",
    "envoy_air",
    "frontier_airlines",
    "hawaiian_airlines",
    "horizon_air",
    "jet_blue",              
    "skywest_airlines",
    "southwest",
    "spirit_airlines",
    "united_airlines"
]

# --------------------------------------------------
# FEATURE COLS — POPRAWIONE
# --------------------------------------------------
feature_cols = [
    *[f"airline_name_{a}" for a in airlines],
    "airline_name_other",

    "time_till_dep_sec",

    "mag", "sig", "nst", "dmin", "rms", "gap", "longitude", "latitude",
    "distance_from_lax_km",
    "type_earthquake", "type_other", "time_since_last_event_sec",

    "temp", "humidity", "precip", "windgust", "windspeed", "winddir",
    "pressure", "cloudcover", "visibility",

    "conditions_clear",
    "conditions_overcast",
    "conditions_partially_cloudy",
    "conditions_other"
]

# --------------------------------------------------
# SANITY CHECK (OBOWIĄZKOWY)
# --------------------------------------------------
missing = set(feature_cols + ["label"]) - set(df.columns)
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

# --------------------------------------------------
# FINAL SELECT
# --------------------------------------------------
data = df.select(*feature_cols, "label")


# ===================================================
# Repartition AFTER sampling
# ===================================================
data = data.repartition(4)

# ===================================================
# Train / test split
# ===================================================
# ---------------------------------------------------
# Chronological train / test split (80% / 20%)
# ---------------------------------------------------

data = df.select(feature_cols + ["label", "timestamp"])

# percent_rank po czasie
w = Window.orderBy(col("timestamp"))
data = data.withColumn("time_rank", percent_rank().over(w))

train_df = data.filter(col("time_rank") <= 0.8).drop("time_rank", "timestamp")
test_df  = data.filter(col("time_rank") > 0.8).drop("time_rank", "timestamp")


# ===================================================
# Vector assembler
# ===================================================
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

# ===================================================
# Models
# ===================================================
models = {
    "LogisticRegression": LogisticRegression(maxIter=50, regParam=0.01),
    "RandomForest": RandomForestClassifier(numTrees=100, maxDepth=10),
    "GBT": GBTClassifier(maxIter=50, maxDepth=5)
}

evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    metricName="areaUnderROC"
)
# ---------------------------------------------------
# Output directories (LOCAL FS)
# ---------------------------------------------------
report_dir = "/reports"
model_dir = "/models"

# Te katalogi są mountami Dockera → tworzymy lokalnie
os.makedirs(report_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# ---------------------------------------------------
# Training loop
# ---------------------------------------------------
for name, clf in models.items():
    print(f"\n[INFO] Training {name}")

    pipeline = Pipeline(stages=[assembler, clf])
    model = pipeline.fit(train_df)

    preds = model.transform(test_df)
    auc = evaluator.evaluate(preds)

    # ---------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------
    cm = (
        preds
        .groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
        .collect()
    )

    # ---------------------------------------------------
    # Label proportions
    # ---------------------------------------------------
    train_dist = (
        train_df.groupBy("label")
        .count()
        .withColumnRenamed("count", "train_count")
        .collect()
    )

    test_dist = (
        test_df.groupBy("label")
        .count()
        .withColumnRenamed("count", "test_count")
        .collect()
    )

    train_size = train_df.count()
    test_size = test_df.count()

    # ---------------------------------------------------
    # Save model
    # ---------------------------------------------------
    model_path = f"file:{model_dir}/{name}"
    model.write().overwrite().save(model_path)

    # ---------------------------------------------------
    # Report
    # ---------------------------------------------------
    report_path = f"{report_dir}/{name}_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"AUC: {auc:.4f}\n\n")

        f.write("=== DATASET SIZES ===\n")
        f.write(f"Train size: {train_size}\n")
        f.write(f"Test size: {test_size}\n\n")

        f.write("=== LABEL DISTRIBUTION ===\n")
        f.write("Train:\n")
        for r in train_dist:
            f.write(f"  label={r['label']}: {r['train_count']} "
                    f"({r['train_count']/train_size:.3f})\n")

        f.write("Test:\n")
        for r in test_dist:
            f.write(f"  label={r['label']}: {r['test_count']} "
                    f"({r['test_count']/test_size:.3f})\n")

        f.write("\n=== CONFUSION MATRIX (label, prediction) ===\n")
        for r in cm:
            f.write(
                f"label={int(r['label'])}, "
                f"pred={int(r['prediction'])} → {r['count']}\n"
            )

    print(f"[RESULT] {name} AUC = {auc:.4f}")
    print(f"[INFO] Model saved to {model_path}")
    print(f"[INFO] Report saved to {report_path}")

print("\n=== TRAINING FINISHED SUCCESSFULLY ===")
spark.stop()
