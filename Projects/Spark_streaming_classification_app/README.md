# Big Data Flight Delay Prediction

A real-time streaming analytics project for predicting flight delays using Docker-based infrastructure.

## Overview

This project simulates and analyzes flight delay patterns using historical data and real-time streaming. The entire setup is containerized and automated with Docker Compose.

## Getting Started

### Initial Setup

1. **Create HDFS structure** (first-time setup only):
```bash
docker compose up -d hdfs-namenode hdfs-datanode
docker exec -it hdfs-namenode bash

# Inside the container
hdfs dfs -mkdir -p /data/nifi/flights
hdfs dfs -mkdir -p /data/nifi/seismic
hdfs dfs -mkdir -p /data/nifi/weather
hdfs dfs -mkdir -p /data/preprocessed
hdfs dfs -chmod -R 777 /data
```

2. **Start all services**:
```bash
docker compose up -d
docker compose ps
```

### Data Preparation

#### Historical Data

1. Run the `create_historic.ipynb` notebook to generate one month of simulated historical data
2. Load the data into HDFS using the NiFi `load_avro` template

#### Real-Time Streaming Data

1. Load the following NiFi templates:
   - `flights_real_time`
   - `seismic_real_time`
   - `weather_real_time`

2. Enable the processors in NiFi
3. Run the `create_file.ipynb` notebook to start the data stream

> **Note:** All NiFi templates are automatically available, and their state persists unless Docker volumes are cleared.

### Model Training

Before running streaming jobs, you must train at least one model. The model name must match its folder name. By default, a standard GBT (Gradient Boosted Trees) model is used.

**Training options:**
```bash
# Train standard models (GBT, Logistic Regression, Random Forest)
docker compose --profile training up spark-training

# Train various GBT configurations
docker compose --profile training-gbt up spark-training-gbt

# View training logs
docker compose logs spark-training
```

### Analytics Dashboard

To display analytics in the UI, run the historical analysis job:
```bash
docker compose --profile historical-analysis up spark-historical-analysis
```

## Web Interfaces

| Component | URL |
|-----------|-----|
| **NiFi** | http://localhost:8080/nifi |
| **Spark Master UI** | http://localhost:8081 |
| **HDFS NameNode UI** | http://localhost:9870 |
| **Kafka UI** | http://localhost:9000 |
| **Streamlit Dashboard** | http://localhost:8501 |

## Useful Commands

### Docker Management
```bash
# View container status
docker compose ps

# Stop containers (preserve volumes and NiFi state)
docker compose down

# Stop containers and remove volumes
docker compose down -v

# Full cleanup (containers, images, volumes)
docker system prune -a --volumes -f

# Cleanup without deleting images
docker system prune -f --volumes
```

### HDFS Operations
```bash
# List HDFS directories
hdfs dfs -ls /data
```

### Kafka Operations
```bash
# Access Kafka container
docker exec -it kafka bash

# List topics
kafka-topics --bootstrap-server kafka:9092 --list

# Describe a topic
kafka-topics --bootstrap-server kafka:9092 --describe --topic flights

# Consume messages from a topic
kafka-console-consumer \
  --bootstrap-server kafka:9092 \
  --topic weather \
  --from-beginning
```

### Spark Monitoring
```bash
# View streaming job logs
docker logs -f spark-kafka-streaming
```

### Cassandra Operations
```bash
# Access Cassandra
docker exec -it cassandra cqlsh

# Query data
USE kafka_stream;
SELECT * FROM flight_weather_seismic LIMIT 10;
```

## Training Environment

For training and experiments without starting the full stack:
```bash
docker compose up -d hdfs-namenode hdfs-datanode spark-master spark-worker
```