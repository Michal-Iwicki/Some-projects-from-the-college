#!/bin/bash
set -euo pipefail

BOOTSTRAP_SERVER="kafka:9092"

echo ">>> Waiting for Kafka to be ready..."

for i in {1..60}; do
  if kafka-topics --bootstrap-server "$BOOTSTRAP_SERVER" --list >/dev/null 2>&1; then
    echo ">>> Kafka is ready!"
    break
  fi
  echo ">>> Kafka not ready yet ($i/60)..."
  sleep 1
done

echo ">>> Creating topics..."

TOPICS=(flights weather seismic)

for topic in "${TOPICS[@]}"; do
  kafka-topics \
    --bootstrap-server "$BOOTSTRAP_SERVER" \
    --create \
    --if-not-exists \
    --topic "$topic" \
    --partitions 3 \
    --replication-factor 1 \
    --config retention.ms=604800000
done

echo "✅ Topics created successfully"
