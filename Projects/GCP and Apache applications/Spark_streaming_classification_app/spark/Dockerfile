FROM apache/spark:3.5.1

USER root

RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --no-cache-dir numpy pandas && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER spark
