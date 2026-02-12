CREATE KEYSPACE IF NOT EXISTS kafka_stream
    WITH replication = {
        'class': 'SimpleStrategy',
        'replication_factor': 1
        };

CREATE KEYSPACE IF NOT EXISTS analytics
    WITH replication = {
        'class': 'SimpleStrategy',
        'replication_factor': 1
        };

-- DROP TABLE IF EXISTS kafka_stream.flight_weather_seismic;

CREATE TABLE IF NOT EXISTS kafka_stream.flight_weather_seismic
(
    -- ======================
    -- PRIMARY KEY
    -- ======================
    flight_num            int,
    planned_dep_timestamp timestamp,
    timestamp             timestamp,

    -- ======================
    -- FLIGHT
    -- ======================
    carrier               text,
    airline_name          text,
    origin                text,
    dest                  text,
    dep_time              int,
    estimated_delay       double,

    -- ======================
    -- WEATHER
    -- ======================
    temp                  double,
    humidity              double,
    precip                double,
    windspeed             double,
    pressure              double,
    visibility            double,
    conditions            text,

    -- ======================
    -- SEISMIC
    -- ======================
    mag                   double,
    sig                   int,
    distance_from_lax_km  double,

    -- ======================
    -- ML OUTPUT
    -- ======================
    prediction            int,
    probability           double,

    PRIMARY KEY (
        (flight_num),
         planned_dep_timestamp,
         timestamp
        )
) WITH CLUSTERING ORDER BY (
                            planned_dep_timestamp DESC,
                            timestamp DESC
    );


-- batch views --------------------------------

CREATE TABLE IF NOT EXISTS analytics.monthly_aggregations
(
    month_start                  timestamp PRIMARY KEY,

    year                         int,
    month                        int,
    month_name                   text,

    seismic_total_events         int,
    seismic_avg_magnitude        double,
    seismic_max_magnitude        double,
    seismic_min_magnitude        double,
    seismic_std_magnitude        double,
    seismic_median_magnitude     double,
    seismic_avg_depth            double,
    seismic_max_depth            double,
    seismic_events_per_day       double,
    seismic_significant_events   int,

    flights_total                int,
    flights_avg_delay            double,
    flights_max_delay            double,
    flights_min_delay            double,
    flights_std_delay            double,
    flights_median_delay         double,
    flights_cancellation_rate    double,
    flights_cancelled_total      int,
    flights_on_time_rate         double,
    flights_severe_delay_rate    double,
    flights_per_day              double,
    flights_top_carrier          text,
    flights_top_carrier_share    double,
    flights_top_route            text,

    weather_avg_temp             double,
    weather_max_temp             double,
    weather_min_temp             double,
    weather_std_temp             double,
    weather_avg_humidity         double,
    weather_avg_windspeed        double,
    weather_max_windspeed        double,
    weather_avg_pressure         double,
    weather_total_precipitation  double,
    weather_rainy_days           int,
    weather_common_condition     text,
    weather_common_condition_pct double
);

CREATE TABLE IF NOT EXISTS analytics.weekly_aggregations
(
    week_start                  date PRIMARY KEY,

    seismic_total_events        int,
    seismic_avg_magnitude       double,
    seismic_max_magnitude       double,
    seismic_min_magnitude       double,
    seismic_median_magnitude    double,
    seismic_avg_depth           double,
    seismic_events_per_day      double,
    seismic_peak_hour           int,
    seismic_weekday_events      int,
    seismic_weekend_events      int,

    flights_total               int,
    flights_avg_delay           double,
    flights_max_delay           double,
    flights_min_delay           double,
    flights_median_delay        double,
    flights_cancellation_rate   double,
    flights_cancelled_total     int,
    flights_on_time_rate        double,
    flights_severe_delay_rate   double,
    flights_per_day             double,
    flights_peak_hour           int,
    flights_top_carrier         text,
    flights_top_route           text,

    weather_avg_temp            double,
    weather_max_temp            double,
    weather_min_temp            double,
    weather_avg_humidity        double,
    weather_avg_windspeed       double,
    weather_max_windspeed       double,
    weather_avg_pressure        double,
    weather_total_precipitation double,
    weather_rainy_days          int,
    weather_common_condition    text
);