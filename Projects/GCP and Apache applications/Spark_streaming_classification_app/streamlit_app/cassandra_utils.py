from pyparsing import Path
from cassandra.cluster import Cluster
from cassandra.query import dict_factory
import pandas as pd
import streamlit as st

# Configuration
CASSANDRA_HOSTS = ['cassandra']
KEYSPACE_ACTUAL = 'kafka_stream'
KEYSPACE_HISTORIC = 'analytics'
TABLE_ACTUAL = 'actual_data'

# Session management
def _get_session(keyspace):
    """Internal: Get Cassandra session for keyspace"""
    cluster = Cluster(CASSANDRA_HOSTS)
    session = cluster.connect(keyspace)
    session.row_factory = dict_factory
    return session, cluster

# ===================================================
# ACTUAL (STREAMING) DATA
# ===================================================

@st.cache_data(ttl=30)  # Cache for 30 seconds for fresh data
def load_actual_data():
    """Load actual streaming data from kafka_stream.actual_data"""
    session, cluster = _get_session(KEYSPACE_ACTUAL)
    
    try:
        query = f"SELECT * FROM {TABLE_ACTUAL}"
            
        rows = session.execute(query)
        data = [row._asdict() for row in rows]
        df = pd.DataFrame(data)
        
        # Sort by timestamp if available
        for col in ["simulation_timestamp", "event_time", "timestamp", "minute"]:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df = df.sort_values(col, ascending=False)
                    break
                except:
                    continue
        
        return df
    finally:
        cluster.shutdown()
        
# ===================================================
# TEST FUNCTIONS
# ===================================================

@st.cache_data
def load_monthly_data():
    data_folder = Path(__file__).parent / "data"
    file_path = data_folder / "monthly_aggregations.csv"
    try:
        df = pd.read_csv(file_path)
        df['month_start'] = pd.to_datetime(df['month_start'])
        df = df.sort_values('month_start', ascending=True)
        return df
    except FileNotFoundError:
        st.error(f"Monthly data file not found at: {file_path}")
        return pd.DataFrame()

@st.cache_data
def load_weekly_data():
    data_folder = Path(__file__).parent / "data"
    file_path = data_folder / "weekly_aggregations.csv"
    try:
        df = pd.read_csv(file_path)
        df['week_start'] = pd.to_datetime(df['week_start'])
        df = df.sort_values('week_start', ascending=True)
        return df
    except FileNotFoundError:
        st.error(f"Weekly data file not found at: {file_path}")
        return pd.DataFrame()

# ===================================================
# HISTORICAL DATA
# ===================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes (historical doesn't change often)
def load_monthly_aggregations(limit=100):
    """Load monthly aggregations from analytics.monthly_aggregations"""
    session, cluster = _get_session(KEYSPACE_HISTORIC)
    
    try:
        query = "SELECT * FROM monthly_aggregations ORDER BY year DESC, month DESC"
        if limit:
            query += f" LIMIT {limit}"
            
        rows = session.execute(query)
        df = pd.DataFrame(list(rows))
        
        if 'month_start' in df.columns:
            df['month_start'] = pd.to_datetime(df['month_start'], errors='coerce')
            
        return df
    finally:
        cluster.shutdown()

@st.cache_data(ttl=300)
def load_weekly_aggregations(limit=100):
    """Load weekly aggregations from analytics.weekly_aggregations"""
    session, cluster = _get_session(KEYSPACE_HISTORIC)
    
    try:
        query = "SELECT * FROM weekly_aggregations ORDER BY week_start DESC"
        if limit:
            query += f" LIMIT {limit}"
            
        rows = session.execute(query)
        df = pd.DataFrame(list(rows))
        
        if 'week_start' in df.columns:
            df['week_start'] = pd.to_datetime(df['week_start'], errors='coerce')
            
        return df
    finally:
        cluster.shutdown()

# ===================================================
# QUICK ACCESS FUNCTIONS
# ===================================================

def get_latest_actual():
    """Get latest actual data entry"""
    df = load_actual_data(1)
    return df.iloc[0] if not df.empty else None

def get_latest_monthly():
    """Get latest monthly aggregation"""
    df = load_monthly_aggregations(1)
    return df.iloc[0] if not df.empty else None

def get_latest_weekly():
    """Get latest weekly aggregation"""
    df = load_weekly_aggregations(1)
    return df.iloc[0] if not df.empty else None

def get_summary_stats():
    """Get quick summary statistics for dashboard"""
    return {
        'actual_count': len(load_actual_data(100)),
        'latest_actual': get_latest_actual(),
        'latest_monthly': get_latest_monthly(),
        'latest_weekly': get_latest_weekly()
    }