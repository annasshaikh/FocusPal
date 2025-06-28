import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import datetime

# --- Configuration ---
DATABASE_FILE = 'presence.db'

# --- Helper Functions ---

def format_duration(seconds):
    """Formats seconds into Hh M:S string."""
    if seconds is None:
        seconds = 0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h {m:02d}m {s:02d}s"

def get_data(start_date, end_date):
    """Fetches session data from the database within a date range."""
    conn = None
    df = pd.DataFrame() # Return empty DataFrame by default
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        # SQL query to select data within the specified date range (inclusive)
        # Use DATE() function to compare only the date part of the TEXT timestamp
        query = """
        SELECT start_time, end_time, duration
        FROM presence
        WHERE DATE(start_time) BETWEEN ? AND ?
        ORDER BY start_time
        """
        # Convert date objects to string format expected by SQLite DATE()
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        df = pd.read_sql_query(query, conn, params=(start_date_str, end_date_str))

        # Convert time columns to datetime objects for plotting and analysis
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])

    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
    finally:
        if conn:
            conn.close()
    return df

def get_total_time_for_today():
    """Calculates total presence time for the current day."""
    conn = None
    total_duration = 0
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        today_str = datetime.date.today().strftime('%Y-%m-%d')
        query = """
        SELECT SUM(duration)
        FROM presence
        WHERE DATE(start_time) = ?
        """
        cursor = conn.cursor()
        cursor.execute(query, (today_str,))
        result = cursor.fetchone()
        if result and result[0] is not None:
            total_duration = result[0]
    except sqlite3.Error as e:
        st.error(f"Database error fetching total time today: {e}")
    except Exception as e:
        st.error(f"An error occurred fetching total time today: {e}")
    finally:
        if conn:
            conn.close()
    return total_duration


# --- Streamlit App ---

st.set_page_config(layout="wide") # Use wide layout

st.title("🧑‍💻 Person Presence Analytics Dashboard")

# --- Display Total Time Today (Always based on current date) ---
st.header("Today's Summary")
total_today_seconds = get_total_time_for_today()
st.metric("Total Time Today", format_duration(total_today_seconds))

st.markdown("---") # Separator

# --- Date Filtering for Historical Data ---
st.header("Historical Data Analytics")
st.sidebar.header("Filter Data")

# Default date range: Last 7 days including today
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=6)

start_date_filter = st.sidebar.date_input("Select Start Date", start_date)
end_date_filter = st.sidebar.date_input("Select End Date", end_date)

# Ensure start date is not after end date
if start_date_filter > end_date_filter:
    st.sidebar.error("Error: Start date must be before or on the end date.")
    data_filtered = pd.DataFrame() # Load empty data if date range is invalid
else:
    # Fetch filtered data
    data_filtered = get_data(start_date_filter, end_date_filter)


# --- Display Filtered Data Metrics ---
if data_filtered.empty:
    st.warning("No presence data available for the selected date range.")
else:
    st.subheader(f"Data from {start_date_filter.strftime('%Y-%m-%d')} to {end_date_filter.strftime('%Y-%m-%d')}")

    # Calculate metrics for the filtered data
    total_sessions = len(data_filtered)
    total_duration_filtered = data_filtered['duration'].sum()
    average_session_duration = data_filtered['duration'].mean() if total_sessions > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sessions", total_sessions)
    col2.metric("Total Time in Range", format_duration(total_duration_filtered))
    col3.metric("Average Session Duration", format_duration(average_session_duration))

    st.markdown("---") # Separator

    # --- Visualizations ---
    st.header("Visualizations")

    # 1. Hourly Activity Bar Chart
    st.subheader("Hourly Activity (Total Duration per Hour)")
    # Extract hour from start_time
    data_filtered['start_hour'] = data_filtered['start_time'].dt.hour
    hourly_activity = data_filtered.groupby('start_hour')['duration'].sum().reset_index()
    hourly_activity['start_hour_str'] = hourly_activity['start_hour'].astype(str) + ":00" # For better x-axis labels
    # Ensure all hours (0-23) are present, even if no activity
    all_hours = pd.DataFrame({'start_hour': range(24)})
    hourly_activity = pd.merge(all_hours, hourly_activity, on='start_hour', how='left').fillna(0)
    hourly_activity['start_hour_str'] = hourly_activity['start_hour'].astype(str) + ":00" # Recreate after merge
    hourly_activity = hourly_activity.sort_values('start_hour') # Ensure correct order

    fig_hourly = px.bar(
        hourly_activity,
        x='start_hour_str',
        y='duration',
        title='Total Presence Duration per Hour of Day',
        labels={'start_hour_str': 'Hour of Day', 'duration': 'Total Duration (seconds)'},
        hover_data={'duration': ':.2f'} # Show duration with 2 decimal places on hover
    )
    fig_hourly.update_layout(xaxis={'categoryorder':'array', 'categoryarray':hourly_activity['start_hour_str']})
    fig_hourly.update_yaxes(title_text='Total Duration (' + format_duration(hourly_activity['duration'].max() or 0) + ')') # Custom Y-axis title

    st.plotly_chart(fig_hourly, use_container_width=True)


    # 2. Session Durations Over Time (Line/Scatter)
    st.subheader("Session Durations Over Time")
    # Sort by start time for the line graph
    data_sorted_by_time = data_filtered.sort_values('start_time')

    fig_sessions_over_time = px.line(
        data_sorted_by_time,
        x='start_time',
        y='duration',
        title='Duration of Each Presence Session',
        labels={'start_time': 'Session Start Time', 'duration': 'Duration (seconds)'},
        hover_data={'duration': ':.2f', 'start_time': True} # Show duration with 2 decimal places
    )
    fig_sessions_over_time.update_yaxes(title_text='Duration (' + format_duration(data_sorted_by_time['duration'].max() or 0) + ')') # Custom Y-axis title

    st.plotly_chart(fig_sessions_over_time, use_container_width=True)

    # 3. Daily Total Activity Bar Chart (Replacing Heatmap for simplicity and clarity)
    st.subheader("Daily Total Presence")
    # Extract date from start_time
    data_filtered['start_date'] = data_filtered['start_time'].dt.date
    daily_activity = data_filtered.groupby('start_date')['duration'].sum().reset_index()
    daily_activity['start_date'] = pd.to_datetime(daily_activity['start_date']) # Convert back to datetime for plotly

    fig_daily = px.bar(
        daily_activity,
        x='start_date',
        y='duration',
        title='Total Presence Duration per Day',
        labels={'start_date': 'Date', 'duration': 'Total Duration (seconds)'},
        hover_data={'duration': ':.2f', 'start_date': True}
    )
    fig_daily.update_layout(xaxis_title='Date', yaxis_title='Total Duration (' + format_duration(daily_activity['duration'].max() or 0) + ')') # Custom Y-axis title

    st.plotly_chart(fig_daily, use_container_width=True)


    st.markdown("---") # Separator

    # --- Session Table ---
    st.header("Session Data Table")
    # Select relevant columns and format duration for display
    display_df = data_filtered[['start_time', 'end_time', 'duration']].copy()
    display_df['start_time'] = display_df['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df['end_time'] = display_df['end_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df['duration_formatted'] = display_df['duration'].apply(format_duration)

    st.dataframe(display_df[['start_time', 'end_time', 'duration_formatted']].rename(columns={
        'start_time': 'Start Time',
        'end_time': 'End Time',
        'duration_formatted': 'Duration'
    }), use_container_width=True)


    # --- CSV Export ---
    st.subheader("Export Data")
    # Create a CSV string from the filtered DataFrame
    csv_data = data_filtered[['start_time', 'end_time', 'duration']].to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Session Data as CSV",
        data=csv_data,
        file_name=f"presence_sessions_{start_date_filter.strftime('%Y%m%d')}_to_{end_date_filter.strftime('%Y%m%d')}.csv",
        mime='text/csv',
    )

# --- How to Run Info ---
st.sidebar.markdown("---")
st.sidebar.info("""
**How to Run:**

1.  Ensure you have `streamlit`, `pandas`, and `plotly` installed (`pip install streamlit pandas plotly`).
2.  Run this file from your terminal:
    ```bash
    streamlit run analytics_dashboard.py
    ```
3.  Make sure `presence_log.db` (generated by `tracker_gui.py`) is in the same directory.
""")