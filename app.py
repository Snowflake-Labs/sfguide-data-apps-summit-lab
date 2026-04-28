# Import required libraries
import streamlit as st
from snowflake.snowpark.context import get_active_session
import pandas as pd
import altair as alt

# Get the current session
session = get_active_session()

# Simple page setup
st.title("🌤️ California Weather Prediction")
st.write("Predict **monthly average temperatures** for Beverly Hills in 2025")

# Input section
col1, col2 = st.columns(2)

with col1:
    zip_options = {
        '90210': 'Beverly Hills (90210)'
    }
    zip_code = st.selectbox("Select Location", list(zip_options.keys()), format_func=lambda x: zip_options[x])

with col2:
    months = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    month = st.selectbox("Month for 2025", list(months.keys()), format_func=lambda x: months[x], index=6)

if st.button("Predict Average Temperature", type="primary"):
    prediction_query = f"SELECT predict_temperature_udf('{zip_code}', {month}) as temp"
    pred_result = session.sql(prediction_query).collect()
    temp_celsius = pred_result[0]['TEMP']
    temp_fahrenheit = (temp_celsius * 9/5) + 32

    historical_query = f"""
    SELECT 
        EXTRACT(MONTH FROM ts.date) as month,
        AVG(ts.value) as avg_temp
    FROM SNOWFLAKE_PUBLIC_DATA_FREE.PUBLIC_DATA_FREE.NOAA_WEATHER_METRICS_TIMESERIES ts
    JOIN SNOWFLAKE_PUBLIC_DATA_FREE.PUBLIC_DATA_FREE.NOAA_WEATHER_STATION_INDEX idx 
        ON ts.noaa_weather_station_id = idx.noaa_weather_station_id
    WHERE ts.variable_name LIKE 'Average daily temperature%'
        AND idx.zip_name = '{zip_code}'
        AND ts.date >= '2024-01-01'
        AND ts.date < '2025-01-01'
    GROUP BY EXTRACT(MONTH FROM ts.date)
    ORDER BY month
    """

    historical_df = session.sql(historical_query).to_pandas()

    if not historical_df.empty:
        historical_df['temp_f'] = (historical_df['AVG_TEMP'] * 9/5) + 32

        if month in historical_df['MONTH'].values:
            historical_temp = historical_df[historical_df['MONTH'] == month]['temp_f'].iloc[0]
            difference = temp_fahrenheit - historical_temp

            col_hist, col_pred, col_diff = st.columns(3)
            with col_hist:
                st.metric("2024 Actual", f"{historical_temp:.1f}°F")
            with col_pred:
                st.metric("2025 Prediction", f"{temp_fahrenheit:.1f}°F")
            with col_diff:
                st.metric("Difference", f"{difference:+.1f}°F")

        month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        historical_df['month_name'] = historical_df['MONTH'].map(lambda x: months[x][:3])

        pred_point = pd.DataFrame({'month_name': [months[month][:3]], 'temp_f': [temp_fahrenheit]})

        line = alt.Chart(historical_df).mark_line(point=True).encode(
            x=alt.X('month_name:N', sort=month_order, title='Month'),
            y=alt.Y('temp_f:Q', title='Temperature (°F)'),
            tooltip=['month_name', 'temp_f']
        )
        point = alt.Chart(pred_point).mark_point(size=150, color='red').encode(
            x=alt.X('month_name:N', sort=month_order),
            y='temp_f:Q',
            tooltip=['month_name', 'temp_f']
        )
        st.altair_chart((line + point).properties(
            title=f"2024 vs 2025 Prediction - {zip_options[zip_code]}"
        ), use_container_width=True)

        annual_query = f"""
        SELECT m.month, predict_temperature_udf('{zip_code}', m.month) as temp
        FROM (SELECT ROW_NUMBER() OVER (ORDER BY SEQ4()) as month FROM TABLE(GENERATOR(ROWCOUNT => 12))) m
        """
        annual_result = session.sql(annual_query).to_pandas()
        annual_result['Temp_F'] = (annual_result['TEMP'] * 9/5) + 32
        annual_result['Temp_C'] = annual_result['TEMP']
        annual_result['Month'] = annual_result['MONTH'].map(lambda x: months[int(x)][:3])
        pred_df = annual_result[['Month', 'Temp_F', 'Temp_C']]

        bar = alt.Chart(pred_df).mark_bar().encode(
            x=alt.X('Month:N', sort=month_order, title='Month'),
            y=alt.Y('Temp_F:Q', title='Temperature (°F)'),
            color=alt.Color('Temp_F:Q', scale=alt.Scale(scheme='redyellowblue', reverse=True)),
            tooltip=['Month', 'Temp_F', 'Temp_C']
        ).properties(title=f"2025 Monthly Predictions - {zip_options[zip_code]}")
        st.altair_chart(bar, use_container_width=True)

        display_df = pred_df.copy()
        display_df['Temperature'] = display_df.apply(lambda row: f"{row['Temp_F']:.1f}°F ({row['Temp_C']:.1f}°C)", axis=1)
        st.dataframe(display_df[['Month', 'Temperature']], use_container_width=True, hide_index=True)
    else:
        st.warning("No 2024 data available for comparison, but prediction is still valid!")

st.write("---")
st.write("**What the app predicts:** Monthly average temperature")
st.write("**Training:** Location-specific models using 20+ years of data (2000-2024)")
st.write("**Features:** Geographic location, elevation, distance from coast")
st.write("**Data:** NOAA weather stations via Snowflake Marketplace")
