# Import required libraries
import streamlit as st
from snowflake.snowpark.context import get_active_session
import pandas as pd
import plotly.express as px

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
    # Make prediction using UDF
    prediction_query = f"SELECT predict_temperature_udf('{zip_code}', {month}) as temp"
    pred_result = session.sql(prediction_query).collect()
    temp_celsius = pred_result[0]['TEMP']
    temp_fahrenheit = (temp_celsius * 9/5) + 32
    
    # Get 2024 data for comparison
    historical_query = f"""
    SELECT 
        EXTRACT(MONTH FROM ts.date) as month,
        AVG(ts.value) as avg_temp
    FROM SNOWFLAKE_PUBLIC_DATA_FREE.PUBLIC_DATA_FREE.NOAA_WEATHER_METRICS_TIMESERIES ts
    JOIN SNOWFLAKE_PUBLIC_DATA_FREE.PUBLIC_DATA_FREE.NOAA_WEATHER_STATION_INDEX idx 
        ON ts.noaa_weather_station_id = idx.noaa_weather_station_id
    WHERE ts.variable_name = 'Average Temperature'
        AND idx.zip_name = '{zip_code}'
        AND ts.date >= '2024-01-01'
        AND ts.date < '2025-01-01'
    GROUP BY EXTRACT(MONTH FROM ts.date)
    ORDER BY month
    """
    
    historical_df = session.sql(historical_query).to_pandas()
    
    if not historical_df.empty:
        # Convert to Fahrenheit
        historical_df['temp_f'] = (historical_df['AVG_TEMP'] * 9/5) + 32
        
        # Show comparison if 2024 data exists for this month
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
        
        # Chart showing 2024 vs 2025 prediction
        historical_df['month_name'] = historical_df['MONTH'].map(lambda x: months[x][:3])
        
        fig = px.line(
            historical_df, 
            x='month_name', 
            y='temp_f',
            title=f"2024 vs 2025 Prediction - {zip_options[zip_code]}",
            markers=True
        )
        
        # Add prediction point
        fig.add_scatter(
            x=[months[month][:3]], 
            y=[temp_fahrenheit],
            mode='markers',
            marker=dict(size=12, color='red'),
            name='2025 Prediction'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        
        # Get predictions for all 12 months
        annual_predictions = []
        for m in range(1, 13):
            pred_query = f"SELECT predict_temperature_udf('{zip_code}', {m}) as temp"
            result = session.sql(pred_query).collect()
            temp_c = result[0]['TEMP']
            temp_f = (temp_c * 9/5) + 32
            annual_predictions.append({
                'Month': months[m][:3],
                'Temp_F': temp_f,
                'Temp_C': temp_c
            })
        
        pred_df = pd.DataFrame(annual_predictions)
        
        # Create annual chart
        annual_fig = px.bar(
            pred_df,
            x='Month',
            y='Temp_F', 
            title=f"2025 Monthly Predictions - {zip_options[zip_code]}",
            color='Temp_F',
            color_continuous_scale='RdYlBu_r'
        )
        st.plotly_chart(annual_fig, use_container_width=True)
        
        # Show data table
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
