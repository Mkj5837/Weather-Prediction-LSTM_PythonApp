import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Weather LSTM Predictor",
    page_icon="🌡️",
    layout="wide"
)

st.title("Weather Prediction with LSTM")
st.markdown(
    "This app predicts the next day's maximum temperature using the last 30 days of Basel (Switzerland) temperature data."
)

@st.cache_resource
def load_artifacts():
    model = load_model('weather_lstm_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# -------------------- Sidebar with user options first, then model details --------------------
input_method = st.sidebar.radio(
    "Input method",
    ["Manual Entry", "Upload CSV", "Sample Data"]
)

st.sidebar.markdown("---")
st.sidebar.header("Model Details")
st.sidebar.markdown(
    "- Architecture: 2-layer LSTM + dropout\n"
    "- Window: last 30 days\n"
    "- Output: next day max temperature\n"
    "- Evaluation metrics (test set):"
)
st.sidebar.write("MAE: 2.38 °C")
st.sidebar.write("RMSE: 2.98 °C")
st.sidebar.write("R²: 0.887")
st.sidebar.write("Accuracy (±2°C): 48.6%")

def predict_next_day(values):
    """Predict next day's max temperature from 30 daily values."""
    if len(values) != 30:
        return None, f"Expected 30 values, got {len(values)}."
    
    # Basic validation: check for non-numeric or out-of-range
    try:
        arr = np.array(values, dtype=float)
    except ValueError:
        return None, "All values must be numbers."
    
    if np.any(arr < -30) or np.any(arr > 60):
        st.warning("Some temperatures are outside typical range (-30°C to 60°C). Prediction may be unreliable.")
    
    # Reshape, scale, predict
    scaled = scaler.transform(arr.reshape(-1, 1))
    X_input = scaled.reshape(1, 30, 1)
    pred_scaled = model.predict(X_input, verbose=0)[0][0]
    pred = scaler.inverse_transform([[pred_scaled]])[0][0]
    return pred, None

def plot_temperature_data(days, historical, prediction):
    """
    Create a matplotlib plot of temperature data with prediction.
    days: list of 31 day numbers (1..31)
    historical: list of 30 historical temperatures
    prediction: next day temperature (float)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot last 30 days (days 1..30)
    ax.plot(days[:30], historical, 'b-o', linewidth=2, markersize=6, label='Last 30 days')
    
    # Plot prediction (day 31)
    ax.plot(days[30], prediction, 'ro', markersize=12, label='Prediction')
    
    # dashed line connecting last historical point to prediction
    ax.plot([days[29], days[30]], [historical[-1], prediction], 'r--', alpha=0.5)
    
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Last 30 Days + Prediction', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

# -------------------- Manual entry --------------------
if input_method == "Manual Entry":
    st.subheader("Enter the last 30 days of max temperatures")
    cols = st.columns(5)
    manual_values = []
    for i in range(30):
        idx = i % 5
        with cols[idx]:
            manual_values.append(
                st.number_input(f"Day {i+1}", value=20.0, step=0.5, format="%.1f", key=f"manual_{i}")
            )
    if st.button("Predict"):
        pred, error = predict_next_day(manual_values)
        if error:
            st.error(error)
        else:
            st.success(f"Predicted next day max temperature: {pred:.1f} °C")
            days = list(range(1, 32))
            fig = plot_temperature_data(days, manual_values, pred)
            st.pyplot(fig)
            plt.close(fig)

# -------------------- CSV upload --------------------
elif input_method == "Upload CSV":
    st.subheader("Upload a CSV file with temperature data")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("First few rows of uploaded file:")
        st.write(df.head())
        
        # Suggest temperature column
        temp_cols = [c for c in df.columns if 'temp' in c.lower() or 'temperature' in c.lower()]
        if temp_cols:
            col = st.selectbox("Select temperature column", temp_cols)
        else:
            col = st.selectbox("Select temperature column", df.columns)
        
        if len(df) >= 30:
            # Extract last 30 values, ensure numeric
            raw_vals = df[col].iloc[-30:].values
            try:
                values = raw_vals.astype(float).tolist()
            except ValueError:
                st.error(f"Column '{col}' contains non-numeric values. Please clean your data.")
                st.stop()
            
            if st.button("Predict from CSV"):
                pred, error = predict_next_day(values)
                if error:
                    st.error(error)
                else:
                    st.success(f"Predicted next day max temperature: {pred:.1f} °C")
                    days = list(range(1, 32))
                    fig = plot_temperature_data(days, values, pred)
                    st.pyplot(fig)
                    plt.close(fig)
        else:
            st.warning(f'File has {len(df)} rows. Need at least 30 rows.')

# -------------------- Sample data --------------------
else:
    st.subheader('Sample data prediction')
    # Use actual sample from the dataset (around 2000-01-01 to 2000-01-30)
    sample_values = [
        3.9, 4.8, 4.8, 7.5, 8.6, 6.7, 5.5, 4.2, 3.1, 2.8,
        3.5, 4.0, 5.0, 6.2, 7.1, 6.8, 5.5, 4.4, 3.9, 4.2,
        5.0, 6.0, 5.5, 4.8, 4.0, 3.5, 2.9, 3.8, 5.2, 6.0
    ]
    st.write("Sample 30-day sequence (actual Basel data, January 2000):")
    st.write(sample_values)
    if st.button('Predict sample data'):
        pred, error = predict_next_day(sample_values)
        if error:
            st.error(error)
        else:
            st.success(f"Predicted next day max temperature: {pred:.1f} °C")
            days = list(range(1, 32))
            fig = plot_temperature_data(days, sample_values, pred)
            st.pyplot(fig)
            plt.close(fig)

st.markdown('---')
st.markdown(
    "**Guide:** Enter 30 consecutive daily maximum temperatures (°C), upload a CSV with a temperature column, "
    "or use the sample data. The model uses the last 30 days to predict the next day's maximum temperature."
)