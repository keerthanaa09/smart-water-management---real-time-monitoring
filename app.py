import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Smart Water Management",
    page_icon="💧",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
    }

    .stMetric {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 12px;
    }

    h1, h2, h3 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- TITLE ----------------
st.title("💧 AI Smart Water Management System")
st.write("Real-time monitoring, prediction, analytics and leak detection")

# ---------------- LOAD DATA ----------------
data = pd.read_csv("water_usage.csv")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Dashboard Menu")

option = st.sidebar.radio(
    "Select Module",
    [
        "Overview",
        "Prediction",
        "Leak Detection",
        "Analytics"
    ]
)

# ---------------- KPIs ----------------
total_usage = data['usage'].sum()
avg_usage = data['usage'].mean()
max_usage = data['usage'].max()
min_usage = data['usage'].min()

# ---------------- MACHINE LEARNING ----------------
X = data[['day']]
y = data['usage']

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)
future_prediction = model.predict([[8]])

mae = mean_absolute_error(y, predictions)

# ==================================================
# OVERVIEW PAGE
# ==================================================
if option == "Overview":

    st.subheader("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Usage", f"{total_usage} L")
    col2.metric("Average Usage", f"{avg_usage:.2f} L")
    col3.metric("Maximum Usage", f"{max_usage} L")
    col4.metric("Minimum Usage", f"{min_usage} L")

    st.divider()

    st.subheader("Water Usage Dataset")
    st.dataframe(data, use_container_width=True)

    st.subheader("Daily Water Usage Trend")

    fig = px.line(
        data,
        x='day',
        y='usage',
        markers=True,
        title='Daily Water Usage'
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# PREDICTION PAGE
# ==================================================
if option == "Prediction":

    st.subheader("AI Water Usage Prediction")

    st.success(
        f"Predicted Water Usage for Day 8: {future_prediction[0]:.2f} Liters"
    )

    st.info(f"Model Accuracy Error (MAE): {mae:.2f}")

    prediction_df = pd.DataFrame({
        'Day': data['day'],
        'Actual Usage': y,
        'Predicted Usage': predictions
    })

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=prediction_df['Day'],
        y=prediction_df['Actual Usage'],
        mode='lines+markers',
        name='Actual Usage'
    ))

    fig2.add_trace(go.Scatter(
        x=prediction_df['Day'],
        y=prediction_df['Predicted Usage'],
        mode='lines+markers',
        name='Predicted Usage'
    ))

    fig2.update_layout(
        title='Actual vs Predicted Water Usage',
        xaxis_title='Day',
        yaxis_title='Usage (Liters)'
    )

    st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# LEAK DETECTION PAGE
# ==================================================
if option == "Leak Detection":

    st.subheader("Leak Detection Analysis")

    current_drop = 30

    threshold = st.slider(
        "Leak Detection Threshold",
        10,
        50,
        20
    )

    st.write(f"Current Water Drop: {current_drop} Liters")

    if current_drop > threshold:
        st.error("⚠️ Leakage Detected!")
    else:
        st.success("✅ No Leakage Detected")

    leak_data = pd.DataFrame({
        'Time': ['8 AM', '10 AM', '12 PM', '2 PM', '4 PM'],
        'Water Drop': [5, 7, 10, 30, 8]
    })

    fig3 = px.bar(
        leak_data,
        x='Time',
        y='Water Drop',
        title='Water Drop Monitoring'
    )

    st.plotly_chart(fig3, use_container_width=True)

# ==================================================
# ANALYTICS PAGE
# ==================================================
if option == "Analytics":

    st.subheader("Advanced Analytics")

    col1, col2 = st.columns(2)

    with col1:

        pie = px.pie(
            values=data['usage'],
            names=data['day'],
            title='Water Consumption Share'
        )

        st.plotly_chart(pie, use_container_width=True)

    with col2:

        histogram = px.histogram(
            data,
            x='usage',
            nbins=5,
            title='Usage Distribution'
        )

        st.plotly_chart(histogram, use_container_width=True)

    st.subheader("Smart AI Insights")

    if future_prediction[0] > avg_usage:
        st.warning(
            "Predicted usage is higher than average. Water conservation recommended."
        )
    else:
        st.success(
            "Water usage is within normal range."
        )

    st.info(
        "Peak water consumption observed during later days of the week."
    )

# ---------------- FOOTER ----------------
st.divider()
st.caption("Developed using Streamlit, Python and Machine Learning")