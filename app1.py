import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from fpdf import FPDF
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
# Page Config
st.set_page_config(page_title="🛍️ Walmart Sales Analysis", layout="wide")

# Title & Introduction
st.title("🛍️ Walmart Weekly Sales Forecasting & Analysis")
st.markdown("Gain insights and predict future sales using machine learning on Walmart’s historical dataset.")

# Sidebar - File Upload
st.sidebar.header("📂 Upload CSV Data")
uploaded_file = st.sidebar.file_uploader("Choose Walmart Sales Data CSV", type="csv")

# Load Data
@st.cache_resource
def load_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    return df

if uploaded_file:
    data = load_data(uploaded_file)
else:
    st.info("📤 Please upload a CSV file to start the analysis.")
    st.stop()

# Data Preview
st.subheader("🗃️ Dataset Preview")
st.dataframe(data.head(), use_container_width=True)

# Summary Statistics
st.subheader("📈 Summary Statistics")
st.write(data.describe())

# Missing Data
st.subheader("🚨 Missing Values")
missing = data.isnull().sum()
st.write(missing[missing > 0] if missing.sum() else "✅ No missing values found!")

# Store-wise Sales Visualization
st.subheader("🏬 Total Sales by Store")
store_sales = data.groupby('Store')['Weekly_Sales'].sum().reset_index()
fig1 = px.bar(store_sales, x='Store', y='Weekly_Sales', title='Total Weekly Sales by Store', color='Weekly_Sales')
st.plotly_chart(fig1, use_container_width=True)

# Sales Trend Over Time
st.subheader("📅 Sales Trend Over Time")
sales_over_time = data.groupby('Date')['Weekly_Sales'].sum().reset_index()
fig2 = px.line(sales_over_time, x='Date', y='Weekly_Sales', title='Weekly Sales Over Time')
st.plotly_chart(fig2, use_container_width=True)

# Feature Engineering
data.rename(columns={"Holiday_Flag": "IsHoliday"}, inplace=True)
features = ['Store', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday']
target = 'Weekly_Sales'

X = data[features]
y = data[target]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.subheader("📉 Model Performance")
st.metric("Mean Squared Error (MSE)", f"{mse:,.2f}")

# Sales Prediction
st.subheader("🔮 Predict Weekly Sales")
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        store = st.number_input("Store", min_value=1, max_value=50, value=1, help="Enter store number (1–50)")
        fuel_price = st.number_input("Fuel Price ($)", value=2.5, step=0.01)

    with col2:
        temp = st.number_input("Temperature (°F)", value=65.0, step=0.1)
        cpi = st.number_input("CPI", value=220.0, step=0.1)

    with col3:
        unemployment = st.number_input("Unemployment Rate (%)", value=7.5, step=0.1)
        is_holiday = st.selectbox("Is it a Holiday Week?", ["No", "Yes"])

    submitted = st.form_submit_button("🚀 Predict")

    if submitted:
        input_data = pd.DataFrame({
            'Store': [store],
            'Temperature': [temp],
            'Fuel_Price': [fuel_price],
            'CPI': [cpi],
            'Unemployment': [unemployment],
            'IsHoliday': [1 if is_holiday == "Yes" else 0]
        })

        prediction = model.predict(input_data)[0]
        st.success(f"🛒 Predicted Weekly Sales: **${prediction:,.2f}**")

# Anomaly Detection with Isolation Forest
st.subheader("🚨 Anomaly Detection")
iso_forest = IsolationForest(contamination=0.1)
anomaly_preds = iso_forest.fit_predict(data[['Weekly_Sales']])

# Mark anomalies in the data
data['Anomaly'] = anomaly_preds
anomalous_data = data[data['Anomaly'] == -1]

# Display anomalous points
st.write(anomalous_data)

# Inventory Forecasting with XGBoost
st.subheader("📦 Inventory Forecasting")
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)

# Predict Sales for Future Weeks
future_sales_pred = xgb_model.predict(X_test)
st.write("Future Sales Predictions:", future_sales_pred)

# Price Optimization
st.subheader("💲 Price Optimization")
fig3, ax = plt.subplots(figsize=(10, 6))  # Create a figure object
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig3)

# Holiday Impact Analysis
st.subheader("🎉 Holiday Impact Analysis")
holiday_sales = data[data['Date'].dt.month == 11]  # Example: November (Black Friday)
holiday_avg_sales = holiday_sales.groupby('Date')['Weekly_Sales'].mean()

fig4 = px.line(holiday_avg_sales, title="Average Sales During Major Holidays (e.g., Black Friday)")
st.plotly_chart(fig4)

# PDF Report Exporter
st.subheader("📄 Download Sales Report")

if 'prediction' in locals():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_auto_page_break(auto=True, margin=15)

    # No emojis or non-ASCII characters!
    pdf.cell(200, 10, txt="Walmart Sales Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt="Key Insights and Predictions", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Predicted Weekly Sales: ${prediction:,.2f}", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=(
        "This report includes insights generated using machine learning models like "
        "Random Forest and XGBoost, along with anomaly detection using Isolation Forest.\n\n"
        "The prediction above was made based on input values such as temperature, fuel price, "
        "CPI, unemployment rate, and whether it's a holiday week."
    ))

    # Get PDF as string and write to BytesIO
    pdf_output = pdf.output(dest='S').encode('latin-1')  # Must encode to latin-1
    pdf_buffer = BytesIO(pdf_output)

    st.download_button(
        label="📥 Download Report as PDF",
        data=pdf_buffer,
        file_name="walmart_sales_report.pdf",
        mime="application/pdf"
    )
else:
    st.info("⚠️ Make a prediction first to generate the report.")

# Footer
st.markdown("---")
st.markdown("🔗 Developed by **Rathod Shanker**  \n📧 [shanker.rathod77@gmail.com](mailto:shanker.rathod77@gmail.com)")
st.markdown("📊 Data Source: [Walmart Sales Dataset on Kaggle](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data)")
