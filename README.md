# 🛍️ Walmart Sales Data Analysis & Forecasting

This project is a **Streamlit-powered data science dashboard** that provides a comprehensive analysis of Walmart's historical weekly sales data, including machine learning-based forecasting, anomaly detection, and live sales prediction. The dashboard leverages advanced models like **Random Forest**, **XGBoost**, and **Prophet** to generate insights into future sales performance.

---

## 📌 Project Overview

The main goal of this project is to analyze Walmart's weekly sales data to:

- 📊 **Analyze Sales Trends & Insights**: Provide visualizations of total sales per store and sales trends over time.
- 🔍 **Detect Missing Data & Anomalies**: Highlight and handle any missing values and anomalies in the dataset.
- 📈 **Use Machine Learning Models for Forecasting**: Build models to predict future sales based on various input features.
- 🔮 **Interactive Sales Prediction**: Allow users to input store details and get a predicted sales figure.
- 📦 **Inventory Forecasting**: Predict inventory needs using **XGBoost**.
- 🎉 **Holiday Impact Analysis**: Analyze the effect of holidays (e.g., Black Friday) on sales performance.
- 📄 **PDF Report Generation**: Export a clean PDF report with the analysis and sales predictions.

---

## ⚙️ Technologies Used

| Category           | Tools / Libraries                                  |
|--------------------|-----------------------------------------------------|
| **Web UI**         | Streamlit                                          |
| **Data Handling**  | pandas                                              |
| **Data Visualization** | plotly, seaborn, matplotlib                      |
| **Machine Learning**| scikit-learn (Random Forest), XGBoost              |
| **Time Series Forecasting** | Prophet                                    |
| **Anomaly Detection** | IsolationForest (from scikit-learn)               |
| **Reporting**      | fpdf                                                |

---

## 🔧 Key Features

### 1. 🗃️ **Data Upload**  
Upload your own CSV file of Walmart sales data and the app will automatically begin analysis.

### 2. 📊 **Exploratory Data Analysis (EDA)**  
- Dataset preview and summary of basic statistics.
- Missing value detection and handling.
- Visualize total sales by store.
- Visualize sales trends over time to see how weekly sales have evolved.

### 3. 💡 **Machine Learning Models for Sales Prediction**
- **Random Forest**: Predict weekly sales based on multiple input features like store number, temperature, fuel price, CPI (Consumer Price Index), unemployment rate, and whether it's a holiday week.
- **XGBoost**: Build an additional model for predicting inventory based on similar features.
- **Prophet**: Use for time-series forecasting to predict future sales trends based on past data.

### 4. 🚨 **Anomaly Detection**  
- The app uses **Isolation Forest** to detect anomalies in sales data (e.g., unusually high or low sales).
- Visualize the identified anomalies.

### 5. 🔮 **Predict Weekly Sales (Interactive)**  
Users can enter the following inputs:
- Store number
- Temperature (°F)
- Fuel Price ($ per gallon)
- CPI (Consumer Price Index)
- Unemployment Rate (%)
- Is it a holiday week?  
➡️ Based on these inputs, the app predicts the weekly sales for the given store.

### 6. 📄 **PDF Report Generation**  
- Generate a **PDF report** with key predictions and insights.
- The PDF contains:
  - Prediction result (Predicted Weekly Sales)
  - Overview of the models used (Random Forest, XGBoost, Prophet)
  - Explanation of the input features and their significance in the model

---

## 🧪 Sample Input Features for Prediction

When you interact with the app, you will be prompted to provide the following details:

- **Store**: Store number (1–50).
- **Temperature**: Store temperature in Fahrenheit.
- **Fuel_Price**: Fuel price in USD per gallon.
- **CPI**: Consumer Price Index (an economic indicator).
- **Unemployment**: Unemployment rate (percentage).
- **Holiday**: Specify if the week corresponds to a holiday (Yes/No).

---

## 🚀 How to Run the Project

### Step 1: Clone the Repository
```bash
git clone https://github.com/shankar212/walmart-sales-data-analysis.git
cd walmart-sales-data-analysis
