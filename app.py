import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("AI Powered E-Commerce Sales Prediction & Analytics Dashboard")
st.write("Machine Learning and Deep Learning based E-Commerce Data Analysis System")

# Load dataset
df = pd.read_csv("online_retail.csv.zip", compression="zip")

# Remove rows where CustomerID is missing
df = df.dropna(subset=["CustomerID"])

# Create revenue column
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.metric("Total Records", len(df))

# Sales by Month Bar Chart
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

monthly_sales = df.groupby(df["InvoiceDate"].dt.to_period("M"))["Revenue"].sum()

st.subheader("Monthly Sales Trend")

fig, ax = plt.subplots()
monthly_sales.plot(kind="line", marker="o", ax=ax)

st.pyplot(fig)

st.bar_chart(monthly_sales)

# Charts Side by Side
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Countries by Revenue")
    country_revenue = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(10)
    st.bar_chart(country_revenue)

with col2:
    st.subheader("Top Selling Products")
    top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_products)

#Country Filter Visualization
st.sidebar.header("Filter")

country = st.sidebar.selectbox("Select Country", df["Country"].unique())

filtered_data = df[df["Country"] == country]

st.write("Filtered Data", filtered_data.head())

# Sales Trend for Selected Country
filtered_data = filtered_data.copy()

# Sales Trend for Selected Country
filtered_data["InvoiceDate"] = pd.to_datetime(filtered_data["InvoiceDate"])

country_monthly_sales = filtered_data.groupby(
    filtered_data["InvoiceDate"].dt.to_period("M")
)["Revenue"].sum()

st.subheader("Monthly Sales Trend for Selected Country")

st.line_chart(country_monthly_sales)

# Top product in Selected Country
country_sales = filtered_data.groupby("Description")["Revenue"].sum().sort_values(ascending=False).head(10)

st.subheader("Top Products in Selected Country")

st.bar_chart(country_sales)

#Country Segmentation
from sklearn.cluster import KMeans

customer_data = df.groupby("CustomerID").agg({
    "Revenue":"sum",
    "InvoiceNo":"count"
}).dropna()

kmeans = KMeans(n_clusters=3, n_init=10)

customer_data["Cluster"] = kmeans.fit_predict(customer_data)

st.subheader("Customer Segmentation")

st.write(customer_data.head())

#Cluster Visualization(AI Graph)

st.subheader("Customer Segmentation Visualization")

fig, ax = plt.subplots()

ax.scatter(customer_data["InvoiceNo"], 
           customer_data["Revenue"], 
           c=customer_data["Cluster"])

ax.set_xlabel("Number of Orders")
ax.set_ylabel("Revenue")

st.pyplot(fig)


# KPI Cards
col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", round(df["Revenue"].sum(),2))

col2.metric("Total Orders", df["InvoiceNo"].nunique())

col3.metric("Total Customers", df["CustomerID"].nunique())

# Slidebar Branding
st.sidebar.title("AI E-Commerce Analytics")

st.sidebar.write("Data Analytics Dashboard")

st.sidebar.write("Built with Python & Streamlit")

# Sales Prediction (AI Model)
from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare data
df["MonthNumber"] = df["InvoiceDate"].dt.month

monthly_data = df.groupby("MonthNumber")["Revenue"].sum().reset_index()

X = monthly_data[["MonthNumber"]]
y = monthly_data["Revenue"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next month sales
next_month = np.array([[monthly_data["MonthNumber"].max()+1]])
prediction = model.predict(next_month)

st.subheader("AI Sales Prediction")

st.metric("Predicted Next Month Sales", round(prediction[0],2))

#Prediction Graph
fig, ax = plt.subplots()

ax.plot(monthly_data["MonthNumber"], monthly_data["Revenue"], marker="o")

ax.scatter(monthly_data["MonthNumber"].max()+1, prediction[0], color="red")

ax.set_title("Sales Prediction Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue")

st.pyplot(fig)

# Dashboard Matrices
avg_order = df["Revenue"].mean()

st.metric("Average Order Value", round(avg_order,2))

# Product Recommendation System
st.subheader("Product Recommendation System")

product = st.selectbox(
    "Select a Product",
    df["Description"].dropna().unique()
)

# Find products frequently bought with selected product
product_data = df[df["Description"] == product]

invoice_ids = product_data["InvoiceNo"].unique()

recommended_products = df[df["InvoiceNo"].isin(invoice_ids)]

top_recommendations = recommended_products["Description"].value_counts().drop(product).head(5)

st.write("Recommended Products")

st.bar_chart(top_recommendations)

# Customer Lifetime Value Analysis
st.subheader("Customer Lifetime Value (CLV)")

customer_value = df.groupby("CustomerID").agg({
    "Revenue": "sum",
    "InvoiceNo": "count"
}).rename(columns={
    "Revenue": "TotalSpend",
    "InvoiceNo": "TotalOrders"
})

customer_value["CLV"] = customer_value["TotalSpend"] / customer_value["TotalOrders"]

top_customers = customer_value.sort_values("CLV", ascending=False).head(10)

st.write("Top High Value Customers")

st.bar_chart(top_customers["CLV"])

# Deep Learning Sales Prediction
try:
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    dl_available = True
except:
    dl_available = False
st.subheader("Deep Learning Sales Prediction")
if not dl_available:
    st.warning("Deep Learning module not available on cloud environment.")
else:
    # Prepare monthly data
    df["MonthNumber"] = df["InvoiceDate"].dt.month
    monthly_data = df.groupby("MonthNumber")["Revenue"].sum().reset_index()

    X = monthly_data[["MonthNumber"]]
    y = monthly_data["Revenue"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    # Neural Network model
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(1,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1)
    ])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.fit(X_scaled, y, epochs=100, verbose=0)

# Predict next month
next_month = scaler.transform([[monthly_data["MonthNumber"].max()+1]])
prediction = model.predict(next_month)

st.metric("Deep Learning Predicted Sales", round(prediction[0][0], 2))

# Deep Learning Graph
fig, ax = plt.subplots()

ax.plot(monthly_data["MonthNumber"], monthly_data["Revenue"], marker="o")

ax.scatter(monthly_data["MonthNumber"].max()+1, prediction[0][0], color="red")

ax.set_title("Deep Learning Sales Forecast")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue")

st.pyplot(fig)

# Download dataset Button
st.download_button(
    "Download Dataset",
    df.to_csv(index=False),
    file_name="ecommerce_data.csv"
)

# Dashboard Footer
st.markdown("---")
st.write("AI Powered E-Commerce Analytics Dashboard")
st.write("Created by Ayush Kumar")
st.write("M.C.A (Data Science Analytics)")


