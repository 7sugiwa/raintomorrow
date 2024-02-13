# eda.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import phik  # Ensure this library is installed

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("weatherAUS.csv")
    return df

def plot_data_distributions(df):
    sns.set_style('whitegrid')
    for column in df.columns:
        plt.figure(figsize=(8,4))
        if len(df[column].unique()) > 10:
            sns.histplot(df[column], kde=True, color='skyblue')
            plt.title(f'Distribution of {column}')
        else:
            sns.countplot(x=column, data=df, palette='Set2')
            plt.title(f'Count of different classes in {column}')
        st.pyplot(plt)

def calculate_phi_k_correlation(df):
    phi_k_correlation = df.phik_matrix()
    plt.figure(figsize=(12, 10))
    sns.heatmap(phi_k_correlation, annot=True, fmt=".2f", linewidths=.5, cmap='coolwarm')
    plt.title('Phi_k Correlation Matrix Heatmap')
    st.pyplot(plt)

def perform_temporal_feature_extraction(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    return df

def plot_annual_and_monthly_trends(df):
    annual_trends = df.groupby('Year')[['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm']].mean()
    monthly_trends = df.groupby('Month')[['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm']].mean()

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    annual_trends[['MinTemp', 'MaxTemp']].plot(ax=axes[0,0], title='Annual Avg Temperature')
    annual_trends['Rainfall'].plot(ax=axes[0,1], title='Annual Avg Rainfall')
    annual_trends[['Humidity9am', 'Humidity3pm']].plot(ax=axes[0,2], title='Annual Avg Humidity')
    annual_trends[['Pressure9am', 'Pressure3pm']].plot(ax=axes[0,3], title='Annual Avg Pressure')
    monthly_trends[['MinTemp', 'MaxTemp']].plot(ax=axes[1,0], title='Monthly Avg Temperature')
    monthly_trends['Rainfall'].plot(ax=axes[1,1], title='Monthly Avg Rainfall')
    monthly_trends[['Humidity9am', 'Humidity3pm']].plot(ax=axes[1,2], title='Monthly Avg Humidity')
    monthly_trends[['Pressure9am', 'Pressure3pm']].plot(ax=axes[1,3], title='Monthly Avg Pressure')

    plt.tight_layout()
    st.pyplot(fig)

def perform_missing_value_analysis(df):
    missing_values_total = df.isnull().sum()
    missing_values_percentage = (df.isnull().sum() / len(df)) * 100
    missing_values_analysis = pd.DataFrame({'Total Missing': missing_values_total, 'Percentage Missing': missing_values_percentage})

    st.write(missing_values_analysis.sort_values(by='Percentage Missing', ascending=False))

def perform_outlier_detection(df, key_columns):
    outlier_analysis = {}
    for col in key_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_analysis[col] = {
            "Outliers": outliers.shape[0],
            "Percentage": (outliers.shape[0] / df.shape[0]) * 100
        }
    st.write(outlier_analysis)

def perform_categorical_data_analysis(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    categorical_analysis = {col: df[col].value_counts() for col in categorical_columns}
    st.write(categorical_analysis)

# Main Function
def main():
    st.title("Exploratory Data Analysis - Weather Forecasting")

    # Load and preprocess data
    df = load_and_preprocess_data()

    # Extract temporal features
    df = perform_temporal_feature_extraction(df)  # This should be called before using 'Year' column

    # Checkboxes and plotting functions
    if st.sidebar.checkbox("Show Data Distributions"):
        plot_data_distributions(df)
    if st.sidebar.checkbox("Show Correlation Heatmap"):
        calculate_phi_k_correlation(df)
    if st.sidebar.checkbox("Show Annual and Monthly Trends"):
        plot_annual_and_monthly_trends(df)
    if st.sidebar.checkbox("Show Missing Value Analysis"):
        perform_missing_value_analysis(df)
    if st.sidebar.checkbox("Show Outlier Analysis"):
        key_columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm']
        perform_outlier_detection(df, key_columns)
    if st.sidebar.checkbox("Show Categorical Data Analysis"):
        perform_categorical_data_analysis(df)

if __name__ == '__main__':
    main()
