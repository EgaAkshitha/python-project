import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("healthcare_dataset.csv")

# --- Data Inspection and Cleaning ---
print("--- Initial Data Inspection ---")
print(df.head())
print("\n--- Data Information ---")
print(df.info())

# Remove rows with non-numeric values in 'Billing Amount' and convert to numeric
df['Billing Amount'] = pd.to_numeric(df['Billing Amount'], errors='coerce')
df.dropna(subset=['Billing Amount'], inplace=True)

# Convert date columns to datetime objects
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], format='%d-%m-%Y', errors='coerce')
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], format='%d-%m-%Y', errors='coerce')
df.dropna(subset=['Date of Admission', 'Discharge Date'], inplace=True)

# --- Basic Statistics using NumPy and Pandas ---
print("\n--- Basic Statistics ---")
print(f"Average Age: {np.mean(df['Age']):.2f} years")
print(f"Maximum Billing Amount: ${np.max(df['Billing Amount']):.2f}")
print(f"Minimum Billing Amount: ${np.min(df['Billing Amount']):.2f}")

# --- Feature Engineering: Length of Stay ---
# Calculate the number of days a patient stayed in the hospital
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

# --- Visualizations using Matplotlib ---

# 1. Histogram of Age
plt.figure(figsize=(8, 5))
plt.hist(df['Age'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Patient Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 2. Bar Chart of Average Billing Amount by Insurance Provider
avg_billing_by_insurance = df.groupby('Insurance Provider')['Billing Amount'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
avg_billing_by_insurance.plot(kind='bar', color='lightcoral')
plt.title('Average Billing Amount by Insurance Provider')
plt.xlabel('Insurance Provider')
plt.ylabel('Average Billing Amount ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3. Line Plot of Admissions Over Time
admissions_over_time = df.groupby('Date of Admission').size()
plt.figure(figsize=(12, 6))
admissions_over_time.plot(kind='line', marker='o', linestyle='-')
plt.title('Number of Admissions Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Admissions')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()