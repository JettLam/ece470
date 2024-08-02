import numpy as np
import pandas as pd


# Read the CSV file into a DataFrame
df = pd.read_csv('C:\\Users\\User\\Desktop\\ECE470 Proj\\Data Cleanu\\train_timeseries.csv')

# Drop the 'fips' column
df = df.drop(columns=['fips'])

df['date'] = pd.to_datetime(df['date'])

# Filter out rows where the date is before 2020
df = df[df['date'] >= '2010-01-01']

# Save the modified DataFrame back to a CSV file
df.to_csv('C:\\Users\\User\\Desktop\\ECE470 Proj\\Data Cleanu\\trimmed_data.csv', index=False)

