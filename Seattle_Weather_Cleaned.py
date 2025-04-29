import pandas as pd

# Load the Seattle Weather dataset from the specified CSV file.
# This dataset is publicly available on Kaggle: https://www.kaggle.com/datasets/ananthr1/weather-prediction
weather = pd.read_csv('seattle-weather.csv')

# Drop rows with missing data and any duplicates from the dataset.
# This ensures that we only work with complete and unique records for analysis.
weather = weather.dropna().drop_duplicates()

# The following lines (currently commented out) can be used to inspect the dataset:
# - weather.describe(): Provides a statistical summary of numerical columns.
# - weather["weather"].value_counts(): Shows the frequency of each weather condition in the 'weather' column.
# Uncomment to explore the dataset further if needed.
# print(weather.describe())
# print(weather["weather"].value_counts())

# Save the cleaned version of the dataset to a new CSV file called 'Seattle_Weather_Cleaned.csv'.
# This cleaned dataset will be used for training and further analysis.
weather.to_csv("Seattle_Weather_Cleaned.csv")