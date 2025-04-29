import pandas as pd

# Load Seattle Weather dataset. (from https://www.kaggle.com/datasets/ananthr1/weather-prediction)
weather = pd.read_csv('seattle-weather.csv')

# Filter out any rows with missing information or duplicate information.
weather = weather.dropna().drop_duplicates()

#print(weather.describe())
#print(weather["weather"].value_counts())

# Create a new random sample database with the newly cleaned Soccer dataset.
weather.to_csv("Seattle_Weather_Cleaned.csv")