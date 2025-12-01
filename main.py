# %%
import pandas as pd
import numpy as np


# %%
energy_df = pd.read_csv('energy_data.csv')
weather_df = pd.read_csv('weather_data.csv')

display(energy_df.head())
display(weather_df.head())

display({"Energy shape": energy_df.shape, "Weather shape": weather_df.shape})


# %%
display(energy_df.columns)
display(weather_df.columns)

display(energy_df.head(3))
display(weather_df.head(3))

# %%
# Fix datetime into a proper datetime column, then extract a pure 'date' column for daily grouping.

energy_df = energy_df.copy()

energy_df['datetime'] = pd.to_datetime(energy_df['Date & Time'], errors='coerce')
energy_df['date'] = energy_df['datetime'].dt.date

display(energy_df[['Date & Time', 'datetime', 'date']].head())


# %%
# create a 'date' column in the weather data that matches the energy data's 'date'

weather_df = weather_df.copy()

weather_df['datetime'] = pd.to_datetime(weather_df['time'], unit='s', errors='coerce')
weather_df['date'] = weather_df['datetime'].dt.date

display(weather_df[['time', 'datetime', 'date']].head())


# %% [markdown]
# 1. Examine the data, parse the time fields wherever necessary. Take the sum of the energy usage 
# (Use [kW]) to get per day usage and merge it with weather data (10 Points). 

# %%
# Examine the data, parse the time fields wherever necessary. Take the sum of the energy usage 


energy_df = energy_df.copy()

daily_energy_df = (
    energy_df
    .groupby('date', as_index=False)['use [kW]']
    .sum()
    .rename(columns={'use [kW]': 'daily_use_kW'})
)

display(daily_energy_df.head())
display({"Daily energy shape": daily_energy_df.shape})

# %%
#   The weather data is recorded multiple times per day with hourly timestamps
#   but our energy data has been summed to one row per day.
#   Group weather_df by 'date' and compute daily values
#   This creates daily_weather_df with one row per date for merging


weather_df = weather_df.copy()

daily_weather_df = (
    weather_df
    .groupby('date', as_index=False)
    .mean(numeric_only=True)  # averages all numeric weather features per day
)

display(daily_weather_df.head())
display({"Daily weather shape": daily_weather_df.shape})


# %%
#  Inner merge on the shared 'date' column between daily_energy_df
#  and daily_weather_df to create merged_df with one row per day that has both
#  daily_use_kW and daily weather statistics.

merged_df = pd.merge(
    daily_energy_df,
    daily_weather_df,
    on='date',
    how='inner'
)

display(merged_df.head())
display({"Merged shape": merged_df.shape})


# %% [markdown]
# 2. Split the data obtained from step 1, into training and testing sets. The aim is to predict the usage 
# for each day in the month of December using the weather data, so split accordingly. The usage as 
# per devices should be dropped, only the “use [kW]” column is to be used for prediction from the 
# dataset (5 points).

# %%
#   Splitting data so December is test set, rest is training/validation.


merged_df = merged_df.copy()
merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')

is_december = merged_df['date'].dt.month == 12
is_not_december = merged_df['date'].dt.month != 12

display(merged_df[is_december].head()) 
display({"December count": is_december.sum(),
         "Non-December count": is_not_december.sum()})


# %%
#   Keep 'daily_use_kW' in the training set as the label.
#   Remove 'daily_use_kW' from the test set as the y value.

train_df = merged_df[is_not_december].copy()
test_df  = merged_df[is_december].copy()

test_features_df = test_df.drop(columns=['daily_use_kW'])

display(train_df.head())
display(test_features_df.head())

display({
    "Train rows": len(train_df),
    "Test rows": len(test_df),
    "Test features shape": test_features_df.shape
})


# %% [markdown]
# 3. Linear Regression - Predicting Energy Usage: 
# Set up a simple linear regression model to train, and then predict energy usage for each day in the 
# month of December using features from weather data (Note that you need to drop the “use [kW]” 
# column in the test set first). How well/badly does the model work? (Evaluate the correctness of 
# your predictions based on the original “use [kW]” column). Calculate the Root mean squared error 
# of your model. 
# Finally generate a csv dump of the predicted values. Format of csv: Two columns, first should be 
# the date and second should be the predicted value. (20 points)

# %%
#   Build matrices X_train, y_train, X_test using only numerical weather columns.
#   Drop 'date' and any non-numeric columns.


feature_cols = [
    'temperature', 'humidity', 'visibility', 'pressure', 'windSpeed',
    'cloudCover', 'time', 'windBearing', 'precipIntensity', 'dewPoint',
    'precipProbability'
]

X_train = train_df[feature_cols].copy()
y_train = train_df['daily_use_kW'].copy()

X_test = test_features_df[feature_cols].copy()

display(X_train.head())
display(y_train.head())
display(X_test.head())

# %%
#  Train a linear regression model using X_train and y_train.

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(X_train, y_train)


display({"Intercept": linreg.intercept_, "Coefficients": linreg.coef_})

# %%
# Testing the model on December data
y_pred = linreg.predict(X_test)
display(y_pred[:10])

# %%
#   Evaluate model performance on December test set using RMSE

from sklearn.metrics import mean_squared_error
import numpy as np

y_test_actual = test_df['daily_use_kW'].values

mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)

display({"RMSE": rmse})

# %%
# Exporting linear regression predictions to CSV
pred_df = test_df[['date']].copy()
pred_df['predicted_daily_use_kW'] = y_pred

display(pred_df.head())

pred_df.to_csv('linear_regression_predictions.csv', index=False)

# %%
#   Compare to base model of just putting mean for everything
baseline_pred_value = y_train.mean()

baseline_pred = np.full_like(y_test_actual, baseline_pred_value, dtype=float)

baseline_rmse = np.sqrt(mean_squared_error(y_test_actual, baseline_pred))

display({
    "Training Mean (Baseline Prediction)": baseline_pred_value,
    "Baseline RMSE": baseline_rmse,
    "Linear Regression RMSE": 10.731619230473463
})


# %% [markdown]
# Compared to just guesing the mean for everyday in December ended up being a better RMSE than linear regresion model, so it's actualy not that good of a model. It could be because the relationship between wetaher and total daily usage wasn't linear and December may have weird patterns.

# %% [markdown]
# 4. Logistic Regression - Temperature classification: 
# Using only weather data we want to classify if the temperature is high or low. Let's assume 
# temperature greater than or equal to 35 is ‘high’ and below 35 is ‘low’. Set up a logistic regression 
# model to classify the temperature for each day in the month of December. Calculate the F1 score 
# for the model. 
# 

# %%
#   Create a new binary feature 'temp_high' in merged_df
#   that is 1 if temperature >= 35 degrees Celsius, else 0.
merged_df = merged_df.copy()

merged_df['temp_high'] = (merged_df['temperature'] >= 35).astype(int)

display(merged_df[['date', 'temperature', 'temp_high']].head(15))


# %%
#   Reuse the same weather feature columns from Task 3.
#   Use merged_df with the temp_high label.
#   Split into December vs non-December sets from Task 2.

feature_cols = [
    'temperature', 'humidity', 'visibility', 'pressure', 'windSpeed',
    'cloudCover', 'time', 'windBearing', 'precipIntensity', 'dewPoint',
    'precipProbability'
]

# Training = Non-December
X_train_logit = merged_df[is_not_december][feature_cols].copy()
y_train_logit = merged_df[is_not_december]['temp_high'].copy()

# Testing = December
X_test_logit = merged_df[is_december][feature_cols].copy()
y_test_logit = merged_df[is_december]['temp_high'].copy()

display(X_train_logit.head())
display(y_train_logit.head())
display(X_test_logit.head())
display(y_test_logit.head())

display({
    "Train rows": len(X_train_logit),
    "Test rows": len(X_test_logit)
})

# %%
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_logit, y_train_logit)

display({
    "Intercept": logreg.intercept_,
    "Coefficients": logreg.coef_
})

# %%
from sklearn.metrics import f1_score

y_pred_logit = logreg.predict(X_test_logit)
display(y_pred_logit[:10])

f1 = f1_score(y_test_logit, y_pred_logit)
display({"F1 Score": f1})

# %%
# Exporting logistic regression predictions to CSV
logit_pred_df = merged_df[is_december][['date']].copy()
logit_pred_df['temp_classification'] = y_pred_logit

display(logit_pred_df.head())

logit_pred_df.to_csv('logistic_regression_temperature_classification.csv', index=False)

# %% [markdown]
# 5. Energy usage data Analysis: 
# We want to analyze how different devices are being used in different times of the day. - Is the washer being used only during the day? - During what time of the day is AC used most? 
# There are a number of questions that can be asked. 
# For simplicity, let’s divide a day in two parts: - Day: 6AM - 7PM - Night: 7PM - 6AM 
# Analyze the usage of any two devices of your choice during the ‘day’ and ‘night’. Plot these 
# trends. Explain your findings. (10 points)

# %%
#   Extract the hour from the 'datetime' column into a new 'hour' column
#   so we can filter (6 ≤ hour < 19) for "day" and (hour >= 19 or hour < 6) for "night".

energy_df = energy_df.copy()
energy_df['hour'] = energy_df['datetime'].dt.hour

display(energy_df[['datetime', 'hour']].head())


# %%
#   Create a new column 'period' for day or night interval

energy_df = energy_df.copy()

energy_df['period'] = np.where(
    (energy_df['hour'] >= 6) & (energy_df['hour'] < 19),
    'day',
    'night'
)

display(energy_df[['datetime', 'hour', 'period']].head(12))


# %%
#   Compare day vs night using period column for the analysis with mean.

devices = ['AC [kW]', 'Washer [kW]']

day_night_usage = (
    energy_df.groupby('period')[devices]
    .mean()
)

display(day_night_usage)

# %%
import matplotlib.pyplot as plt

# Extract values
ac_day = day_night_usage.loc['day', 'AC [kW]']
ac_night = day_night_usage.loc['night', 'AC [kW]']

washer_day = day_night_usage.loc['day', 'Washer [kW]']
washer_night = day_night_usage.loc['night', 'Washer [kW]']

# AC
plt.figure(figsize=(8, 5))
plt.bar(['Day', 'Night'], [ac_day, ac_night])
plt.title('AC Usage: Day vs Night')
plt.ylabel('Average kW')
plt.xlabel('Period')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# Washer
plt.figure(figsize=(8, 5))
plt.bar(['Day', 'Night'], [washer_day, washer_night])
plt.title('Washer Usage: Day vs Night')
plt.ylabel('Average kW')
plt.xlabel('Period')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# %% [markdown]
# AC consumption is higher at night than during the day.
# This is probably because of heating cycles or nighttime cooling/heating when outside temperature drops.
# Since AC and heating responds to temperature, nighttime temperature changes might cause it to work more.
# 
# Washer usage is higher during the day than at night.
# This makes sense since people use laundry in the day time.


