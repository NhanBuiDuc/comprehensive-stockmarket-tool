import requests
import datetime
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder

# Define the categories
categories = ['Sunny', 'Partly cloudy', 'Cloudy', 'Overcast', 'Mist', 'Patchy rain possible',
              'Patchy snow possible', 'Patchy sleet possible', 'Patchy freezing drizzle possible',
              'Thundery outbreaks possible', 'Blowing snow', 'Blizzard', 'Fog', 'Freezing fog',
              'Patchy light drizzle', 'Light drizzle', 'Freezing drizzle', 'Heavy freezing drizzle',
              'Patchy light rain', 'Light rain', 'Moderate rain at times', 'Moderate rain',
              'Heavy rain at times', 'Heavy rain', 'Light freezing rain',
              'Moderate or heavy freezing rain', 'Light sleet', 'Moderate or heavy sleet',
              'Patchy light snow', 'Light snow', 'Patchy moderate snow', 'Moderate snow',
              'Patchy heavy snow', 'Heavy snow', 'Ice pellets', 'Light rain shower',
              'Moderate or heavy rain shower', 'Torrential rain shower', 'Light sleet showers',
              'Moderate or heavy sleet showers', 'Light snow showers', 'Moderate or heavy snow showers',
              'Light showers of ice pellets', 'Moderate or heavy showers of ice pellets',
              'Patchy light rain with thunder', 'Moderate or heavy rain with thunder',
              'Patchy light snow with thunder', 'Moderate or heavy snow with thunder']


def download_historical_whether(api_key, query, delta):
    # Subtract the timedelta object from the from_date to get the to_date
    from_date = datetime.now() - delta
    results = []
    folder_path = "./NLP/whether_data"
    filename = query + "_whether_data.csv"
    # Loop over the dates between from_date and to_date
    dt = from_date
    while dt < datetime.now():
        # Convert the date to a string in the format expected by the API
        dt_str = dt.strftime('%Y-%m-%d')

        # Construct the URL for the API call
        url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={query}&dt={dt_str}"

        # Make the API call
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            results.extend(data['forecast']['forecastday'])

        # Move on to the previous date
        dt = dt + timedelta(days=1)

    # Check if file with the given name exists in the folder
    file_path = os.path.join(folder_path, filename)
    df = pd.DataFrame.from_records(results)

    # Convert the `datetime` column to a pandas datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Set the `datetime` column as the index
    df.set_index('date', inplace=True)

    # Export the DataFrame to a CSV file
    df.to_csv(file_path)


def prepare_whether_data(new_data=False):
    # Set the API key
    api_key = 'b7a439cb870a4a09be9114748230705'
    # Create the encoder
    encoder = OneHotEncoder(categories=[categories])
    window_size = 7

    # Set the search parameters
    query1 = "Ha Noi"
    query2 = "Ho Chi Minh"
    query3 = "Da Nang"
    # Define the from_date as the current date and time
    delta = timedelta(days=window_size - 1)
    if new_data:
        download_historical_whether(api_key, query1, delta)
        download_historical_whether(api_key, query2, delta)
        download_historical_whether(api_key, query3, delta)

    # Create an empty list to store day arrays
    day_arrays = []

    # Iterate over all CSV files in the folder
    folder_path = './NLP/whether_data/'
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Load CSV file into DataFrame
            df = pd.read_csv(os.path.join(folder_path, filename))
            # Extract 'day' column into NumPy array and append to list
            dict_column = df['day'].values
            results = []
            for j in dict_column:
                dict = json.loads(j.replace("'", "\""))
                whether = np.array([dict.pop('condition')["text"]])
                encoded_data = np.array(encoder.fit_transform(whether.reshape(-1, 1)).toarray())
                encoded_data = encoded_data.reshape(-1)
                my_array = np.array(list(dict.values()))
                my_array = np.insert(my_array, -1, encoded_data, axis=0)
                results.append(my_array)

            day_arrays.append(results)

    # Merge all day arrays into one NumPy array
    whether_day_array = np.array(day_arrays)
    whether_day_array = whether_day_array.reshape(window_size, -1)
    return whether_day_array
