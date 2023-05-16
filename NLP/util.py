import requests
import datetime
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder
from transformers import pipeline, AutoTokenizer, AutoModel
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import torch
import nltk
from keras_preprocessing.sequence import pad_sequences
from GoogleNews import GoogleNews
import finnhub
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


def download_historical_whether(api_key, query, from_date, to_date):
    # Subtract the timedelta object from the from_date to get the to_date
    if to_date is None:
        to_date = datetime.now()
    else:
        to_date = datetime.strptime(to_date, "%Y-%m-%d")
    results = []
    folder_path = "./NLP/whether_data"
    filename = query + "_whether_data.csv"
    # Loop over the dates between from_date and to_date
    dt = datetime.strptime(from_date, "%Y-%m-%d")
    while dt < to_date:
        # Convert the date to a string in the format expected by the API
        dt_str = dt.strftime('%Y-%m-%d')

        # Construct the URL for the API call
        url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={query}&dt={dt_str}"

        # Make the API call
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            results.extend(data['forecast']['forecastday'])

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


def prepare_time_series_whether_data(data, window_size, output_step, dilation, stride=1):
    features = data.shape[-1]
    n_samples = (len(data) - dilation * (window_size - 1) - output_step)
    X = np.zeros((n_samples, window_size, features))
    features = len(data)

    for i in range(n_samples):
        for j in range(window_size):
            X[i][j] = (data[i + (j * dilation)])
    return X


def prepare_time_series_news_data(data, window_size, output_step, dilation, stride=1):
    features = data.shape[-1]
    n_samples = (len(data) - dilation * (window_size - 1) - output_step)
    X = np.zeros((n_samples, window_size, features))
    features = len(data)

    for i in range(n_samples):
        for j in range(window_size):
            X[i][j] = (data[i + (j * dilation)])
    return X
def prepare_time_series_news_raw_data(data, window_size, output_step, dilation, stride=1):
    features = data.shape[-1]
    n_samples = (len(data) - dilation * (window_size - 1) - output_step)
    X = np.empty((n_samples, window_size), dtype=object)
    features = len(data)

    for i in range(n_samples):
        for j in range(window_size):
            X[i][j] = (data[i + (j * dilation)])
    return X

def prepare_whether_data(stock_df, window_size, from_date, to_date, output_step, new_data=False):
    # Set the API key
    api_key = 'b7a439cb870a4a09be9114748230705'
    # Create the encoder
    encoder = OneHotEncoder(categories=[categories])

    # Set the search parameters
    query1 = "Ha Noi"
    query2 = "Ho Chi Minh"
    query3 = "Da Nang"
    # Define the from_date as the current date and time
    # if new_data:
    #     download_historical_whether(api_key, query1, from_date, to_date)
    #     download_historical_whether(api_key, query2, from_date, to_date)
    #     download_historical_whether(api_key, query3, from_date, to_date)

    # Create an empty list to store day arrays
    day_arrays = []

    # Iterate over all CSV files in the folder
    folder_path = './NLP/whether_data/'
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Load CSV file into DataFrame
            df = load_data_with_index(os.path.join(folder_path, filename), stock_df.index)
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
    whether_day_array = whether_day_array.reshape(whether_day_array.shape[1],
                                                  whether_day_array.shape[0] * whether_day_array.shape[2])

    output = prepare_time_series_whether_data(whether_day_array, window_size, output_step, 1)
    return output


def prepare_news_data(stock_df, symbol, window_size, from_date, to_date, output_step, topK, new_data=False):
    # Read the csv file
    # Get the index stock news save with stock dataframe
    # Get top 5 summary text
    # Merge into 1 text, preprocess
    # if merged text have lenght < max_input_lenght, add zero to it to meet the lenght
    # if larger, remove sentences until meet the lenght, than add zero
    # tokenize the text, convert tokens into ids
    # convert into (batch, 14, n) data
    max_string_lenght = 10
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    sentence_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    file_path = './NLP/news_data/' + symbol + "/" + symbol + "_" + "data.csv"
    merged_dataset_path = "./csv/" + symbol + "/" + symbol + "_" + "price_news.csv"
    news_query_folder = "./NLP/news_query"
    news_query_file_name = symbol + "_" + "query.json"
    news_query_path = news_query_folder + "/" + news_query_file_name
    with open(news_query_path, "r") as f:
        queries = json.load(f)
    keyword_query = list(queries.values())
    df = load_data_with_index(file_path, stock_df.index)
    top_sentences_dict = []
    for index in df.index:
        
        summary_columns = df.loc[index, "summary"]
        if isinstance(summary_columns, str):

            top_sentences = summary_columns[:max_string_lenght]
            ids, tokens = tokenize(top_sentences, tokenizer)
            top_sentences_dict.append(ids)
        else:
            top_sentences = get_similar_summary(summary_columns, queries, sentence_model, topK)
    max_length = max([len(lst) for lst in top_sentences_dict])
    # pad sequences to a fixed length
    padded_top_sentences_seq = pad_sequences(top_sentences_dict, maxlen=max_length, dtype="long",
                                             value=tokenizer.pad_token_id, truncating="post", padding="post")
    data = prepare_time_series_news_data(padded_top_sentences_seq, window_size, output_step, 1)

    return data

def prepare_raw_news_data(stock_df, symbol, window_size, from_date, to_date, output_step, topK, new_data=False):
    # Read the csv file
    # Get the index stock news save with stock dataframe
    # Get top 5 summary text
    # Merge into 1 text, preprocess
    # if merged text have lenght < max_input_lenght, add zero to it to meet the lenght
    # if larger, remove sentences until meet the lenght, than add zero
    # tokenize the text, convert tokens into ids
    # convert into (batch, 14, n) data
    max_string_lenght = 1000

    file_path = './NLP/news_data/' + symbol + "/" + symbol + "_" + "data.csv"
    news_query_folder = "./NLP/news_query"
    news_query_file_name = symbol + "_" + "query.json"
    news_query_path = news_query_folder + "/" + news_query_file_name
    with open(news_query_path, "r") as f:
        queries = json.load(f)
    keyword_query = list(queries.values())
    df = load_data_with_index(file_path, stock_df.index)
    top_sentences_dict = []
    for index in df.index:
        
        summary_columns = df.loc[index, "summary"]
        if isinstance(summary_columns, str):

            top_sentences = summary_columns[:max_string_lenght]
            top_sentences_dict.append(top_sentences)
    top_sentences_dict = np.array(top_sentences_dict)
    data = prepare_time_series_news_raw_data(top_sentences_dict, window_size, output_step, 1)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j] = f"string_{i}_{j}"

    # Split the strings into a third dimension
    array_3d = np.array([s.split("_") for s in data.flat]).reshape((data.shape[0], data.shape[1], -1))
    return array_3d
# define function to load csv file with given index
def load_data_with_index(csv_file, index):
    df = pd.read_csv(csv_file, index_col='date')
    df.index = pd.to_datetime(df.index)
    # merge the two dataframes on index
    df = df[df.index.isin(index)]
    df = df.set_index(index)
    return df


def tokenize(text, tokenizer):
    text = preprocess_text(text)
    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Remove stopwords
    stopwords = tokenizer.get_added_vocab()
    filtered_tokens = [token for token in tokens if token not in stopwords]

    # Perform lemmatization
    lemmatized_tokens = [tokenizer.decode(tokenizer.encode(token, add_special_tokens=False)) for token in
                         filtered_tokens]

    # Convert tokens to IDs
    token_ids = tokenizer.convert_tokens_to_ids(lemmatized_tokens)
    # Convert IDs back to text
    # processed_text = tokenizer.decode(token_ids)
    return token_ids, lemmatized_tokens


def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove non-text elements
    text = re.sub(r'[^\w\s.]', '', text)

    text = re.sub(r'[ \t]+(?=\n)', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text


def extract_sentences(text, queries):
    # Split the text into sentences

    nltk.download('punkt', quiet=True)
    sentences = nltk.sent_tokenize(text)

    # Create an empty list to store the matching sentences
    matching_sentences = []

    # Loop through each sentence
    for sentence in sentences:
        # Check if any of the query words are in the sentence
        if any(query.lower() in sentence for query in queries):
            # If a query word is found, add the sentence to the list of matching sentences
            matching_sentences.append(sentence)

    return matching_sentences


def get_similarity_score(sentence, queries, sentence_model):
    sentence_embeddings = sentence_model.encode(sentence)
    sim_list = []
    for query in queries:
        query_embeddings = sentence_model.encode(query).reshape(1, -1)
        similarity = cosine_similarity(sentence_embeddings, query_embeddings)
        sim_list.append(similarity)

    similarity = sum(sim_list) / len(sim_list)
    return similarity


def get_similar_sentences(paragraph, queries, sentence_model, threshold=0.1):
    if isinstance(paragraph, list):
        # Split the paragraph into individual sentences
        sentences = extract_sentences(paragraph, queries)

        # Calculate the similarity scores using vectorized operations
        # similarity_scores = np.array([get_similarity_score(sentence, queries, sentence_model) for sentence in sentences])
        similarity_scores = np.array(get_similarity_score(sentences, queries, sentence_model))

        sorted_indices = np.argsort(similarity_scores, axis=0)[::-1]
        sorted_sentences = sentences[sorted_indices]


        return sorted_sentences
    elif isinstance(paragraph, str):
        # Split the paragraph into individual sentences
        sentences = np.array(extract_sentences(paragraph, queries), dtype=str)
        if len(sentences) > 0:
            # Calculate the similarity scores using vectorized operations
            # similarity_scores = np.array([get_similarity_score(sentence, queries, sentence_model) for sentence in sentences])
            similarity_scores = np.array(get_similarity_score(sentences, queries, sentence_model))

            sorted_indices = np.argsort(similarity_scores, axis=0)[::-1]
            sorted_indices = sorted_indices.flatten().astype(int)
            sorted_sentences = sentences[sorted_indices]
            return sorted_sentences.tolist()
        else:
            return []

def get_similar_summary(summaries, queries, sentence_model, topK, threshold=0.1):
    # Calculate the similarity scores using vectorized operations
    if isinstance(summaries, list):
        similarity_scores = np.array(get_similarity_score(summaries, queries, sentence_model))
        top_sentences = [summaries[i] for i in np.argsort(similarity_scores)[::-1] if similarity_scores[i] > threshold]
        return top_sentences[:topK]
    elif isinstance(summaries, str):
        return summaries

# If there isn't any file in news_web_url/ + symbol /+ symbol_url.csv than:
# Download news from all source with from_Date of "2022-07-01" and to_date is Now
# Merge them with unique URL
# Save in the news_web_url/ + symbol /+ symbol_url.csv
# Else:
# read the file in news_web_url/ + symbol /+ symbol_url.csv
# to_date is Now, from_date is to_date - window_size
# concat the dataframe into the main csv file and filter by url
# while also return the dataframe of the window size news


def download_nyt_news(query, folder, from_date, to_date, page_size):
    # Subtract the timedelta object from the from_date to get the to_date
    # api_key = 'fa24ffdbf32f4feeb3ef755fad66a2bd'
    api_key = '8rcgDc65Yn8HCXG68vhA42bvawaKJ8xk'
    date_format = '%Y-%m-%d'
    if to_date is None:
        to_date = datetime.now()
        to_date = to_date.strftime(date_format)
    # else:
    #     to_date = datetime.strptime(to_date, date_format)

    # from_date = datetime.strptime(from_date, date_format)

    # Define an empty list to store the results
    results = []

    # Define a counter to keep track of the number of articles retrieved
    count = 0

    # Define the API endpoint
    endpoint = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'
    for page in range(0, page_size):
        # Define the query parameters
        params = {
            'q': query,
            'begin_date': from_date,
            'end_date': to_date,
            'page': page,
            'api-key': "fs5aF1Yb7hl5u7L3AZDrAAZn125qhxv8"
        }

        # Send the HTTP GET request to the API endpoint and retrieve the response
        response = requests.get(endpoint, params=params)

        # Check if the response was successful
        if response.status_code != 200:
            print(f"Error retrieving news articles: {response.text}")
            break

        # Parse the JSON response
        data = json.loads(response.text)

        # Check if there are no more articles to retrieve
        if not data['response']['docs']:
            break
        data = data['response']['docs']
        # Loop through the articles and extract the relevant information
        for article in data:
            # Check if the article has a valid publication date
            if 'pub_date' not in article or not article['pub_date']:
                continue

            # Convert the publication date string to a datetime object
            pub_date = datetime.strptime(article['pub_date'], '%Y-%m-%dT%H:%M:%S%z').date()
            # Check if the article's _id already exists in results
            article_id = article.get('_id')
            if article_id and any(a.get('_id') == article_id for a in results):
                continue
            # Add the relevant information to the results list
            results.append(article)

    # Save the results to a CSV file
    folder_path = "./news_data/" + folder + "/"
    filename = "nyt_news.csv"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Check if file with the given name exists in the folder
    file_path = os.path.join(folder_path, filename)
    if os.path.exists(file_path):
        # If it does, find an available numeric prefix
        i = 1
        while True:
            new_filename = f"{i}_{filename}"
            new_file_path = os.path.join(folder_path, new_filename)
            if not os.path.exists(new_file_path):
                break
            i += 1
        # Rename the file with the new prefixed filename
        os.rename(file_path, new_file_path)
        print(f"Renamed file {filename} to {new_filename}")
        # Update the file path to the new prefixed filename
        file_path = new_file_path
    # Save the articles to a CSV file
    df = pd.DataFrame.from_records(results)
    df = df.rename(columns={'pub_date': 'date'})
    df = df.rename(columns={'web_url': 'url'})
    # Convert the `datetime` column to a pandas datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Set the `date` column as the index
    df.set_index('date', inplace=True)
    df.to_csv(file_path, index=False)

def increase_n_days(date, n_days):
    date_obj = datetime.strptime(date, '%Y-%m-%d').date()
    delta = timedelta(days=n_days)
    date_obj += delta
    return date_obj.strftime('%Y-%m-%d')

def download_nyt_news(query, folder, from_date, to_date, page_size):
    # Subtract the timedelta object from the from_date to get the to_date
    # api_key = 'fa24ffdbf32f4feeb3ef755fad66a2bd'
    api_key = '8rcgDc65Yn8HCXG68vhA42bvawaKJ8xk'
    date_format = '%Y-%m-%d'
    if to_date is None:
        to_date = datetime.now()
        to_date = to_date.strftime(date_format)
    # else:
    #     to_date = datetime.strptime(to_date, date_format)

    # from_date = datetime.strptime(from_date, date_format)

    # Define an empty list to store the results
    results = []

    # Define a counter to keep track of the number of articles retrieved
    count = 0

    # Define the API endpoint
    endpoint = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'
    for page in range(0, page_size):
        # Define the query parameters
        params = {
            'q': query,
            'begin_date': from_date,
            'end_date': to_date,
            'page': page,
            'api-key': "fs5aF1Yb7hl5u7L3AZDrAAZn125qhxv8"
        }

        # Send the HTTP GET request to the API endpoint and retrieve the response
        response = requests.get(endpoint, params=params)

        # Check if the response was successful
        if response.status_code != 200:
            print(f"Error retrieving news articles: {response.text}")
            break

        # Parse the JSON response
        data = json.loads(response.text)

        # Check if there are no more articles to retrieve
        if not data['response']['docs']:
            break
        data = data['response']['docs']
        # Loop through the articles and extract the relevant information
        for article in data:
            # Check if the article has a valid publication date
            if 'pub_date' not in article or not article['pub_date']:
                continue

            # Convert the publication date string to a datetime object
            pub_date = datetime.strptime(article['pub_date'], '%Y-%m-%dT%H:%M:%S%z').date()
            # Check if the article's _id already exists in results
            article_id = article.get('_id')
            if article_id and any(a.get('_id') == article_id for a in results):
                continue
            # Add the relevant information to the results list
            results.append(article)

    # Save the results to a CSV file
    folder_path = "./news_data/" + folder + "/"
    filename = "nyt_news.csv"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Check if file with the given name exists in the folder
    file_path = os.path.join(folder_path, filename)
    if os.path.exists(file_path):
        # If it does, find an available numeric prefix
        i = 1
        while True:
            new_filename = f"{i}_{filename}"
            new_file_path = os.path.join(folder_path, new_filename)
            if not os.path.exists(new_file_path):
                break
            i += 1
        # Rename the file with the new prefixed filename
        os.rename(file_path, new_file_path)
        print(f"Renamed file {filename} to {new_filename}")
        # Update the file path to the new prefixed filename
        file_path = new_file_path
    # Save the articles to a CSV file
    df = pd.DataFrame.from_records(results)
    df = df.rename(columns={'pub_date': 'date'})
    df = df.rename(columns={'web_url': 'url'})
    # Convert the `datetime` column to a pandas datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Set the `date` column as the index
    df.set_index('date', inplace=True)
    df.to_csv(file_path, index=False)

def download_finhub_news(symbol, from_date, to_date, save_folder, new_data):

    filename = f'{symbol}_finhub_url.csv'
    filepath = os.path.join(save_folder, filename)
    if os.path.exists(filepath):
        if not new_data:
            print("File already exists. Skipping download.")
            df = pd.read_csv(filepath, index_col='date')
            return df
        else:
            stop_date = to_date.strptime(from_date, "%Y-%m-%d")
            from_date_dt = datetime.strptime(from_date, "%Y-%m-%d")

            final_df = pd.DataFrame()
            finnhub_client = finnhub.Client(api_key="chdrr1pr01qi6ghjsatgchdrr1pr01qi6ghjsau0")
            while from_date_dt <= stop_date:
                to_date = increase_n_days(from_date, 10)
                df = pd.DataFrame(finnhub_client.company_news(symbol, _from=from_date, to=to_date))
                df['datetime'] = df['datetime'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
                df = df.sort_values('datetime')
                df.drop_duplicates(subset='id', keep='first', inplace=True)
                final_df = pd.concat([final_df, df])
                from_date_dt = datetime.strptime(to_date, "%Y-%m-%d")
                from_date = from_date_dt.strftime('%Y-%m-%d')
            final_df.drop_duplicates(subset='id', keep='first', inplace=True)
            final_df = final_df.rename(columns={'datetime': 'date'})
            # Convert the `datetime` column to a pandas datetime format
            final_df['date'] = pd.to_datetime(final_df['date'])
            # # Set the `datetime` column as the index
            final_df.set_index('date', inplace=True)
            os.makedirs(save_folder, exist_ok=True)
            # Export the DataFrame to a CSV file
            final_df.to_csv(filepath, index=True)
            return final_df

def download_google_news(symbol, from_date, to_date, save_folder, new_data):

    filename = f'{symbol}_google_url.csv'
    filepath = os.path.join(save_folder, filename)
    if os.path.exists(filepath):
        if not new_data:
            print("File already exists. Skipping download.")
            df = pd.read_csv(filepath, index_col='date')
            return df
        else:
            # Subtract the timedelta object from the from_date to get the to_date
            # Create a new GoogleNews object
            googlenews = GoogleNews(lang='en')
            # Set the search query and time range
            googlenews.search(symbol)
            googlenews.set_time_range(from_date, to_date)
            # Retrieve the news articles
            articles = googlenews.result()


            df = pd.DataFrame.from_records(articles)
            df = df.drop(columns=['date'])
            df = df.rename(columns={'link': 'url'})
            df = df.rename(columns={'datetime': 'date'})
            df = df.rename(columns={'media': 'source'})
            # Convert the `datetime` column to a pandas datetime format
            df['date'] = pd.to_datetime(df['date'])
            # Convert 'date' to the format 'year-month-day'
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            # Set the `datetime` column as the index
            df.set_index('date', inplace=True)
            # Create the folder if it doesn't exist
            os.makedirs(save_folder, exist_ok=True)
            # Save the DataFrame to the specified path
            df.to_csv(filepath, index=True)
            return df
    else:
        # Subtract the timedelta object from the from_date to get the to_date
        # Create a new GoogleNews object
        googlenews = GoogleNews(lang='en')
        # Set the search query and time range
        googlenews.search(symbol)
        googlenews.set_time_range(from_date, to_date)
        # Retrieve the news articles
        articles = googlenews.result()


        df = pd.DataFrame.from_records(articles)
        df = df.drop(columns=['date'])
        df = df.rename(columns={'link': 'url'})
        df = df.rename(columns={'datetime': 'date'})
        df = df.rename(columns={'media': 'source'})
        # Convert the `datetime` column to a pandas datetime format
        df['date'] = pd.to_datetime(df['date'])
        # Convert 'date' to the format 'year-month-day'
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        # Set the `datetime` column as the index
        df.set_index('date', inplace=True)
        # Create the folder if it doesn't exist
        os.makedirs(save_folder, exist_ok=True)
        # Save the DataFrame to the specified path
        df.to_csv(filepath, index=True)
        return df

def download_alpha_vantage_news(symbol, from_date, to_date, save_folder, new_data):

    filename = f'{symbol}_alpha_vantage_url.csv'
    filepath = os.path.join(save_folder, filename)
    if os.path.exists(filepath):
        if not new_data:
            print("File already exists. Skipping download.")
            df = pd.read_csv(filepath, index_col='date')
            return df
        else:
            # Subtract the timedelta object from the from_date to get the to_date
            # Create a new AlphaVantage object or use the appropriate library

            # Set up the necessary parameters for Alpha Vantage API
            api_key = 'XOLA7URKCZHU7C9X'  # Replace with your actual API key
            api_endpoint = 'https://www.alphavantage.co/query'

            # Define the API parameters
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'sort': "EARLIEST",
                'limit': '5',
                'apikey': api_key
            }

            # Convert from_date to datetime object
            from_date_obj = datetime.strptime(from_date, '%Y-%m-%d')

            # Get the current date
            current_date_obj = datetime.strptime(to_date, '%Y-%m-%d')

            articles = []  # List to store all news articles

            # Loop through each day
            while from_date_obj <= current_date_obj:
                # Format the current date as time_from and time_to
                params['time_from'] = from_date_obj.strftime('%Y%m%dT%H%M')
                params['time_to'] = from_date_obj.strftime('%Y%m%dT%H%M')

                # Make the API request to get the daily news articles
                response = requests.get(api_endpoint, params=params)
                data = response.json()
                articles.extend(data['articles'])

                from_date_obj += timedelta(days=1)

            # Convert the articles list into a DataFrame
            df = pd.DataFrame.from_records(articles)
            df = df.drop(columns=['date'])
            df = df.rename(columns={'link': 'url'})
            df = df.rename(columns={'datetime': 'date'})
            df = df.rename(columns={'media': 'source'})
            # Convert the `date` column to a pandas datetime format
            df['date'] = pd.to_datetime(df['date'])
            # Convert 'date' to the format 'year-month-day'
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            # Set the `date` column as the index
            df.set_index('date', inplace=True)
            os.makedirs(save_folder, exist_ok=True)
            # Save the DataFrame to the specified path
            df.to_csv(filepath, index=True)
            return df
    else:
            # Subtract the timedelta object from the from_date to get the to_date
            # Create a new AlphaVantage object or use the appropriate library

            # Set up the necessary parameters for Alpha Vantage API
            api_key = 'XOLA7URKCZHU7C9X'  # Replace with your actual API key
            api_endpoint = 'https://www.alphavantage.co/query'

            # Define the API parameters
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'sort': "EARLIEST",
                'limit': '5',
                'apikey': api_key
            }

            # Convert from_date to datetime object
            from_date_obj = datetime.strptime(from_date, '%Y-%m-%d')

            # Get the current date
            current_date_obj = datetime.strptime(to_date, '%Y-%m-%d')

            articles = []  # List to store all news articles

            # Loop through each day
            while from_date_obj <= current_date_obj:
                # Format the current date as time_from and time_to
                params['time_from'] = from_date_obj.strftime('%Y%m%dT%H%M')
                params['time_to'] = from_date_obj.strftime('%Y%m%dT%H%M')

                # Make the API request to get the daily news articles
                response = requests.get(api_endpoint, params=params)
                data = response.json()
                articles.extend(data['articles'])

                from_date_obj += timedelta(days=1)

            # Convert the articles list into a DataFrame
            df = pd.DataFrame.from_records(articles)
            df = df.drop(columns=['date'])
            df = df.rename(columns={'link': 'url'})
            df = df.rename(columns={'datetime': 'date'})
            df = df.rename(columns={'media': 'source'})
            # Convert the `date` column to a pandas datetime format
            df['date'] = pd.to_datetime(df['date'])
            # Convert 'date' to the format 'year-month-day'
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            # Set the `date` column as the index
            df.set_index('date', inplace=True)
            os.makedirs(save_folder, exist_ok=True)
            # Save the DataFrame to the specified path
            df.to_csv(filepath, index=True)
            return df
def download_news(symbol, from_date, window_size, new_data = True):
    to_date = datetime.now().strftime('%Y-%m-%d')
    save_folder = f'./NPL/news_web_url/{symbol}/'
    file_name = f'{symbol}_url.csv'
    if not os.path.exists(save_folder + file_name):
        # Download news from all sources with the specified date range
        google_df = download_google_news(symbol, from_date, to_date, save_folder, new_data)
        finhub_df = download_finhub_news(symbol, from_date, to_date, save_folder, new_data)
        alphavantage_df = download_alpha_vantage_news(symbol, from_date, to_date, save_folder, new_data)

        # Merge the dataframes and save to CSV
        main_df = pd.concat([apinews_df, google_df, finhub_df, alphavantage_df])
        main_df.to_csv(save_path, index=False)
    else:
        # Read the existing CSV file
        main_df = pd.read_csv(save_path)

        # Set the new date range
        from_date = (datetime.strptime(to_date, '%Y-%m-%d') - timedelta(days=window_size)).strftime('%Y-%m-%d')

        # Download news from additional sources
        additional_df = download_additional_sources_news(from_date, to_date)

        # Concatenate the dataframes
        main_df = pd.concat([main_df, additional_df])

        # Filter the dataframe by URL
        window_df = main_df[main_df['URL'].isin(main_df['URL'].unique())]

        # Save the updated dataframe to the CSV file
        main_df.to_csv(save_path, index=False)

    return main_df.values