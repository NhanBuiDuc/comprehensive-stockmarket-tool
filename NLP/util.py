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

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    file_path = './NLP/news_data/' + symbol + "_" + "data.csv"
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
        top_sentences = get_similar_summary(summary_columns)[:topK]

        merged_sentences = " ".join(top_sentences)
        ids, tokens = tokenize(merged_sentences, tokenizer)
        top_sentences_dict.append(ids)

    max_length = max([len(lst) for lst in top_sentences_dict])
    # pad sequences to a fixed length
    padded_top_sentences_seq = pad_sequences(top_sentences_dict, maxlen=max_length, dtype="long", 
                                            value=tokenizer.pad_token_id, truncating="post", padding="post")
    data = prepare_time_series_news_data(padded_top_sentences_seq, window_size, output_step, 1)

    return data

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


def get_similarity_score(sentence, queries):
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    sentence_embeddings = model.encode(sentence).reshape(1, -1)
    sim_list = []
    for query in queries:
        query_embeddings = model.encode(query).reshape(1, -1)
        similarity = cosine_similarity(sentence_embeddings, query_embeddings)
        sim_list.append(similarity)

    similarity = sum(sim_list) / len(sim_list)
    return similarity


def get_similar_sentences(paragraph, queries):

    # Split the paragraph into individual sentences
    sentences = extract_sentences(paragraph, queries)

    # Calculate the similarity score between each sentence and the queries
    similarity_scores = []
    for sentence in sentences:
        similarity = get_similarity_score(sentence, queries)
        similarity_scores.append(similarity)

    # Sort the sentences by similarity score (only consider scores > 0.5) and return the top k
    top_sentences = [sentences[i] for i in
                     sorted(range(len(sentences)), key=lambda i: similarity_scores[i], reverse=True)
                     if similarity_scores[i] > 0.5]

    return top_sentences
def get_similar_summary(summaries, queries):

    # Calculate the similarity score between each sentence and the queries
    similarity_scores = []
    for summary in summaries:
        similarity = get_similarity_score(summary, queries)
        similarity_scores.append(similarity)

    # Sort the sentences by similarity score (only consider scores > 0.5) and return the top k
    top_sentences = [summaries[i] for i in
                     sorted(range(len(summaries)), key=lambda i: similarity_scores[i], reverse=True)
                     if similarity_scores[i] > 0.5]

    return top_sentences
