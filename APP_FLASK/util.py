import numpy as np
from sentence_transformers import SentenceTransformer, util
import json
import pandas as pd
import util as u
from NLP import util as nlp_u
def prepare_stock_data(data, window_size):
    features = data.shape[-1]
    n_samples = (len(data) - (window_size - 1))
    X = np.zeros((n_samples, window_size, features))
    for i in range(n_samples):
        for j in range(window_size):
            X[i][j] = (data[i + (j)])
    timeseries_price = X[:, :6]
    timeseries_stock = X[:, 6:]
    return timeseries_price, timeseries_stock

def prepare_news_data(stock_df, symbol, window_size, topK, new_data=False):
    # Read the csv file
    # Get the index stock news save with stock dataframe
    # Get top 5 summary text
    # Merge into 1 text, preprocess
    # if merged text have lenght < max_input_lenght, add zero to it to meet the lenght
    # if larger, remove sentences until meet the lenght, than add zero
    # tokenize the text, convert tokens into ids
    # convert into (batch, 14, n) data
    max_string_lenght = 50
    model_name = "bert-base-uncased"
    sentence_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    file_path = './NLP/news_data/' + symbol + "/" + symbol + "_" + "data.csv"
    news_query_folder = "./NLP/news_query"
    news_query_file_name = symbol + "_" + "query.json"
    news_query_path = news_query_folder + "/" + news_query_file_name
    with open(news_query_path, "r") as f:
        queries = json.load(f)
    keyword_query = list(queries.values())
    # df = load_data_with_index(file_path, stock_df.index)
    news_df = pd.read_csv(file_path)
    news_df['date'] = pd.to_datetime(news_df['date'])
    news_df = nlp_u.filter_news_by_stock_dates(news_df, stock_df, max_rows=5)
    top_sentences_dict = []
    # news_df.reset_index(drop=True, inplace=True)
    for date in news_df["date"].unique():
        summmary_columns = news_df[news_df["date"] == date]["summary"]
        top_summary = nlp_u.get_similar_summary(summmary_columns.values, keyword_query, sentence_model, topK, 0.5)
        flattened_array = np.ravel(np.array(top_summary))
        merged_summary = ' '.join(flattened_array)
        ids = nlp_u.sentence_tokenize(merged_summary, sentence_model)
        top_sentences_dict.append(ids)

    top_sentences_dict = np.array(top_sentences_dict)
    data = top_sentences_dict[-window_size:, :]
    return data