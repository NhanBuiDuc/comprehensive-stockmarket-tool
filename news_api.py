from datetime import datetime, timedelta
from transformers import pipeline, AutoTokenizer, AutoModel
import re
import json
from bs4 import BeautifulSoup
import requests
import os
import math

import pandas as pd

import requests
import json
from datetime import datetime, timedelta
import csv
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from NLP import util as u
from newsplease import NewsPlease
from sentence_transformers import SentenceTransformer
from configs.config import config as cf
untrustworthy_url = ["https://www.zac.com",
                     "https://www.thefly.com",
                     "https://www.investing.com",
                     'https://investorplace.com',
                     'https://www.nasdaq.com'
                     ]
trustworthy_url = ["zac.com",
                   "guru.com",
                   "Investing.com",
                   ]
trustworthy_source = ["CNBC", "GuruFocus"
                      ]
untrustworthy_source = ["Nasdaq", "Thefly.com", "Yahoo"]

if __name__ == "__main__":

    # Set the search parameters
    # Define the from_date as the current date and time
    from_date = "2022-07-01"
    to_date = None
    page_size = 100
    total_results = 1000
    topK = 5
    window_size = cf["data"]["window_size"]
    max_summary_lenght = 60
    symbol = "MSFT"
    news_web_url_folder = "./NLP/news_web_url"
    news_web_file_name = news_web_url_folder + f'/{symbol}/{symbol}_url.csv'
    news_data_path = "./NLP/news_data/" + symbol + "/" + symbol + "_" + "data.csv"
    news_query_folder = "./NLP/news_query"
    news_query_file_name = symbol + "_" + "query.json"
    news_query_path = news_query_folder + "/" + news_query_file_name
    with open(news_query_path, "r") as f:
        queries = json.load(f)
    keyword_query = list(queries.values())
    model_name = 'philschmid/bart-large-cnn-samsum'
    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    sentence_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    u.download_news(symbol, from_date=from_date, window_size=window_size)

    dataframes_to_concat = []
    file_encoding = 'ISO-8859-1'
    df = pd.read_csv(news_web_file_name, encoding=file_encoding)
    # filtered_df = df[df['source'].isin(trustworthy_source)]
    df = df[~df["source"].isin(untrustworthy_source)]
    df = df[:10]
    for index, row in df.iterrows():
        # summary = row["summary"]
        # summary = u.preprocess_text(summary, tokenizer)
        # if u.get_similarity_score(summary, keyword_query) > 0.0:
        url = row["url"]
        source = row["source"]
        date = row["date"]
        top_sentences_str = ""
        try:
            response = requests.get(url, timeout=20)
            base_url = urlparse(response.url).scheme + '://' + urlparse(response.url).hostname

            # if source in trustworthy_source:
            if base_url not in untrustworthy_url:
                article = NewsPlease.from_url(response.url)
                if article is not None:
                    if article.maintext is not None:
                        # Preprocess the input text and the query
                        full_text = u.preprocess_text(article.maintext)
                        top_sentence = u.get_similar_sentences(full_text, keyword_query,
                                                               sentence_model)[:5]
                        if len(top_sentence) > 0:
                            summary_top_sentence = summarizer(top_sentence)
                            # summary_top_sentence = summary_top_sentence[0]["summary_text"]
                            unique_summaries = []
                            for summary in summary_top_sentence:
                                # Check if this summary is unique
                                if summary not in unique_summaries:
                                    unique_summaries.append(summary["summary_text"])

                            # Merge the unique summaries into a single string
                            summary_top_sentence = " ".join(unique_summaries)
                            print(date)
                            print(base_url)
                            print(source)
                            print(summary_top_sentence)
                            summary_df = pd.DataFrame({
                                'date': date,
                                'symbol': symbol,
                                'source': source,
                                'summary': summary_top_sentence,
                                "base_url": base_url,
                                "url": response.url
                            }, index=[index])
                            dataframes_to_concat.append(summary_df)
        except Exception as e:
            print("An exception occurred:", e)
            print(date)
            print(base_url)
            print(source)
    # Concatenate all the DataFrames into one
    dataframe = pd.concat(dataframes_to_concat)
    # Set the `datetime` column as the index
    dataframe.set_index('date', inplace=True)
    # Export the DataFrame to a CSV file
    os.makedirs(os.path.dirname(news_data_path), exist_ok=True)
    dataframe.to_csv(news_data_path, index=True)
