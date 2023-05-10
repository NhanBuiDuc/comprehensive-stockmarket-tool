from newsapi import NewsApiClient
from datetime import datetime, timedelta
from transformers import pipeline
import json
from bs4 import BeautifulSoup
import requests
import os
import math
from GoogleNews import GoogleNews
import pandas as pd

import requests
import json
from datetime import datetime, timedelta
import csv
import finnhub


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

def download_news(query, from_date, delta, page_size, total_results):
    # Subtract the timedelta object from the from_date to get the to_date
    to_date = from_date - delta
    from_date_str = from_date.strftime('%Y-%m-%d')
    to_date_str = to_date.strftime('%Y-%m-%d')
    # Define an empty list to store the results
    results = []

    # Set the page parameter for the API request
    page = int(len(results) / page_size) + 1

    # Send the API request
    articles = newsapi.get_everything(q=query, language="en", from_param=from_date, to=to_date, page=page,
                                      page_size=page_size)

    # Check if the API request was successful
    if articles['status'] == 'ok':
        # Add the articles to the results list
        results += articles['articles']
    else:
        # Handle the error
        print('Error:', articles['message'])

    # Print the total number of articles retrieved
    print('Total articles retrieved:', len(results))
    """
    JSON includes source, author, content, 
    """
    # Print the results
    folder_path = "./NLP/news_data/"
    filename = "news.json"

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
    else:
        # Open the file (either the original or the new prefixed one) in write mode
        with open(file_path, "w") as json_file:
            # Write data to the file
            json.dump(results, json_file)


def download_google_news(query, from_date, delta, page_size, total_results):
    # Subtract the timedelta object from the from_date to get the to_date
    to_date = from_date - delta
    from_date_str = from_date.strftime('%Y-%m-%d')
    to_date_str = to_date.strftime('%Y-%m-%d')

    # Create a new GoogleNews object
    googlenews = GoogleNews(lang='en')
    # Set the search query and time range
    googlenews.search(query)
    googlenews.set_time_range(from_date_str, to_date_str)
    # Define an empty list to store the results
    results = []

    # Retrieve the news articles
    articles = googlenews.result()

    folder_path = "./NLP/news_data/"
    filename = "news.csv"

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
    else:
        # Open the file (either the original or the new prefixed one) in write mode

        # Save the articles to a JSON file
        # with open(file_path, 'w', encoding='utf-8') as json_file:
        #     json.dump(articles, json_file, ensure_ascii=False, indent=4)
        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame.from_records(articles)

        # Convert the `datetime` column to a pandas datetime format
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Set the `datetime` column as the index
        df.set_index('datetime', inplace=True)

        # Export the DataFrame to a CSV file
        df.to_csv(file_path, index=True)

def download_finnhub_news(query, folder, from_date, to_date, symbol):
    finnhub_client = finnhub.Client(api_key='chdrr1pr01qi6ghjsatgchdrr1pr01qi6ghjsau0')
    df = pd.DataFrame(finnhub_client.company_news('AAPL', _from="2022-06-01", to="2023-05-10"))

if __name__ == "__main__":
    # Set the API key
    api_key = 'fa24ffdbf32f4feeb3ef755fad66a2bd'

    # Init the NewsApiClient
    newsapi = NewsApiClient(api_key=api_key)

    # Set the search parameters
    query1 = "APPLE"
    # Define the from_date as the current date and time
    from_date = "2015-01-01"
    to_date = None
    delta = timedelta(days=14)
    page_size = 100
    total_results = 1000
    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    # download_news(query1, from_date, delta, page_size, total_results)
    # download_news(query2, from_date, delta, page_size, total_results)
    # download_google_news(query1, from_date, delta, page_size, total_results)
    # Read JSON file and convert to dictionary
    # download_nyt_news(query1, query1, from_date, to_date, page_size)
    folder_path = "./news_data"

    # Loop through all the subfolders in the main folder
    for subdir_name in os.listdir(folder_path):
        # Get the full path of the subfolder
        subdir_path = os.path.join(folder_path, subdir_name)

        # Check if the subfolder is actually a folder
        if os.path.isdir(subdir_path):
            # Loop through all the files in the subfolder
            for filename in os.listdir(subdir_path):
                # Get the full path of the file
                file_path = os.path.join(subdir_path, filename)
                if os.path.isfile(file_path) and filename.endswith(".csv"):
                    # Open the file and load its contents into a dictionary
                    with open(file_path, 'r') as csv_file:
                        csv_reader = csv.DictReader(csv_file)
                        for row in csv_reader:
                            url = row["url"]
                            response = requests.get(url)
                            html_content = response.content
                            # Parse the HTML content using BeautifulSoup
                            soup = BeautifulSoup(html_content, 'html.parser')
                            # Parse the HTML content using BeautifulSoup
                            # Find all the <p> tags in the document
                            p_tags = soup.find_all('p')
                            full_page = ""
                            # Print the content of each <p> tag
                            for p in p_tags:
                                full_page += p.text + " "

                            for i in range(0, len(full_page), 1000):
                                chunk = full_page[i:i + 1000]
                                print(summarizer(chunk))
