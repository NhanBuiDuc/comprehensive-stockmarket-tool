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

    # Convert datetime objects to strings and remove articles with NaN datetime
    # for article in articles:
    #     try:
    #         # Check if the datetime object is valid
    #         article["datetime"] = article["datetime"].strftime('%Y-%m-%d %H:%M:%S')
    #     except:
    #         # Delete the datetime key if the value is invalid
    #         del article["datetime"]
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
        df.to_csv(file_path)

if __name__ == "__main__":
    # Set the API key
    api_key = 'fa24ffdbf32f4feeb3ef755fad66a2bd'

    # Init the NewsApiClient
    newsapi = NewsApiClient(api_key=api_key)

    # Set the search parameters
    query1 = "Vietnam agriculture cofffe rice corn"
    query2 = "vietnam+agriculture+(soybeans OR corn OR wheat flour OR barley OR sugarcane OR tobacco OR jute OR cotton OR shrimp OR fish OR squid OR octopus OR crab OR chicken OR duck OR turkey OR beef OR pork OR buffalo meat OR milk OR cheese OR yogurt OR honey OR silk OR roses OR chrysanthemums OR carrots OR cucumbers OR tomatoes OR apples OR oranges OR grapes OR almonds OR walnuts OR pecans OR cardamom OR cloves OR ginger OR ginseng OR turmeric OR peppermint OR lemongrass)"
    # Define the from_date as the current date and time
    from_date = datetime.now()
    delta = timedelta(days=14)
    page_size = 100
    total_results = 1000
    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    # download_news(query1, from_date, delta, page_size, total_results)
    # download_news(query2, from_date, delta, page_size, total_results)
    download_google_news(query1, from_date, delta, page_size, total_results)
    # Read JSON file and convert to dictionary
    folder_path = "./NLP/news_data"

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if the file is a regular file and has a ".json" extension
        if os.path.isfile(file_path) and filename.endswith(".json"):
            # Open the file and load its contents into a dictionary
            with open(file_path, "r") as json_file:
                json_data = json_file.read()
                results = json.loads(json_data)
            # Process the dictionary as needed
            for article in results:
                url = article["link"]
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
                    chunk = full_page[i:i+1000]
                    print(summarizer(chunk))