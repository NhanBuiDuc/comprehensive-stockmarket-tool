from newsapi import NewsApiClient
from datetime import datetime, timedelta
from transformers import pipeline
import json
from bs4 import BeautifulSoup
import requests


def download_news(query, sources, from_date, delta, page_size, total_results):
    # Subtract the timedelta object from the from_date to get the to_date
    to_date = from_date - delta
    from_date_str = from_date.strftime('%Y-%m-%d')
    to_date_str = to_date.strftime('%Y-%m-%d')
    # Define an empty list to store the results
    results = []

    # Loop over the API request until we have the desired number of articles
    while len(results) < total_results:
        # Set the page parameter for the API request
        page = int(len(results) / page_size) + 1

        # Send the API request
        articles = newsapi.get_everything(q=query, sources=sources, from_param=from_date, to=to_date, page=page,
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
    print(results)
    # Save dictionary as a JSON file
    with open("news_json/news.json", "w") as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    # Set the API key
    api_key = 'fa24ffdbf32f4feeb3ef755fad66a2bd'

    # Init the NewsApiClient
    newsapi = NewsApiClient(api_key=api_key)

    # Set the search parameters
    query = 'apple'
    sources = 'bbc-news, bloomberg, business-insider, cnbc, cnn, financial-post, reuters, the-wall-street-journal, the-washington-post, time'
    # Define the from_date as the current date and time
    from_date = datetime.now()
    delta = timedelta(days=3)
    page_size = 1000
    total_results = 1000
    #
    # download_news(query, sources, from_date, delta, page_size, total_results)
    # Read JSON file and convert to dictionary
    with open("./NLP/news_json/news.json", "r") as json_file:
        json_data = json_file.read()
        results = json.loads(json_data)
    #
    # sentiment_analysis = pipeline("sentiment-analysis")
    # Loop over each article in the results list
    for article in results:
        url = article["url"]
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
    print(full_page)
    print("done")