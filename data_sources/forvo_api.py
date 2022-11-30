import requests
import time
import json
import os
from google.cloud import storage

FORVO_API_KEY = os.environ['FORVO_API_KEY']
BUCKET_NAME   = os.environ['BUCKET_NAME']

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)


def top1000_words() -> list:
    """Read top1000_words.txt on GCP bucket, return a list of words"""
    b = bucket.blob('forvo_api/top1000_words.txt')
    with b.open("r") as f:
        return [ line.rstrip() for line in f.readlines() ]


# need further testing
def api_call(words: list = []):
    """call forvo api and save response to gcp bucket"""
    forvo_api_url = 'https://apifree.forvo.com'
    key = 'key/d60fccb5217bf6ee726ab45e85ae91cd'
    format = 'format/json'
    language = 'language/en'  # https://forvo.com/languages-codes/

    # gets all the pronunciations from a word.
    # https://api.forvo.com/documentation/word-pronunciations/
    action = 'action/word-pronunciations'
    country = 'country/gbr'  # https://en.wikipedia.org/wiki/ISO_3166-1
    order = 'order/rate-desc' # date-desc, date-asc, rate-desc, rate-asc
    limit = 'limit/50'
    
    for word in words[:400]:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        url = f'{forvo_api_url}/{key}/{format}/{action}/word/{word}/{language}/{country}/{order}/{limit}'
        json_file = f'forvo_api/json/{timestamp}-{word}.json'
        blob = bucket.blob(json_file)
        response = requests.get(url, timeout=10).json()
        return response
        with blob.open("w") as f:
               f.write(response)
