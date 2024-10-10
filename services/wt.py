import json
from dotenv import load_dotenv
import requests
import os

load_dotenv()
SHARED_SECRET = os.getenv('SHARED_SECRET')
WT_URL = os.getenv('WT_URL')


def to_wt_filters(filters):
    return filters


def get_transactions(user_id: str, filters: dict):
    transactions = []

    headers = {
        'x-user-id': user_id,
        'X-APP-Version': '3.10.0',
        'Content-Type': 'application/json',
        'x-jupiter-forwarded-shared-secret': SHARED_SECRET
    }

    wt_filters = to_wt_filters(filters)
    page_number = 1
    url = f"http://{WT_URL}/wealth/v1/transactions?pageNumber={page_number}"
    response = requests.request("POST", url, headers=headers, data=json.dumps(wt_filters))
    data = response.json()
    transactions.extend(data["transactions"])
    print(data["pagination"])
    while data["pagination"]["totalRecords"] > len(transactions):
        print(data["pagination"])
        page_number+=1
        url = f"http://localhost:11000/wealth/v1/transactions?pageNumber={page_number}"
        response = requests.request("POST", url, headers=headers, data=json.dumps(wt_filters))
        data = response.json()
        transactions.extend(data["transactions"])

    return transactions


def get_insights_aggregates(user_id, filters: dict):
    wt_filters = to_wt_filters(filters)
    url = f"http://{WT_URL}/wealth/v1/insights/aggregate"
    headers = {
        'x-user-id': user_id,
        'X-App-Version': '3.10.0',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=json.dumps(wt_filters))
    return response.json()
