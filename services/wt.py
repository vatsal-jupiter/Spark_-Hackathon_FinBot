import json
from dotenv import load_dotenv
import requests
import os

load_dotenv()
SHARED_SECRET = os.getenv('SHARED_SECRET')
WT_URL = os.getenv('WT_URL')
print(WT_URL)

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
    url = f"{WT_URL}/wealth/v1/transactions?pageNumber={page_number}"
    print(url)
    response = requests.request("POST", url, headers=headers, data=json.dumps(wt_filters))
    data = response.json()
    transactions.extend(data["transactions"])
    print(data["pagination"])
    while data["pagination"]["totalRecords"] > len(transactions):
        print(data["pagination"])
        page_number+=1
        url = f"{WT_URL}/wealth/v1/transactions?pageNumber={page_number}"
        response = requests.request("POST", url, headers=headers, data=json.dumps(wt_filters))
        data = response.json()
        transactions.extend(data["transactions"])

    # print('transactions', transactions)
    return transactions


def get_insights_aggregates(user_id, filters: dict):
    wt_filters = to_wt_filters(filters)
    url = f"{WT_URL}/wealth/v1/insights/aggregate"
    headers = {
        'x-user-id': user_id,
        'X-App-Version': '3.10.0',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=json.dumps(wt_filters))
    return response.json()


def get_accounts_summary(user_id, products: list):
    url = f"{WT_URL}/wealth/v1/user-accounts/summary"

    payload = {
        "accountTypes": products,
    }
    headers = {
        'Content-Type': 'application/json',
        'X-App-Version': '3.10.0',
        'x-user-id': user_id,
        'x-jupiter-forwarded-shared-secret': SHARED_SECRET,
    }
    print(payload)
    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    print(response.content)
    return response.json()

# print(get_transactions('5ed1417f-bc66-4aeb-8a60-c7564e4964f9', {
#     'startDate': '2024-10-01',
#     'endDate': '2024-10-01'
# }))

# print(get_insights_aggregates('5ed1417f-bc66-4aeb-8a60-c7564e4964f9', {
#     "filters": {
# 		"startDate": "2024-10-01",
# 		"endDate": "2024-10-31",
# 		"aggregateField": "CATEGORY",
# 		"products": ["JUPITER"]
# 	}
# }))

# print(get_accounts_summary('5ed1417f-bc66-4aeb-8a60-c7564e4964f9', ["JUPITER"]))