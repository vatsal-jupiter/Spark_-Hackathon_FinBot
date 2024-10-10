import json
from dotenv import load_dotenv
import requests
import os

load_dotenv()
SHARED_SECRET = os.getenv('SHARED_SECRET')
BUDGETING_URL = os.getenv('BUDGETING_URL')


def to_filters(filters):
    return filters


def get_budget_instances(user_id: str, filters: dict):
    headers = {
        'x-user-id': user_id,
        'X-APP-Version': '3.10.0',
        'Content-Type': 'application/json',
        'x-jupiter-forwarded-shared-secret': SHARED_SECRET
    }

    filters = to_filters(filters)
    url = f"{BUDGETING_URL}/ubs/v1/budget/instances"
    response = requests.request("POST", url, headers=headers, data=json.dumps(filters))
    return response.json()

# print(get_budget_instances('5ed1417f-bc66-4aeb-8a60-c7564e4964f9', {
#     "startDate": "2024-10-01",
#     "endDate": "2024-10-31",
#     "includeBudgetUtilisedAmount": True
# }))