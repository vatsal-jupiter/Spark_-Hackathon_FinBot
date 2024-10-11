import json
from dotenv import load_dotenv
import requests
import os

load_dotenv()
SHARED_SECRET = os.getenv('SHARED_SECRET')
MCS_URL = os.getenv('MCS_URL')


def to_filters(filters):
    return filters


def get_upcoming_transactions(user_id: str):
    headers = {
        'x-user-id': user_id,
        'X-APP-Version': '3.10.0',
        'Content-Type': 'application/json',
    }

    filters = {
        "startDate": "2024-10-01",
        "endDate": "2024-10-30",
        "recurrenceStatus": [
            "PENDING",
            "DONE"
        ],
        "patternStatus": [
            "ACTIVE",
            "UNSET"
        ]
    }
    url = f"{MCS_URL}/mcs/v1/recurrences?source=MONEY_CALENDAR"
    response = requests.request("POST", url, headers=headers, data=json.dumps(filters))
    print("upcoming-trans"+response.text)
    fields_to_consider = ['payee', 'amount', 'category', 'isAutopay', 'dueOn', 'recurrenceStatus', 'lastTransactionDateTime']

    data = response.json()['recurrences']
    recs = [{field: value for field, value in item.items() if field in fields_to_consider} for item in data]
    print("recs", recs)
    return recs

# print(get_upcoming_transactions('5ed1417f-bc66-4aeb-8a60-c7564e4964f9', {
#     "startDate": "2024-09-01",
#     "endDate": "2024-09-30",
#     "recurrenceStatus": [
#         "PENDING",
#         "DONE"
#     ],
#     "patternStatus": [
#         "ACTIVE",
#         "UNSET"
#     ]
# }))