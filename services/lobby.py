import json
from dotenv import load_dotenv
import requests
import os

load_dotenv()
SHARED_SECRET = os.getenv('SHARED_SECRET')
LOBBY_URL = os.getenv('LOBBY_URL')


def to_filters(filters):
    return filters


def to_payload(response):
    data = response.json()
    return {
        'accountNumber': data['prospectiveUserInfo']['accountNumber'],
        'ifsc': data['prospectiveUserInfo']['ifsc'],
    }

    # return {
    #     'accountNumber': data['prospectiveUserInfo'][0]['accountNumber'],,
    #     'bank': 'Jupiter',
    #     'balance': ,
    #     'accountType':metadataJson['accountType'],
    #     'branch':metadataJson['branch'],
    #     'ifscCode':metadataJson['ifscCode'],
    #     'nextAccountRefreshInHours': metadataJson['nextAccountRefreshIn'],
    #     'isJupiterAccount': False
    # }


def get_upcoming_transactions(user_id: str):
    url = f"{LOBBY_URL}/lobby/v1/prospectiveuser/08134773-3a20-4525-a6bf-182846dcb93b?infosets=jupiter.infosets.customeraccount"

    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)
    return to_payload(response)

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