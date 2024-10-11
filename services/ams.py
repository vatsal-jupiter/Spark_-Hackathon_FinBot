import os
from dotenv import load_dotenv
import json
import requests
load_dotenv()
AMS_URL = os.getenv('AMS_URL')
fields_to_extract = ['accountType', 'product', 'status', 'displayName', 'balance', 'nextAccountRefreshIn', 'accountRefreshFailedAt']

def get_user_accounts(user_id: str):
    url = f"{AMS_URL}/ams/account/search"

    payload = {
        "userId": user_id
    }
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))

    accounts = response.json()['accounts']

    result = []
    for account in accounts:
        if account['deleteFlag']:
            continue
        metadataJson = account['metadataJson']
        result.append({
            'accountNumber':metadataJson['maskedAccountNumber'],
            'bank':metadataJson['bank'],
            'balance':metadataJson['currentBalance'],
            'accountType':metadataJson['accountType'],
            'branch':metadataJson['branch'],
            'ifscCode':metadataJson['ifscCode'],
            'nextAccountRefreshInHours': metadataJson['nextAccountRefreshIn'],
            'isJupiterAccount': False

        })
    return result
