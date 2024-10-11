import json
from dotenv import load_dotenv
import requests
import os
from collections import defaultdict
# from services.ams import get_user_accounts

load_dotenv()
SHARED_SECRET = os.getenv('SHARED_SECRET')
WT_URL = os.getenv('WT_URL')
fields_to_extract = ['accountType', 'product', 'status', 'displayName', 'balance', 'nextAccountRefreshIn', 'accountRefreshFailedAt']
fields_naming_map = {
    'nextAccountRefreshIn': 'nextAccountRefreshInHours'
}
# print(WT_URL)

def to_wt_filters(filters):
    new_filters = {}
    # print(filters)
    if 'credit_debit_indicator' in filters:
        new_filters['creditDebitIndicator'] = filters['credit_debit_indicator'][0]
    if 'product' in filters:
        products = []
        if 'JUPITER' in filters['product']:
            products.append('JUPITER')
        if 'Edge VISA card' in filters['product']:
            products.append('EDGE_CARD')
        if 'Edge Rupay card' in filters['product']:
            products.append('BLAZE')
        non_ada = ['JUPITER', 'BLAZE', 'EDGE_CARD']
        difference = [item for item in filters['product'] if item not in non_ada]
        if len(difference) > 0:
            products.append('ADA')
        new_filters['product'] = ','.join(products)
        if 'ALL' in filters['product']:
            new_filters['product'] = 'ALL'
    if 'transaction_date_range' in filters:
        start_date = filters['transaction_date_range'].start.date()
        end_date = filters['transaction_date_range'].end.date()
        new_filters['startDate'] = str(start_date)
        new_filters['endDate'] = str(end_date)
    if 'coarse_grain_category' in filters:
        new_filters['category'] = ','.join(filters['coarse_grain_category'])

    new_filters["reconStatus"] = 'RECONCILIATION_CBS_INITIATED,RECONCILIATION_JUPITER_SUCCESS,RECONCILIATION_SUCCESSFUL,RECONCILIATION_JUPITER_INITIATED'
    new_filters["shouldIncludeUPICardsTPAP"] = False
    new_filters["shouldIncludeSavingsAccountTPAP"] = True
    new_filters["excludeFailedTransactions"] = False
    return new_filters



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
    print("wt_filters",wt_filters)
    response = requests.request("POST", url, headers=headers, data=json.dumps(wt_filters))

    data = response.json()
    transactions.extend(data["transactions"])
    while data["pagination"]["totalRecords"] > len(transactions):
        page_number+=1
        url = f"{WT_URL}/wealth/v1/transactions?pageNumber={page_number}"
        response = requests.request("POST", url, headers=headers, data=json.dumps(wt_filters))
        if page_number >= 50:
            break
        print(f'resp {page_number}', response.content)
        data = response.json()
        transactions.extend(data["transactions"])

    # sum_by_A = defaultdict(int)
    # for item in transactions:
    #     sum_by_A[item['category']] += item['transactionAmount']
    # print(sum_by_A)
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


def get_accounts_summary(user_id, products=None):
    if products is None:
        products = ['JUPITER', 'ADA']
    url = f"{WT_URL}/wealth/v1/user-accounts/summary"

    payload = {
        "accountTypes": products
    }
    headers = {
        'Content-Type': 'application/json',
        'X-App-Version': '3.10.0',
        'x-user-id': user_id,
        'x-jupiter-forwarded-shared-secret': SHARED_SECRET
    }

    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    data = response.json()
    accounts = data['accounts']
    modified_data = [{ field: item[field] for field in fields_to_extract } for item in accounts]
    for account in modified_data:
        for key in list(account):
            if key in fields_naming_map:
                value = account[key]
                new_field = fields_naming_map[key]
                account[new_field] = value
    # print(modified_data)
    return modified_data

# def get_user_account_details(user_id, products=None):
#     return get_user_accounts(user_id)


# print(get_user_account_details('08134773-3a20-4525-a6bf-182846dcb93b'))

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