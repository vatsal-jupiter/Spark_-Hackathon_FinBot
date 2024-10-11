#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from dotenv import load_dotenv
import os
from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated

from services.mcs import get_upcoming_transactions
from services.wt import get_transactions, get_accounts_summary

WT_URL = os.getenv('WT_URL')
# In[2]:


load_dotenv()

# In[3]:


# os.environ["OPENAPI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
# os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# In[4]:


# import pandas as pd

# df = pd.read_csv("./vatsal_txn_data.csv")
# print(df.shape)
# print(df.columns.tolist())


# In[5]:


# df.groupby(['transactionChannel','transactiontype']).size()


# In[6]:


# from langchain_openai import ChatOpenAI
# chat_model = ChatOpenAI(model="gpt-4o-mini")


# In[7]:


os.environ["AZURE_OPENAI_API_KEY"] = "991aaf91f70042f79584ae6a32330c3d"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ds-genai-gpt4o-hackathon.openai.azure.com/"

end_point = "https://Meta-Llama-3-1-405B-ds-jupiter.eastus2.models.ai.azure.com/v1/chat/completions"
api_key = "had1kdX8avzidelf04sAr4pVYIgybOcG"

from langchain_community.chat_models.azureml_endpoint import (
    AzureMLEndpointApiType,
    CustomOpenAIChatContentFormatter,
)

from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint
meta_chat_model = AzureMLChatOnlineEndpoint(
    endpoint_url=end_point,
    endpoint_api_type=AzureMLEndpointApiType.serverless,
    endpoint_api_key=api_key,
    content_formatter=CustomOpenAIChatContentFormatter(),
)

from langchain_openai import AzureChatOpenAI

chat_model = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini-2",  # or your deployment
    api_version="2023-03-15-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# In[8]:


# chat_model.invoke("Hi")


# In[9]:


from langchain_core.messages import ToolMessage
from langchain_core.pydantic_v1 import ValidationError


class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    def respond(self, state):
        response = []
        for attempt in range(3):
            response = self.runnable.invoke(
                state, {"tags": [f"attempt:{attempt}"]}
            )
            try:
                self.validator.invoke(response)
                return {"intermediate_steps": [response]}
            except ValidationError as e:
                state["intermediate_steps"] = state["intermediate_steps"] + [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\n\nPay close attention to the function schema.\n\n"
                                + self.validator.schema_json()
                                + " Respond by fixing all validation errors.",
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]
        return {"intermediate_steps": [response]}


# In[10]:


from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder


def create_agent(chat_model: ChatOpenAI, tools: list, system_prompt: str, input_variables = []):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    prompt.input_variables += input_variables

    agent = create_openai_tools_agent(chat_model, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor


# In[11]:


# Converting to executable node
def agent_node(state, agent, name):
    # result = agent.invoke(state)
    result = agent.invoke({"messages": state["messages"], 'user_id': state['user_id']})
    return {"messages": [AIMessage(content=result["output"], name=name)]}


# ### Data fetcher

# In[12]:


transactionChannel = ['MANDATE', 'RTGS', 'NEFT', 'IMPS', 'DEBIT_CARD', 'CREDIT_CARD', 'UPI', 'ATM']
bank_names = ["State Bank of India", "HDFC Bank", "Kotak Mahindra Bank", "Bank of Baroda", "Axis Bank", "ICICI Bank",
              "IDFC First Bank",
              "Federal Bank", "Union Bank", "Canara Bank", "Punjab National Bank", "IndusInd Bank", "Bank of India",
              "Yes Bank",
              "AU Small Finance Bank", "Indian Overseas Bank", "UCO Bank", "Karnataka Bank", "Karur Vysya Bank",
              "Punjab and Sind Bank",
              "Indian Bank", "Bank of Maharashtra", "Central Bank of India"]
credit_cards = ["Edge VISA card", "Edge Rupay card"]
# existing_products = ['JUPITER', 'ADA','CREDIT_CARD']
existing_products = ['JUPITER'] + bank_names + credit_cards
creditDebitIndicator = ['DEBIT', 'CREDIT']
coarsegrain_category = ['food & drinks', 'pots', 'transfers', 'groceries', 'rent', 'pots withdrawal', 'miscellaneous',
                        'atm', 'credit', 'uncategorised',
                        'bills & utilities', 'charity', 'shopping', 'medical', 'credit bills', 'entertainment',
                        'travel', 'commute', 'personal care',
                        'investments', 'fuel', 'refund', 'home services', 'family & pets',
                        'money transfers', 'edge card bill', 'investments',
                        'interest', 'household', 'loans', 'rewards']
'''spends_coarsegrain_category = ['food & drinks', 'transfers', 'groceries', 'rent', 'miscellaneous', 'pots withdrawal',
                        'atm', 'credit', 'uncategorised',
                        'bills & utilities', 'charity', 'shopping', 'medical', 'credit bills', 'entertainment',
                        'travel', 'commute', 'personal care',
                        'fuel', 'refund', 'home services', 'family & pets',
                        'money transfers', 'edge card bill',
                        'interest', 'household', 'loans', 'rewards']
investment_coarsegrain_category = []'''


# In[13]:


# from pydantic import BaseModel, Field, validator
# from typing import List, Literal, Optional, Tuple
# from datetime import datetime, timedelta

# class AmountRange(BaseModel):
#     min: Optional[float] = Field(
#         default=None,
#         description="The minimum transaction amount for filtering. Can be None."
#     )
#     max: Optional[float] = Field(
#         default=None,
#         description="The maximum transaction amount for filtering. Can be None."
#     )


# class DateRange(BaseModel):
#     start: Optional[datetime] = Field(
#         default_factory=lambda: datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
#         description="The start date for filtering transactions. Defaults to today's date."
#     )
#     end: Optional[datetime] = Field(
#         default_factory=datetime.now,
#         description="The end date for filtering transactions. Defaults to the current time if not provided."
#     )


# class TransactionData(BaseModel):
#     transaction_channel: List[Literal[*transactionChannel]] = Field(
#         default=transactionChannel,
#         description="The channels through which the transaction was made."
#     )
#     product: List[Literal[*existing_products]] = Field(
#         default=existing_products,
#         description="The products associated with the transaction. Can be a JUPITER or external bank (ADA)"
#     )
#     credit_debit_indicator: List[Literal[*creditDebitIndicator]] = Field(
#         default=creditDebitIndicator,
#         description="Indicators whether the transactions are debits or credits."
#     )
#     coarse_grain_category: List[Literal[*coarsegrain_category]] = Field(
#         default=coarsegrain_category,
#         description="Categories for the transactions like food & drinks, entertainment, rewards etc."
#     )
#     transactionAmount_range: Optional[AmountRange] = Field(
#         default=AmountRange(),
#         description="An object containing 'min' and 'max' fields representing the minimum and maximum transaction amounts for filtering. Both can be None."
#     )
#     transaction_date_range: Optional[DateRange] = Field(
#         default=DateRange(),
#         description="An object containing 'start' and 'end' fields representing the minimum and maximum transaction dates for filtering. If not provided, the default is from 6 months ago to the current time."
#     )
#     merchant: Optional[str] = Field(
#         default=None,
#         description="A regex pattern to filter transactions based on the merchant's name. If not provided, no filtering is applied."
#     )

#     @validator('merchant', pre=True, always=True)
#     def lowercase_merchant(cls, merchant_val):
#         if merchant_val is not None:
#             return merchant_val.lower()
#         return merchant_val


# In[14]:


from pydantic import BaseModel, Field, validator
from typing import List, Literal, Optional, Tuple
from datetime import datetime, timedelta


class AmountRange(BaseModel):
    min: Optional[float] = Field(
        default=None,
        description="The minimum transaction amount for filtering. Can be None."
    )
    max: Optional[float] = Field(
        default=None,
        description="The maximum transaction amount for filtering. Can be None."
    )


class DateRange(BaseModel):
    start: Optional[datetime] = Field(
        default_factory=lambda: datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        description="The start date for filtering transactions. Defaults to today's date."
    )
    end: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="The end date for filtering transactions. Defaults to the current time if not provided."
    )


class TransactionData(BaseModel):
    transaction_channel: List[Literal[*transactionChannel]] = Field(
        default=transactionChannel,
        description="The channels through which the transaction was made."
    )
    product: List[Literal[*existing_products]] = Field(
        default=existing_products,
        description="The products associated with the transaction. Can be a JUPITER or external bank (ADA)"
    )
    credit_debit_indicator: List[Literal[*creditDebitIndicator]] = Field(
        default=creditDebitIndicator,
        description="Indicators whether the transactions are debits or credits."
    )
    coarse_grain_category: List[Literal[*coarsegrain_category]] = Field(
        default=coarsegrain_category,
        description="Categories for the transactions like food & drinks, entertainment, rewards etc."
    )
    transactionAmount_range: Optional[AmountRange] = Field(
        default=AmountRange(),
        description="An object containing 'min' and 'max' fields representing the minimum and maximum transaction amounts for filtering. Both can be None."
    )
    transaction_date_range: Optional[DateRange] = Field(
        default=DateRange(),
        description="An object containing 'start' and 'end' fields representing the minimum and maximum transaction dates for filtering. If not provided, the default is from 6 months ago to the current time."
    )


# In[15]:


# from langchain_core.output_parsers.openai_tools import PydanticToolsParser
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# import datetime

# txn_data_fetcher_system_prompt = """
# You are expert finncial advisor working for jupiter (1-app for everything money). Below is the decription about jupiter's product.
# Edge Rupay card: It's a Credit Card launched by jupiter in partnership with CSB Bank and RuPay network.
# Edge VISA card: It's a Credit Card launched by jupiter in partnership with Federal Bank and VISA network.


# You are an agent designed to filter and get the transaction data based on specific criteria. Your role is to assist in processing
# financial transactions by applying filters such as transaction channels, product types, credit or debit indicators,
# coarse-grain categories, transaction amount ranges, and transaction date ranges.

# Filtering Criteria:
# Transaction Channel: One or more of the available channels such as 'MANDATE', 'RTGS', 'NEFT', etc.
# Product: Specific product types such as 'JUPITER', 'HDFC Bank', 'ICICI Bank', 'Edge Rupay card', etc.
# Credit/Debit Indicator: Whether the transaction is a 'DEBIT' or 'CREDIT'.
# Coarse-Grain Category: Transaction categories like 'food & drinks', 'groceries', 'rent', etc.
# Transaction Amount Range: Filter transactions by specifying a minimum and/or maximum amount. If not provided, consider all transaction amounts.
# Transaction Date Range: Filter transactions based on a specific date and time range, with minute-level precision. If the date range is not provided, default is today's transactions
# Merchant: Will be a regex, that will be applied to filter out based on merchant

# Alternative terms you may find in question:
# spend : Credit/Debit Indicator -> DEBIT
# external banks: exclude JUPITER, Edge Rupay card, Edge VISA card from Product
# non jupiter: exclude JUPITER, Edge Rupay card, Edge VISA card from Product

# keep this information in mind, think step by step on users question and Invoke the {function_name} with complete accuracy in applying the necessary
# filters as arguments to the function.
# This is a current time, use it if require: {time}
# """


# txn_data_fetcher_prompt_template = ChatPromptTemplate.from_messages(
#     [
#         ("system",txn_data_fetcher_system_prompt),
#         MessagesPlaceholder(variable_name="messages")
#     ]
# ).partial(
#     time=lambda: datetime.datetime.now().isoformat(), function_name=TransactionData.__name__
# )


# In[16]:


from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime

txn_data_fetcher_system_prompt = """
You are expert finncial advisor working for jupiter (1-app for everything money). Below is the decription about jupiter's product.
Edge Rupay card: It's a Credit Card launched by jupiter in partnership with CSB Bank and RuPay network.
Edge VISA card: It's a Credit Card launched by jupiter in partnership with Federal Bank and VISA network.
Pots: Pots is an jupiter's product designed for investment purpose for users. Pots and Investments categories are pertaining to investment transactions of the user.
You are an agent designed to filter and get the transaction data based on specific criteria. Your role is to assist in processing
financial transactions by applying filters such as transaction channels, product types, credit or debit indicators,
coarse-grain categories, transaction amount ranges, and transaction date ranges.
Filtering Criteria:
Transaction Channel: One or more of the available channels such as 'MANDATE', 'RTGS', 'NEFT', etc.
Product: Specific product types such as 'JUPITER', 'HDFC Bank', 'ICICI Bank', 'Edge Rupay card', etc. 'JUPITER' indicates transactions happening jupiter's saving account.
Credit/Debit Indicator: Whether the transaction is a 'DEBIT' or 'CREDIT'.
Coarse-Grain Category: Transaction categories like 'food & drinks', 'groceries', 'rent', etc.
Transaction Amount Range: Filter transactions by specifying a minimum and/or maximum amount. If not provided, consider all transaction amounts.
Transaction Date Range: Filter transactions based on a specific date and time range, with minute-level precision. If the date range is not provided, default is today's transactions
Alternative terms you may find in question:
spend : Credit/Debit Indicator -> DEBIT
external banks: exclude JUPITER, Edge Rupay card, Edge VISA card from Product
non jupiter: exclude JUPITER, Edge Rupay card, Edge VISA card from Product
exclude cards: exclude Edge Rupay card, Edge VISA card from Product
keep this information in mind, think step by step on users question and Invoke the {function_name} with complete accuracy in applying the necessary
filters as arguments to the function.
This is a current time, use it if require: {time}
"""

txn_data_fetcher_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", txn_data_fetcher_system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(), function_name=TransactionData.__name__
)

# In[17]:


txn_data_fetcher_chain = txn_data_fetcher_prompt_template | chat_model.bind_tools(tools=[TransactionData])

validator = PydanticToolsParser(tools=[TransactionData])

txn_data_fetcher = ResponderWithRetries(runnable=txn_data_fetcher_chain, validator=validator)


# In[18]:


# from langchain_core.messages import HumanMessage, ToolMessage

# example_question = "how much did I spent on swiggy starting from feb 2024 to today ?"
# initial = txn_data_fetcher.respond({"messages":[HumanMessage(content=example_question)]})
# import json
# retrospect = initial['intermediate_steps'][0].additional_kwargs['tool_calls'][0]['function']
# print(json.loads(retrospect['arguments']))


# In[19]:


# from langchain_core.messages import HumanMessage, ToolMessage

# example_question = "What is the amount debited via UPI in last two months?"
# initial = txn_data_fetcher.respond({"messages":[HumanMessage(content=example_question)]})
# import json
# retrospect = initial['intermediate_steps'][0].additional_kwargs['tool_calls'][0]['function']
# print(json.loads(retrospect['arguments']))


# In[20]:


# from langchain_core.messages import HumanMessage, ToolMessage

# example_question = "compare my spending of May and July?"
# initial = txn_data_fetcher.respond({"messages":[HumanMessage(content=example_question)]})
# import json
# retrospect = initial['intermediate_steps'][0].additional_kwargs['tool_calls'][0]['function']
# print(json.loads(retrospect['arguments']))
# retrospect = initial['messages'][0].additional_kwargs['tool_calls'][1]['function']
# print(json.loads(retrospect['arguments']))


# ### Get Txn Data Tool

# In[ ]:


# In[21]:


def get_merchant(row):
    if row['creditDebitIndicator'] == "DEBIT":
        return row["payee"]
    else:
        return row["payer"]


def data_preprocessing(df):
    df['merchant'] = df.apply(get_merchant, axis=1)
    df['merchant'] = df['merchant'].str.lower()

    col_int = ['transactionDateTime', 'transactionAmount', 'transactionChannel', 'product',
               'category',
               'creditDebitIndicator', 'merchant']
    df = df[col_int]
    df['transactionDateTime'] = pd.to_datetime(df['transactionDateTime'], format='ISO8601', utc=True).dt.tz_convert('Asia/Kolkata')
    return df


# In[ ]:


# In[22]:


from langchain.tools import tool


@tool
def get_txn_data_tool(filters: TransactionData, user_id: str):
    """
    Get the transaction data for user with the provided filters
    """
    if "DEBIT" in filters.credit_debit_indicator:
        if len(filters.coarse_grain_category) <= 2 and "investment" in filters.coarse_grain_category:
            pass
        else:
            filters.coarse_grain_category.remove("investment")
            filters.coarse_grain_category.remove("pots")
    df = pd.DataFrame(get_transactions(user_id, filters.__dict__))
    # print('user-transactions', df.to_dict())
    df = data_preprocessing(df)
    # Return the result as a JSON object
    return df

def upcomming_txns_preprocessing(df):
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    current_date = datetime.now()
    df['dueOn'] = pd.to_datetime(df['dueOn'])
    df['dueOn'] = df['dueOn'].apply(lambda x: x + relativedelta(months=1) if current_date > x else x)
    df["dueOn"] = df["dueOn"].astype(str)
    df = df[["payee","amount","dueOn"]]
    def format_transaction(row):
        return f":date: **You have an upcoming payment of ₹{row['amount']} to {row['payee']} on {row['dueOn']}!** :money_with_wings:\nMake sure to mark your calendar.\n"
    formatted_transactions = '\n'.join(df.apply(format_transaction, axis=1).tolist())
    return formatted_transactions

def get_upcoming_txn_data_tool(state):
    """
    Get the upcoming transaction data for user with the provided filters
    """
    # hardcoding the filters for upcoming transactions
    user_id = state["user_id"]
    df = pd.DataFrame(get_upcoming_transactions(user_id))
    # print('user-transactions', df.to_dict())
    response = upcomming_txns_preprocessing(df)
    # Return the result as a JSON object

    return {"messages": [AIMessage(content=response)]}

# In[23]:


from langgraph.prebuilt.tool_executor import ToolExecutor

tools = [get_txn_data_tool]
tool_executor = ToolExecutor(tools)

# In[24]:


from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

description_generation_template = """
I am using one tool with my AI agent. Below is the argument schema of the function:
{argument_schema}

This tool will return dataframe of transaction data but it will apply filter based on arguments
provided below:
{tool_argumnts}

your job is to generate accurate description in 50-100 words about transaction data present in the dataframe, 
that description will be used by large language model to get an information about kind of data present in the dataframe. 
"""

description_generation_prompt_template = PromptTemplate(template=description_generation_template,
                                                        input_variables=['tool_argumnts']).partial(
    argument_schema=TransactionData.schema()
)

description_generator = description_generation_prompt_template | chat_model | StrOutputParser()

# In[25]:


# initial


# In[26]:


from langgraph.prebuilt import ToolInvocation
import json


def invoke_get_txn_data_tool(state):
    tool_calls = state['intermediate_steps'][-1].additional_kwargs.get("tool_calls", [])
    df_list = []
    for tool_call in tool_calls:
        tool_details = tool_call
        function_name = get_txn_data_tool.name
        args = tool_details.get("function").get("arguments")
        print(args)
        dataframe_description = description_generator.invoke({"tool_argumnts": args})
        action = ToolInvocation(tool=function_name, tool_input={'filters': json.loads(args), 'user_id': state['user_id']})
        response = tool_executor.invoke(action)
        dataframe_update = {
            "dataframe": response,
            "description": dataframe_description
        }
        df_list.append(dataframe_update)
    return {"dataframe_store": df_list, "intermediate_steps": [AIMessage(content=args)]}


# In[ ]:


# In[27]:


# temp_args = {"transaction_date_range": {"start": "2024-05-01T00:00:00", "end": "2024-05-31T23:59:59"}}
# out = description_generator.invoke({"tool_argumnts":temp_args})


# In[28]:


# print(out)


# In[29]:


# invoke_get_txn_data_tool(initial)


# In[ ]:


# ### Answer Query

# In[30]:


# txn_data_fetcher_system_prompt = """
# You are expert finncial advisor working for jupiter (1-app for everything money). Below is the decription about jupiter's product.
# Edge Rupay card: It's a Credit Card launched by jupiter in partnership with CSB Bank and RuPay network.
# Edge VISA card: It's a Credit Card launched by jupiter in partnership with Federal Bank and VISA network.


# You are an agent designed to filter and get the transaction data based on specific criteria. Your role is to assist in processing
# financial transactions by applying filters such as transaction channels, product types, credit or debit indicators,
# coarse-grain categories, transaction amount ranges, and transaction date ranges.

# Filtering Criteria:
# Transaction Channel: One or more of the available channels such as 'MANDATE', 'RTGS', 'NEFT', etc.
# Product: Specific product types such as 'JUPITER', 'HDFC Bank', 'ICICI Bank', 'Edge Rupay card', etc.
# Credit/Debit Indicator: Whether the transaction is a 'DEBIT' or 'CREDIT'.
# Coarse-Grain Category: Transaction categories like 'food & drinks', 'groceries', 'rent', etc.
# Transaction Amount Range: Filter transactions by specifying a minimum and/or maximum amount. If not provided, consider all transaction amounts.
# Transaction Date Range: Filter transactions based on a specific date and time range, with minute-level precision. If the date range is not provided, default is today's transactions
# Merchant: Will be a regex, that will be applied to filter out based on merchant

# Alternative terms you may find in question:
# spend : Credit/Debit Indicator -> DEBIT
# external banks: exclude JUPITER, Edge Rupay card, Edge VISA card from Product
# non jupiter: exclude JUPITER, Edge Rupay card, Edge VISA card from Product

# keep this information in mind, think step by step on users question and Invoke the {function_name} with complete accuracy in applying the necessary
# filters as arguments to the function.
# This is a current time, use it if require: {time}
# """


# In[31]:


txn_data_fetcher_system_prompt = """
You are expert finncial advisor working for jupiter (1-app for everything money). Below is the decription about jupiter's product.
Edge Rupay card: It's a Credit Card launched by jupiter in partnership with CSB Bank and RuPay network.
Edge VISA card: It's a Credit Card launched by jupiter in partnership with Federal Bank and VISA network.


You are an agent designed to filter and get the transaction data based on specific criteria. Your role is to assist in processing 
financial transactions by applying filters such as transaction channels, product types, credit or debit indicators, 
coarse-grain categories, transaction amount ranges, and transaction date ranges.

Filtering Criteria:
Transaction Channel: One or more of the available channels such as 'MANDATE', 'RTGS', 'NEFT', etc.
Product: Specific product types such as 'JUPITER', 'HDFC Bank', 'ICICI Bank', 'Edge Rupay card', etc.
Credit/Debit Indicator: Whether the transaction is a 'DEBIT' or 'CREDIT'.
Coarse-Grain Category: Transaction categories like 'food & drinks', 'groceries', 'rent', etc.
Transaction Amount Range: Filter transactions by specifying a minimum and/or maximum amount. If not provided, consider all transaction amounts.
Transaction Date Range: Filter transactions based on a specific date and time range, with minute-level precision. If the date range is not provided, default is today's transactions


Alternative terms you may find in question:
spend : Credit/Debit Indicator-> DEBIT
external banks: exclude JUPITER, Edge Rupay card, Edge VISA card from Product
non jupiter: exclude JUPITER, Edge Rupay card, Edge VISA card from Product

keep this information in mind, think step by step on users question and Invoke the {function_name} with complete accuracy in applying the necessary 
filters as arguments to the function.
This is a current time, use it if require: {time}
"""

# In[32]:


import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonAstREPLTool
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.messages import AIMessage


def analyse_transaction_data(state):
    dataframe_dict = {}
    dataframe_description = ""
    for i in range(len(state["dataframe_store"])):
        data = state["dataframe_store"][i]
        dataframe_dict["df_" + str(i + 1)] = data["dataframe"]
        prompt = f""" 

        data in df_{i + 1} describes as follows:
        {data["description"]}

        """
        dataframe_description = dataframe_description + prompt

    analyse_data_system_prompt = """
    You are expert in writing code, you are working for jupiter (1-app for everything money). Below is the decription about jupiter's product.
    Edge Rupay card: It's a Credit Card launched by jupiter in partnership with CSB Bank and RuPay network.
    Edge VISA card: It's a Credit Card launched by jupiter in partnership with Federal Bank and VISA network.
    Pots: Pots is an jupiter's product designed for investment purpose for users. Pots and Investments categories are pertaining to investment transactions of the user.
    You have access to a pandas dataframes. Below is the head of the dataframes in markdown format:
    {markdown}
    below are the set of columns availables:
    {columns}
    Description about data of each dataframe availabe:
    {dataframe_description}
    Given a user question, write the Python code to answer it. \
    Return ONLY the valid Python code and nothing else. \
    Don't assume you have access to any libraries other than built-in Python ones and pandas.
    write down a python code in such way that it will print final answer with proper description about what you are printing.
    use only one print statement and print all the final answer with with proper description.
    """

    df = state["dataframe_store"][0]["dataframe"]
    df_head = str(df.head().to_markdown())
    df_columns = str(df.columns.tolist())

    code_tool = PythonAstREPLTool(globals=dataframe_dict)
    llm_with_tools = chat_model.bind_tools([code_tool], tool_choice=code_tool.name)
    parser = JsonOutputKeyToolsParser(key_name=code_tool.name, first_tool_only=True)

    # print(dataframe_description)

    for attempt in range(3):
        # analyser_prompt = PromptTemplate(template = analyse_data_prompt_template,
        #                                                        input_variables = ['question']).partial(
        #     markdown= df_head, columns = df_columns, dataframe_description = dataframe_description
        # )

        analyser_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", analyse_data_system_prompt),
                MessagesPlaceholder(variable_name="messages")
            ]
        ).partial(
            markdown=df_head, columns=df_columns, dataframe_description=dataframe_description
        )

        chain = analyser_prompt | llm_with_tools | parser  # | code_tool
        # generated_code = chain.invoke({"question": state["messages"][-1]})
        generated_code = chain.invoke(state)
        print("*****************************")
        print(generated_code)
        # print("*****************************")
        response = code_tool.invoke(generated_code)
        # print("*****************************")
        # print(response)
        # print("*****************************")
        if 'Error' not in response:
            state["intermediate_steps"] = state["intermediate_steps"] + [AIMessage(content=generated_code["query"])]
            break
        # print("====== Error Occured ======")
        generated_code = generated_code["query"].replace("{", "{{").replace("}", "}}")
        analyse_data_system_prompt = analyse_data_system_prompt + f"""
        \n\n below is your code: \n {str(generated_code)}
        \n\n Error occured, look at the error details provided below and make correction in code: \n {str(response)}
        """
        # print(analyse_data_system_prompt)
        # analyse_data_system_prompt = analyse_data_system_prompt + f"""
        # \n\n Error occured, look at the error details provided below and make correction in code: \n {str(response)}
        # """
    return {"messages": [AIMessage(content=response)]}


# In[ ]:


# ### General Purpose Agnet

# In[ ]:


# In[33]:


from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)

# In[34]:


from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()


class PythonREPLInput(BaseModel):
    command: str = Field(
        description="A valid Python command to execute. If you want to see the output of a value, you should print it out with `print(...)`.")


code_tool = StructuredTool.from_function(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command.",
    func=python_repl.run,
    input_schema=PythonREPLInput
)

# In[ ]:


# In[35]:


system_prompt = """
You are a general purpose agent to search financial data and give financial advice.
search the web if required to answer users query.
use the code tool if you need to execute code and for data plotting using matplotlib.
As part of response, Don't include your introduction. Be specific & crisp.
"""

# In[36]:


tools = [tavily_tool, code_tool]
research_agent = create_agent(chat_model, tools, system_prompt)

# In[37]:


import functools

research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# In[ ]:


# ### Jupiter Info

# In[38]:


from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool


# In[39]:


@tool("account-deatils-tool")
def get_account_details(user_id: str):
    """
    Get the personal information and account information of the user.
    """

    account_details = get_accounts_summary(user_id)

    # Return the result as a JSON object
    return json.dumps(account_details)


# In[40]:


system_prompt = """
You have an access if user's personal and account information. also, you have an information about 
products of juiter (1-app for everything money)
You will get an access to user's personal and account information through account-deatils-tool.
Call the tool with the given user id: {user_id}
"""

# In[41]:


tools = [get_account_details]
jupiter_info_agent = create_agent(chat_model, tools, system_prompt, ['user_id'])

# In[42]:


import functools

jupiter_info_node = functools.partial(agent_node, agent=jupiter_info_agent, name="jupiter_info")

# In[ ]:


# In[ ]:

# ### Spends Insights

# In[43]:


import requests

def get_response(url, headers, data):
    response = requests.request("POST", url, headers=headers, data=data)
    return json.loads(response.text)


# In[44]:


from calendar import monthrange

def update_payload(current_month):
    # global payload_dict
    payload_dict= {
        'payload_aggregate': {
            'filters': {
                'startDate': '2024-07-01',
                'endDate': '2024-08-31',
                'aggregateField': 'CATEGORY',
                'products': ['JUPITER', 'ADA']
            }
        }
    }
    from datetime import datetime
    current_month_date = datetime.strptime(current_month, "%Y-%m")

    start_date = current_month_date.replace(day=1).strftime("%Y-%m-%d")
    last_day = monthrange(current_month_date.year, current_month_date.month)[1]
    end_date = current_month_date.replace(day=last_day).strftime("%Y-%m-%d")

    payload_dict['payload_aggregate']['filters']['startDate'] = start_date
    payload_dict['payload_aggregate']['filters']['endDate'] = end_date
    return payload_dict


# In[45]:


def extract_data(data):
    categories = []
    amount = []
    cat_percentage = []
    credit_debit_indicators = []

    this_month_data = data['aggregateData']
    for entry in this_month_data:
        credit_debit = entry['creditDebitIndicator']
        for field in entry['aggregateFields']:
            categories.append(field['name'])
            amount.append(field['amount'])
            cat_percentage.append(field['percentage'])
            credit_debit_indicators.append(credit_debit)

    df = pd.DataFrame({
        'category': categories,
        'creditDebitIndicator': credit_debit_indicators,
        'amount': amount,
        'category%': cat_percentage
    })
    return df


# In[46]:


import json
import pandas as pd

def get_spend_insights_for_user(user_id, current_month,past_month):
    #1. get all data needed

    #1a: get or set API query params and call API
    url_aggregate=f"{WT_URL}/wealth/v1/insights/aggregate"
    payload_dict = update_payload(current_month)
    payload_aggregate = json.dumps(payload_dict["payload_aggregate"])
    headers_aggregate = {
        'Content-Type': 'application/json',
        'X-App-Version': '2.4.5',
        'x-user-id': user_id
    }
    current_aggregate= get_response(url_aggregate , headers_aggregate, payload_aggregate)
    #print(current_aggregate)
    payload_dict = update_payload(past_month)
    payload_aggregate = json.dumps(payload_dict["payload_aggregate"])
    headers_aggregate = {
        'Content-Type': 'application/json',
        'X-App-Version': '2.4.5',
        'x-user-id': user_id
    }
    past_aggregate= get_response(url_aggregate , headers_aggregate, payload_aggregate)
    #print(past_aggregate)
    #1b: <extract_data_method> to get final DF
    data_current = extract_data(current_aggregate)
    data_past = extract_data(past_aggregate)
    data = pd.merge(data_current, data_past, on=["category", "creditDebitIndicator"], suffixes=('_this_month', '_last_month'), how='outer')
    data = data.fillna(0)
    #2: pre-process data for calling LLM
    data['creditDebitIndicator'] = pd.Categorical(data['creditDebitIndicator'], categories=['CREDIT', 'DEBIT'], ordered=True)
    data = data.sort_values('creditDebitIndicator')
    data = data[data['creditDebitIndicator'] == 'DEBIT']
    data['category'] = data['category'].str.replace("credit bills", "credit card bills")
    data_markdown = data.to_markdown(index=False)
    return data_markdown


# In[ ]:





# In[ ]:





# In[47]:


from pydantic import BaseModel, Field, validator
from typing import List, Literal, Optional, Tuple
from langchain.output_parsers import PydanticOutputParser

class WealthTabMonth(BaseModel):
    current_month: str = Field(pattern=r"^\d{4}-\d{2}$",
                               description="Current month in 'YYYY-MM' format, e.g., '2024-08'"
                               )

current_month_parser = PydanticOutputParser(pydantic_object=WealthTabMonth)


# In[48]:


# This is a current time, use it if require: {time}


# In[49]:


from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime

current_month_system_prompt = """
Your job is to generate a month and a year.
Write your answer in given format: \n {format_instructions}
"""

current_month_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",current_month_system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ]
).partial(
    format_instructions=current_month_parser.get_format_instructions()
)


# In[50]:


current_month_chain = current_month_prompt_template | chat_model | current_month_parser

# from langchain_core.messages import HumanMessage
# temp = current_month_chain.invoke({"messages":[HumanMessage(content = "Give me a spends insights for my august, 2024 transaction data")]})


# In[51]:


def get_current_month(state):
    response = current_month_chain.invoke(state)
    return {"current_month":response.current_month}


# In[ ]:





# In[ ]:





# In[52]:


from pydantic import BaseModel, Field, validator
from typing import List, Literal, Optional, Tuple
from langchain.output_parsers import PydanticOutputParser

class CategoryReasoning(BaseModel):
    Category: str = Field(
        description="Identify categories where you could save more money."
    )
    savings_explanation: str = Field(
        description = "Explain for category, why do you think there are saving oppertunities"
    )

class Reasoning(BaseModel):
    reasoning: List[CategoryReasoning] = Field(
        description = "List of categories and and reaoning on why and how there is a saving oppertunities"
    )

category_reasoning_parser = PydanticOutputParser(pydantic_object=Reasoning)


# In[ ]:





# In[53]:


from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime

system_prompt = """
You are expert finncial advisor working for jupiter (1-app for everything money) who has a experties in how to maximize savings by 
reducing unneccessary spends and by finding cheaper alternative options.

Below is the decription about jupiter's product:
edge card: It's a Credit Card launched by jupiter in partnership with CSB Bank and RuPay network.

Below you have been provided with category-wise aggregated information for current and previous month in the dataframe in markdown format. 
Traverse through this dataframe, compare the spends and % of each category for this month to the previous month to provide insights into 
spending trends and category-wise changes.
all the mentioned amounts are in Indian National Rupee (INR),
look at the data carefully:
{markdown}

Think in step by step as mentioned below inorder to maximize savings for the user:
Step 1: Identify categories where you could save more money.
Step 2: explain for every category why do you think there are savings oppertunities

keep this information in mind, think step by step on provided information.
Write your answer in given format: \n {format_instructions}
"""

categoey_reasoning_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt)
    ]
).partial(
    format_instructions=category_reasoning_parser.get_format_instructions()
)


# In[54]:


maximize_savings_chain = categoey_reasoning_prompt_template | chat_model | category_reasoning_parser


# In[55]:


#user_id = "01c503a7-17bf-42d2-9705-26e6e84c2c9a"
# previous_month = "2024-07"
# current_month = "2024-08"


# In[ ]:





# In[56]:


def get_savings_strategies(state):
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    current_month = state["current_month"]
    current_month_date = datetime.strptime(current_month, "%Y-%m")
    previous_month_date = current_month_date - relativedelta(months=1)
    previous_month = previous_month_date.strftime("%Y-%m")
    #print("----- month ---->",current_month, previous_month)

    markdown = get_spend_insights_for_user(state["user_id"], current_month, previous_month)
    out = maximize_savings_chain.invoke({"markdown":markdown})
    reasoning = out.dict()['reasoning']
    return {"reasoning":reasoning, "current_index": -1}


# In[60]:


from datetime import timedelta

def tag_festival_day(transaction_date):
    festival_df = pd.read_csv('festivals.csv')
    festival_df['Date'] = pd.to_datetime(festival_df['Date'] + ' 2024', format='%B %d %Y')
    festival_df = festival_df.drop_duplicates(subset='Date', keep='last')
    # Iterate through the global festival_df
    for _, row in festival_df.iterrows():
        festival_day = row['Date']

        # Define pre-festival, festival day, and post-festival ranges
        pre_festival_start = festival_day - timedelta(days=5)
        post_festival_end = festival_day + timedelta(days=2)

        # Check if the transaction_date falls in pre-festival, festival day, or post-festival
        if pre_festival_start <= transaction_date < festival_day:
            return 'pre festival day'
        elif transaction_date == festival_day:
            return 'festival day'
        elif festival_day < transaction_date <= post_festival_end:
            return 'post festival day'

    # If none of the conditions match, it's a normal day
    return 'normal day'


# In[61]:


import pandas as pd
from datetime import timedelta


# Create the salary date for each transaction month
def get_salary_date(row):
    # Get the year and month from the transaction date
    year = row['transactiondatetime'].year
    month = row['transactiondatetime'].month
    # year = transactiondatetime.year
    # month = transactiondatetime.month

    # If salary_day is greater than the number of days in the month, handle month end cases
    try:
        salary_date = pd.Timestamp(year=year, month=month, day=row['salary_day'])
    except ValueError:
        # If salary day doesn't exist in the month (e.g. 30th in February), use the last day of the month
        salary_date = pd.Timestamp(year, month, pd.Timestamp(year, month, 1).days_in_month)

    return salary_date

def tag_pay_day(df ,user_id):
    # Merge the two dataframes on user_id
    #df = pd.merge(df, salary_date, on='user_id', how='left')
    salary_date = pd.DataFrame([[user_id,30]], columns = ["user_id", "salary_day"])
    df = pd.merge(df, salary_date, on='user_id', how='inner')

    # Apply the function to create the salary date
    df['salary_date'] = df.apply(get_salary_date, axis = 1)
    df['salary_date'] = pd.to_datetime(df['salary_date']).dt.tz_localize('Asia/Kolkata')

    # Define the condition for "pay day" (transactions within 7 days after salary date)
    df['pay_day'] = df.apply(
        lambda row: 'pay day' if row['salary_date'] <= row['transactiondatetime'] <= row['salary_date'] + timedelta(days=7) else 'non pay day',
        axis=1
    )

    return df


# In[62]:


import pandas as pd

# Function to assign tags, specifically handling ranges crossing midnight
def assign_special_time_period(df, time_splits_df):
    # df['transactiondatetime'] = pd.to_datetime(df['transactiondatetime'])

    # Function to classify time, splitting the range when it crosses midnight
    def classify_time(transaction_time):
        time_only = transaction_time.replace(microsecond=0).time()
        for index, row in time_splits_df.iterrows():
            start_time = pd.to_datetime(row['start_time'], format='%H:%M:%S').time()
            end_time = pd.to_datetime(row['end_time'], format='%H:%M:%S').time()

            if start_time <= end_time:  # Same day range (e.g., 05:00:00 to 10:00:00)
                if start_time <= time_only <= end_time:
                    return row['name']
            else:  # Crossing midnight range (e.g., 23:00:00 to 02:00:00)
                # First part: from start_time to 23:59:59
                if start_time <= time_only <= pd.to_datetime("23:59:59", format='%H:%M:%S').time():
                    return row['name']
                # Second part: from 00:00:00 to end_time
                if pd.to_datetime("00:00:00", format='%H:%M:%S').time() <= time_only <= end_time:
                    return row['name']
        return None
        # return "unknown"

    # Apply classification based on the provided time splits
    df['special_time_period'] = df['transactiondatetime'].apply(classify_time).str.lower().str.replace("-"," ")
    return df



# In[63]:


def assign_tags(df,user_id):

    df['transactiondatetime'] = pd.to_datetime(df['transactiondatetime'])

    # Define a dictionary for time period classification
    time_periods = {
        "late night": (pd.to_datetime("00:00:00").time(), pd.to_datetime("04:00:00").time()),
        "early morning": (pd.to_datetime("04:00:01").time(), pd.to_datetime("07:00:00").time()),
        "morning": (pd.to_datetime("07:00:01").time(), pd.to_datetime("10:00:00").time()),
        "late morning": (pd.to_datetime("10:00:01").time(), pd.to_datetime("12:00:00").time()),
        "afternoon": (pd.to_datetime("12:00:01").time(), pd.to_datetime("15:00:00").time()),
        "late afternoon": (pd.to_datetime("15:00:01").time(), pd.to_datetime("18:00:00").time()),
        "evening": (pd.to_datetime("18:00:01").time(), pd.to_datetime("21:00:00").time()),
        "night": (pd.to_datetime("21:00:01").time(), pd.to_datetime("23:59:59").time())
    }

    # Function to classify time periods
    def classify_time_of_day(time):
        time_of_day = time.replace(microsecond=0).time()  # Extract time component
        for period, (start, end) in time_periods.items():
            if start <= time_of_day <= end:
                return period
        return "unknown"

    # Tag each transaction with day_of_week, weekend/working, and time period
    df['day_of_week'] = df['transactiondatetime'].dt.day_name()
    df['day_of_week'] = df['day_of_week'].str.lower()
    df['day_type'] = df['transactiondatetime'].dt.weekday.apply(lambda x: "weekend" if x >= 5 else "working day")
    #df['time_period'] = df['transactiondatetime'].apply(classify_time_of_day)
    print("df.sahpe",df.shape)
    print("df.markdown",df.head().to_markdown())
    df = tag_pay_day(df,user_id)
    df['is_festival_day'] = df['transactiondatetime'].apply(tag_festival_day)

    return df


# In[64]:


import pandas as pd

def generate_time_periods_from_csv(time_splits_df):
    # Load the CSV file
    # df = pd.read_csv(file_path)

    # Generate the time periods with descriptions
    time_periods = []
    for index, row in time_splits_df.iterrows():
        time_period = {
            "name": row['name'],
            "start_time": row['start_time'],
            "end_time": row['end_time'],
            "description": row['description']
        }
        time_periods.append(time_period)

    # Format the time period categories into the required format
    time_periods_prompt = "special_time_period: A categorical field representing the time period of the day when the transaction took place. Possible values are based on predefined time ranges:\n\n"

    for period in time_periods:
        time_periods_prompt += f'- {period["name"]} ({period["start_time"]} to {period["end_time"]}): {period["description"]}\n'

    return time_periods_prompt


# In[65]:


def filter_data(df, user_id, category):
    print("df.columns",df.columns)
    df['transactionDateTime'] = pd.to_datetime(df['transactionDateTime'], format='ISO8601', utc=True).dt.tz_convert(
        'Asia/Kolkata')
    mapping = {
        'customerID' : 'user_id',
        'transactionDateTime' : 'transactiondatetime',
        'transactionAmount' : 'transactionamount',
        'category': 'jupiter_coarsegrain_category'
    }
    df.rename(mapping, axis = 1, inplace = True)
    con1 = df['jupiter_coarsegrain_category'].str.lower() == category.lower()
    con2 = df['user_id'] == user_id
    col_intr = ['user_id', 'transactiondatetime', 'transactionamount',
       'jupiter_coarsegrain_category']
    df = df[col_intr]
    return df[con1&con2]

def get_data_cuts_for_category(category, current_month, user_id):
    # category = state['current_category']

    tags_groupings_sum = [
        ['special_time_period'],
        ['day_of_week'],
        ['day_of_week', 'special_time_period']
    ]


    tags_groupings_avg = [
        ['day_type'],
        ['pay_day'],
        ['is_festival_day'],
        ['day_type', 'special_time_period'],
        ['pay_day', 'special_time_period'],
        ['day_type', 'day_of_week']
    ]


    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    current_month_date = datetime.strptime(current_month, "%Y-%m")
    first_of_current_month = current_month_date.replace(day=1)
    last_of_current_month = (current_month_date + relativedelta(months=1, days=-1)).date()
    current_month_range = DateRange(start=first_of_current_month, end=last_of_current_month)

    start_month_date = current_month_date - relativedelta(months=7)
    start_of_start_month = start_month_date.replace(day=1)
    end_month_date = current_month_date - relativedelta(months=1)
    end_of_end_month = (end_month_date + relativedelta(months=1, days=-1)).replace(hour=23, minute=59, second=59)

    start_end_month_range = DateRange(start=start_of_start_month, end=end_of_end_month)
    print("----- data cuts months---->", current_month, start_month_date, end_month_date)

    prv_all_user_data = pd.DataFrame(get_transactions(user_id, {
        'transaction_date_range': current_month_range,
        'product': [ 'ALL' ]
    }))
    aug_all_user_data = pd.DataFrame(get_transactions(user_id, {
        'transaction_date_range': start_end_month_range,
        'product': [ 'ALL' ]
    }))
    print(prv_all_user_data.shape)
    print("user-id",user_id)
    aug_user_data = filter_data(aug_all_user_data, user_id, category)
    print(aug_user_data.shape)
    prv_user_data = filter_data(prv_all_user_data, user_id, category)
    aug_user_data = assign_tags(aug_user_data,user_id)
    prv_user_data = assign_tags(prv_user_data,user_id)

    time_splits_df = pd.read_csv(category.replace(" ","_")+"_time_splits_24hr.csv")
    prv_user_data = assign_special_time_period(prv_user_data, time_splits_df)
    aug_user_data = assign_special_time_period(aug_user_data, time_splits_df)
    special_time_period_description = generate_time_periods_from_csv(time_splits_df)

    # return prv_user_data

    if aug_user_data.shape[0] == 0:
        return ""

    highest_spends_data = []
    for tags_grouping in tags_groupings_sum:
        data_dict = {}
        # description = "below is the spends data for food & drink category in descending order of total spends of current month by grouping "+str(tags_grouping)
        description = f"Below is the spends data for {category} category in descending order of total spends of the current month by grouping {tags_grouping}"
        data_markdown = aug_user_data.groupby(tags_grouping)['transactionamount'].sum().sort_values(ascending = False).to_markdown()
        data_dict["description"] = description
        data_dict["data"] = data_markdown
        highest_spends_data.append(data_dict)


    tags_groupings = tags_groupings_sum + tags_groupings_avg
    for tags_grouping in tags_groupings:
        data_dict = {}
        description = f"Below is the spends data for {category} category in descending order of average spends of the current month by grouping {tags_grouping}"
        data_markdown = aug_user_data.groupby(tags_grouping)['transactionamount'].mean().sort_values(ascending = False).to_markdown()
        data_dict["description"] = description
        data_dict["data"] = data_markdown
        highest_spends_data.append(data_dict)

    average_spends_data = []
    for tags_grouping in tags_groupings:
        data_dict = {}
        prv_df = prv_user_data.groupby(tags_grouping)['transactionamount'].mean().to_frame()
        prv_df.columns = ['previous_avg_transaction_amount']

        aug_df = aug_user_data.groupby(tags_grouping)['transactionamount'].mean().to_frame()
        aug_df.columns = ['current_avg_transaction_amount']
        final_df = pd.merge(aug_df, prv_df, left_index = True, right_index = True)
        final_df['percentage_change'] = (final_df.pct_change(axis = 1)['current_avg_transaction_amount'].values)*100
        final_df['absolute_difference'] = final_df['current_avg_transaction_amount'] - final_df['previous_avg_transaction_amount']
        data_markdown = final_df.to_markdown()

        description = f"Below is the data on previous month's average and current month's average for {category} category. The data also shows percentage change and absolute difference. Data is grouped by columns {tags_grouping}"
        data_dict["description"] = description
        data_dict["data"] = data_markdown
        average_spends_data.append(data_dict)

    markdown = ""
    for ele in highest_spends_data:
        markdown = markdown+ "\n\n\n" + ele['description'] + "\n" + ele['data']
    for ele in average_spends_data:
        markdown = markdown+ "\n\n\n" + ele['description'] + "\n" + ele['data']

    return markdown, special_time_period_description


# In[ ]:





# In[66]:


from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field

class Strategy(BaseModel):
    observations: List[str] = Field(description='Critical observation in provided data cuts')
    assumptions: List[str] = Field(description='Assumptions taken while making strategy for given category')
    strategy: List[str] = Field(description='Detailed step by step strategy personalized to user to maximize saving')
    info_needed: List[str] = Field(description='Extra information needed to recommand more personalized strategy')

strategy_parser = PydanticOutputParser(pydantic_object=Strategy)


# In[67]:


field_description = """

day_of_week: A categorical field containing the names of the days of the week. Possible values are: "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", and "sunday". It is derived from the 'transactiondatetime' field, representing the day of the transaction.


day_type: A categorical field indicating whether the transaction occurred on a "working day" (Monday to Friday) or a "weekend" (Saturday and Sunday). It is determined based on the day of the week from the 'transactiondatetime' field.



pay_day: A categorical field that indicates whether a transaction occurred within 7 days after the user's salary date. Possible values are:
"pay day" for transactions that occurred within the 7-day window after the salary date.
"non pay day" for transactions that occurred outside this window.
The salary date is based on the user’s salary day (either provided or computed based on the last day of the month if needed).


is_festival_day: A categorical field indicating whether the transaction occurred around a festival day. It has three possible values:
"pre festival day" for transactions within 5 days before the festival.
"festival day" for transactions on the day of the festival.
"post festival day" for transactions within 2 days after the festival.
If the transaction date does not match any of these conditions, the value will be "normal Day". It is determined by checking the transaction date against a predefined list of festival days.


salary_date: A datetime field representing the calculated salary payment date for the transaction month. It is derived from the 'salary_day' field and the transaction's year and month. If the 'salary_day' exceeds the number of days in the transaction month, the last day of the month is used.

{special_time_period_description}
"""


# In[68]:


category_descriptions = {
    "groceries": "purchases made at grocery stores, supermarkets, or through online platforms like Blinkit and Zepto, typically involving food items and household supplies needed to prepare meals at home.",
    "shopping": "general purchases of goods or services, including clothing, electronics, home goods, and other retail products.",
    "commute": "expenses related to transportation, such as public transit fares, ridesharing services, or other travel costs related to daily commuting through cab, auto, bike taxi etc.",
    "entertainment": "expenses related to leisure activities, including movies, concerts, sports events, gaming, streaming services, and other forms of recreational enjoyment.",
    "food & drinks": "spending on meals and beverages, including dining at restaurants, cafes, takeout, bars, and food orders from online food delivery apps."
}


# In[69]:


from langchain_core.prompts import PromptTemplate
import datetime

prompt_template = """
You are expert finncial advisor working for jupiter (1-app for everything money) who has a experties in how to maximize savings by 
reducing unneccessary spends and by finding cheaper alternative options.


Below you have been provided with the data of {category} category in markdown format. {category} category contains {category_description}. Each data is 
provided with description about how data  was prepared and what it contains.
all the mentioned amounts are in Indian National Rupee (INR) and times are in Indian Standard Time (IST).
data has different cuts based on different combinations of below fields,
{field_description}


look at the data provided below carefully:
{markdown}


Use this information if required, current time: {time}

Question: Recommand savings strategies that is personalised to the user to maximize savings on {category} category as best you can.
Think step by step on provided information and Write your answer in given format: \n {format_instructions}
"""

strategy_prompt = PromptTemplate(template = prompt_template,input_variables = ["markdown","category","special_time_period_description"]).partial(
    time=lambda: datetime.datetime.now().isoformat(),field_description = field_description,
    format_instructions = strategy_parser.get_format_instructions()
)


# In[70]:


strategy_chain = strategy_prompt | chat_model | strategy_parser


# In[ ]:





# In[ ]:





# In[ ]:





# In[71]:


def category_selector(state):
    # state["current_index"] += 1
    current_index = state["current_index"]
    current_index += 1
    return {"current_index":current_index}


# In[72]:


def category_router(state):
    current_index = state["current_index"]
    generated_category_list = state["reasoning"]

    # Continue processing if there are values left in the list
    if current_index < len(generated_category_list):
        return "CategoryProcessor"
    else:
        return "RefineAnswer"


# In[73]:


def process_category(state):
    category_reasoning = state['reasoning'][state['current_index']]
    current_category = category_reasoning['Category'].lower()
    category_of_interest = ["food & drinks","shopping","commute","groceries","entertainment"]
    if current_category not in category_of_interest:
        return {"strategy":[]}
    # state["current_category"] = current_category
    current_month = state["current_month"]
    markdown, special_time_period_description = get_data_cuts_for_category(current_category, current_month, state["user_id"])
    if markdown == "":
        return {"strategy":[]}
    #print(markdown)
    print(current_category)
    response = strategy_chain.invoke({"markdown":markdown, "category":current_category,
                                      "category_description" : category_descriptions[current_category],
                                      "special_time_period_description":special_time_period_description })
    response_dict = response.dict()
    print("out")
    response_dict['category'] = current_category
    return {"strategy":[response_dict]}


# In[ ]:





# In[ ]:





# In[74]:


from langchain_core.prompts import PromptTemplate
import datetime

refine_prompt_template = """
You are expert finncial advisor working for jupiter (1-app for everything money) who has a experties in how to maximize savings by 
reducing unneccessary spends and by finding cheaper alternative options.

{content}

Your task is to list down all the observations in user-friendly manner.

While mentioning observations, specifically mentioned about category that you are talking about.
"""

refine_prompt = PromptTemplate(template = refine_prompt_template,
                               input_variables = ["category","category_description","observation1","observation2","strategy1","strategy2"])


# In[75]:


refine_chain = refine_prompt | chat_model


# In[76]:


def refine_answer(state):
    strategies = state["strategy"]
    final_content = ""
    for strategy in strategies:
        content = f"""
        # Review the following observations and their associated strategies for maximizing savings in the {strategy['category']} category.
        # {strategy['category']} category contains {category_descriptions[strategy['category']]}:
        # Observations: {strategy['observations']}
        # Strategy: {strategy['strategy']} \n\n"""
        content = f"""
        Review the following observations for maximizing savings in the {strategy['category']} category.
        {strategy['category']} category contains {category_descriptions[strategy['category']]}:
        Observations: {strategy['observations']}"""
        final_content = final_content + content
    #print(final_content)
    response = refine_chain.invoke({"content":final_content})
    # response_dict = {}
    # response_dict['category'] = current_category
    # response_dict['response'] = response.content
    # print("=======",response.content)
    # return {"final_answer":response.content}
    return {"messages":[AIMessage(content = response.content)]}



# ### Supervior

# In[43]:
def get_monthly_recap(state):
    return {"messages": [AIMessage(content="You can access your Money Recap using the following link:\n\n- [Money Recap Dashboard for September](https://app.jupiter.money/money-recap)")]}

def not_supported_question(state):
    return {"messages": [AIMessage(content="This is a financial bot. This query doesn't seems to be related to financial information hence can't help with this.")]}

# members = ["transaction_data_analyst" , "Researcher", "Coder"]
members = ["insights_generator", "analyse_upcoming_transaction_data", "not_supported", "monthly_recap", "transaction_data_analyst", "Researcher", "jupiter_info"]
options = ["FINISH"] + members

# In[44]:


from pydantic import BaseModel, Field
from typing import Literal
from langchain.tools import tool
import json


class RoleSchema(BaseModel):
    next: Literal[*options] = Field(description="The next role to select. Possible values: " + (' ,'.join(options)))


@tool("route", args_schema=RoleSchema, return_direct=True)
def route(next: str):
    """
    Select the next role based on the provided input.
    """

    return json.dumps({
        "next": next
    })


# In[45]:


from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.tools import tool

system_prompt = """
You are a supervisor tasked with managing a conversation between the
following workers:  {members}. Given the following user request,
respond with the worker to act next.

Use insights_generator agent when user ask for monthly spends insights. for example, give me spend insights for August, 2024.
Use analyse_upcoming_transaction_data when the question asks for upcoming transactions, future transactions, or transactions that are expected to occur in the future 
Use monthly_recap agent when question asks for Oct month summary/overview etc. If it's any other month then Current month/Oct then don't return this flow & move on to next
Use transaction_data_analyst agent when question needs an analysis on user's past transaction data
Use Research agent when questions require seearching a web to get generic financial information 
Use jupiter_info when question require access to user's personal and account details or need an information about 
jupiter's (1-app for everythinh money) product.
Use not_supported agent when question is not related to financial data/query be it generic information or user specific information


Each worker will perform a task and respond with their results and status.
When finished, respond with FINISH.
"""
# Use Code Agent when questions require code generation and code execution
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

route_openai_function_schema = convert_to_openai_function(route)


# here we need to use bind_functions because JsonOutputFunctionsParser will fetch 'function_call' from additional_kwargs.
# if we use bind_tools then we will get 'tool_calls' in additional_kwargs.
supervisor_chain = (
        prompt
        | chat_model.bind_functions([route_openai_function_schema])
        | JsonOutputFunctionsParser()
)


# In[ ]:


# ### Human in loop

# In[ ]:


# In[46]:


def human_in_loop(state):
    ai_message = state["intermediate_steps"][-1]
    inp = input(ai_message.content)
    state["messages"] = state["messages"] + [ai_message]
    return {"messages": [HumanMessage(content=inp)]}


# In[ ]:


# In[ ]:


# ### Router

# In[ ]:


# In[47]:


def data_fetch_router(state):
    ai_message = state["intermediate_steps"][-1]
    if ai_message.content == '':
        print("-----> Fetching data")
        return "fetch_data"
    else:
        state["messages"] = state["messages"] + [ai_message]
        # print(state)
        print("-----> ask_human")
        return "ask_human"


# In[48]:


def router(state):
    # print("----------->",state["next"])
    if state["next"] == "FINISH":
        last_message = state["messages"][-1]
        # print("----------->",last_message.type)
        if last_message.type != "ai":
            return "Researcher"
    return state["next"]


# In[ ]:


# In[ ]:


# ### Build Graph

# In[ ]:


# In[49]:


from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage
import operator
from typing import Annotated
from typing_extensions import TypedDict
import pandas as pd


class DataSchema(TypedDict):
    dataframe: pd.DataFrame
    description: str


class AgentState(TypedDict):
    user_id: str
    current_month: str
    messages: Annotated[list, add_messages]
    intermediate_steps: Annotated[list, operator.add]
    dataframe_store: List[DataSchema]
    reasoning: List[CategoryReasoning]
    current_index: int
    strategy : Annotated[list, operator.add]
    next: str = Field(description="next role to perform")


# In[50]:


workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_chain)

workflow.add_node("filter_generator", txn_data_fetcher.respond)
workflow.add_node("fetch_data", invoke_get_txn_data_tool)
workflow.add_node("analyse_upcoming_transaction_data", get_upcoming_txn_data_tool)
workflow.add_node("analyse_transaction_data", analyse_transaction_data)
workflow.add_node("ask_human", human_in_loop)
workflow.add_node("monthly_recap", get_monthly_recap)
workflow.add_node("not_supported", not_supported_question)

workflow.add_node("Researcher", research_node)
workflow.add_node("jupiter_info", jupiter_info_node)

workflow.add_node("CurrentMonth", get_current_month)
workflow.add_node("Reasoning", get_savings_strategies)
workflow.add_node("CategorySelector", category_selector)
workflow.add_node("CategoryProcessor", process_category)
workflow.add_node("RefineAnswer", refine_answer)



workflow.set_entry_point("supervisor")

path_mapping = {k:k for k in members}
path_mapping["transaction_data_analyst"] = "filter_generator"
path_mapping["insights_generator"] = "CurrentMonth"
path_mapping["FINISH"] = END


workflow.add_conditional_edges(source="supervisor", path=router, path_map=path_mapping)

path_mapping = {"ask_human": "ask_human", "fetch_data": "fetch_data"}
workflow.add_conditional_edges(source="filter_generator", path=data_fetch_router, path_map=path_mapping)
workflow.add_edge("ask_human", "supervisor")
# workflow.add_edge("ask_human", "filter_generator")
# workflow.add_edge("filter_generator", "fetch_data")
workflow.add_edge("fetch_data", "analyse_transaction_data")
# workflow.add_edge("analyse_transaction_data", END)

workflow.add_edge("Researcher", "supervisor")
workflow.add_edge("jupiter_info", "supervisor")
workflow.add_edge("analyse_transaction_data", "supervisor")
workflow.add_edge("monthly_recap", END)
workflow.add_edge("not_supported", END)
workflow.add_edge("analyse_upcoming_transaction_data", END)

workflow.add_edge("CurrentMonth", "Reasoning")
workflow.add_edge("Reasoning", "CategorySelector")
insights_graph_mapper = {
    "CategoryProcessor":"CategoryProcessor",
    "RefineAnswer":"RefineAnswer"
}
workflow.add_conditional_edges("CategorySelector", path = category_router, path_map = insights_graph_mapper)
workflow.add_edge("CategoryProcessor", "CategorySelector")
workflow.add_edge("RefineAnswer", END)

# import sqlite3
# from langgraph.checkpoint.sqlite import SqliteSaver
# # memory = MemorySaver()
# memory = SqliteSaver.from_conn_string(":memory:")
# conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
# memory = SqliteSaver(conn)
# graph = workflow.compile(checkpointer=memory)

graph = workflow.compile()

# In[51]:


from IPython.display import Image, display

display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

# In[52]:


from langchain_core.messages import HumanMessage
# #
# # events = graph.stream(
# #     {"messages": HumanMessage(content="What is my UPI spend on march 2024 ?")},
# #     stream_mode="values"
# # )
#
# for i, step in enumerate(events):
#     print(f"Step {i}")
#     mod_step = step['messages']
#     mod_step[-1].pretty_print()

# In[50]:


# from langchain_core.messages import HumanMessage

# events = graph.stream(
#     {"messages": HumanMessage(content="What specific categories in your transactions are you most concerned about?")},
#     stream_mode="values"
# )


# for i, step in enumerate(events):
#     print(f"Step {i}")
#     mod_step = step['messages']
#     mod_step[-1].pretty_print()


# In[51]:


# response = graph.invoke({"messages": HumanMessage(content="how much did I spent on swiggy starting from march 2024 to today ?")})
# response["messages"][-1].content


# In[52]:


# mod_step


# In[53]:


# step


# In[54]:


# from langchain_core.messages import HumanMessage
# config = {"configurable": {"thread_id": "chat_1"}}
# events = graph.stream(
#     {"messages": [HumanMessage(content="How can I reduce my monthly expenses?")]},
#     stream_mode="values"
# )


# for i, step in enumerate(events):
#     print(f"Step {i}")
#     mod_step = step['messages']
#     mod_step[-1].pretty_print()


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[55]:


# from langchain_core.messages import HumanMessage
# config = {"configurable": {"thread_id": "chat_1"}}
# events = graph.stream(
#     {"messages": [HumanMessage(content="can you look at my transaction and tell me how can I reduce my monthly expenses?")]},
#     stream_mode="values"
# )


# for i, step in enumerate(events):
#     print(f"Step {i}")
#     mod_step = step['messages']
#     mod_step[-1].pretty_print()


# In[98]:


from langchain_core.messages import HumanMessage

# config = {"configurable": {"thread_id": "chat_1"}}
# events = graph.stream(
#     {"messages": [HumanMessage(content="what is my current balance ?")]},
#     stream_mode="values"
# )
#
# for i, step in enumerate(events):
#     print(f"Step {i}")
#     mod_step = step['messages']
#     mod_step[-1].pretty_print()

# In[109]:


from langchain_core.messages import HumanMessage

# config = {"configurable": {"thread_id": "chat_1"}}
# events = graph.stream(
#     {"messages": [HumanMessage(content="what is my total balance in saving accounts?")]},
#     stream_mode="values"
# )
#
# for i, step in enumerate(events):
#     print(f"Step {i}")
#     mod_step = step['messages']
#     mod_step[-1].pretty_print()

# In[110]:


# 35435345.64 + 354668657.75

# In[88]:


from langchain_core.messages import HumanMessage

# config = {"configurable": {"thread_id": "chat_1"}}
# events = graph.stream(
#     {"messages": [HumanMessage(content="what is my IFSC Code ?")]},
#     stream_mode="values"
# )
#
# for i, step in enumerate(events):
#     print(f"Step {i}")
#     mod_step = step['messages']
#     mod_step[-1].pretty_print()

# In[52]:


from langchain_core.messages import HumanMessage

# config = {"configurable": {"thread_id": "chat_1"}}
# events = graph.stream(
#     {"messages": [HumanMessage(content="can you tell me my account details ?")]},
#     stream_mode="values"
# )
#
# for i, step in enumerate(events):
#     print(f"Step {i}")
#     mod_step = step['messages']
#     mod_step[-1].pretty_print()

# In[53]:


from langchain_core.messages import HumanMessage

# events = graph.stream(
#     {"messages": HumanMessage(content="what is india's last 5 years GDP ?")},
#     stream_mode="values"
# )
#
# for i, step in enumerate(events):
#     print(f"Step {i}")
#     mod_step = step['messages']
#     mod_step[-1].pretty_print()

# In[54]:


from langchain_core.messages import HumanMessage

# events = graph.stream(
#     {"messages": HumanMessage(content="plot india's last 5 years GDP ?")},
#     stream_mode="values"
# )
#
# for i, step in enumerate(events):
#     print(f"Step {i}")
#     mod_step = step['messages']
#     mod_step[-1].pretty_print()

# In[55]:

#
# events = graph.stream(
#     {"messages": HumanMessage(
#         content="Can I get a detailed report of my spending habits? consider last 6 months transactions")},
#     stream_mode="values"
# )
#
# for i, step in enumerate(events):
#     print(f"Step {i}")
#     mod_step = step['messages']
#     mod_step[-1].pretty_print()
#
# # In[53]:
#
#
# from langchain_core.messages import HumanMessage
#
# events = graph.stream(
#     {"messages": HumanMessage(content="how much did I spent on swiggy starting from march 2024 to today ?")},
#     stream_mode="values"
# )
#
# for i, step in enumerate(events):
#     print(f"Step {i}")
#     mod_step = step['messages']
#     print(mod_step)
#     mod_step[-1].pretty_print()
#
# # In[54]:
#
#
# from langchain_core.messages import HumanMessage
#
# events = graph.stream(
#     {"messages": HumanMessage(content="how much did I spent on swiggy on month of May and July ?")},
#     stream_mode="values"
# )
#
# for i, step in enumerate(events):
#     print(f"Step {i}")
#     mod_step = step['messages']
#     print(mod_step)
#     mod_step[-1].pretty_print()
#
# # In[58]:
#
#
# events = graph.stream(
#     {"messages": HumanMessage(content="What is the amount debited via UPI in last two months?")},
#     stream_mode="values"
# )
#
# for i, step in enumerate(events):
#     print(f"Step {i}")
#     mod_step = step['messages']
#     mod_step[-1].pretty_print()
#
# # In[73]:
#
#
# events = graph.stream(
#     {"messages": HumanMessage(content="how much did I spent of food on last month on jupiter ?")},
#     stream_mode="values"
# )
#
# for i, step in enumerate(events):
#     print(f"Step {i}")
#     mod_step = step['messages']
#     mod_step[-1].pretty_print()
#
# # In[59]:
#
#
# events = graph.stream(
#     {"messages": HumanMessage(content="how much did I spent of swiggy on jupiter ?")},
#     stream_mode="values"
# )
#
# for i, step in enumerate(events):
#     print(f"Step {i}")
#     mod_step = step['messages']
#     mod_step[-1].pretty_print()
#
# # In[64]:
#
#
# from langchain_core.messages import HumanMessage
#
# events = graph.stream(
#     {"messages": HumanMessage(content="What are my most frequent expenses?")},
#     stream_mode="values"
# )
#
# for i, step in enumerate(events):
#     print(f"Step {i}")
#     mod_step = step['messages']
#     mod_step[-1].pretty_print()
#
# # In[ ]:
#
#
# # In[49]:
#
#
# examples = [
#     {
#         "user question": "Can I get a detailed report of my spending habits? consider last 6 months data",
#         "data query": "fetch last 6 months transaction data",
#     },
#     {
#         "question": "Which actors played in the movie Casino?",
#         "query": "MATCH (m:Movie{title:'Casino'})<-[:ACTED_IN]-(p:Person) RETURN p.name",
#     },
#     {
#         "question": "How many movies has Tom Hanks acted in?",
#         "query": "MATCH (a:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie) RETURN count(m)",
#     },
#     {
#         "question": "List all the genres of the movie Four Rooms",
#         "query": "MATCH (m:Movie{title:'Four Rooms'})-[:IN_GENRE]->(g:Genres) RETURN g.type",
#     },
#     {
#         "question": "Which actors have worked in movies from both the comedy and action genres?",
#         "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g1:Genres), (a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g2:Genres) WHERE g1.type = 'Comedy' AND g2.type = 'Action' RETURN DISTINCT a.name",
#     },
#     {
#         "question": "Identify movies where directors also played a role in the film.",
#         "query": "MATCH (p:Person)-[:DIRECTED]->(m:Movie), (p)-[:ACTED_IN]->(m) RETURN m.title, p.name",
#     },
#     {
#         "question": "Find the actor with the highest number of movies in the database.",
#         "query": "MATCH (a:Person)-[:ACTED_IN]->(m:Movie) RETURN a.name, COUNT(m) AS movieCount ORDER BY movieCount DESC LIMIT 1",
#     },
# ]

# In[ ]:


# In[ ]:



