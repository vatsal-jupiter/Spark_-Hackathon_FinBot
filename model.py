#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from dotenv import load_dotenv
import os

from services.wt import get_transactions

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

from langchain_openai import AzureChatOpenAI

chat_model = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",  # or your deployment
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


def create_agent(chat_model: ChatOpenAI, tools: list, system_prompt: str):
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
    agent = create_openai_tools_agent(chat_model, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor


# In[11]:


# Converting to executable node
def agent_node(state, agent, name):
    # result = agent.invoke(state)
    result = agent.invoke({"messages": state["messages"]})
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
spend : Credit/Debit Indicator -> DEBIT
external banks: exclude JUPITER, Edge Rupay card, Edge VISA card from Product
non jupiter: exclude JUPITER, Edge Rupay card, Edge VISA card from Product

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


# In[23]:


from langgraph.prebuilt.tool_executor import ToolExecutor

tools = [get_txn_data_tool]
tool_executor = ToolExecutor(tools)

# In[24]:


from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

description_generation_template = """
I am uisng one tool with my AI agent. Below is the argument schema of the function:
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
def get_account_details():
    """
    Get the personal information and account information of the user.
    """
    account_details = [{
        "Name": "Vatsal Patel",
        "Gender": "Male",
        "IFSC Code": "FDRL0007777",
        "Account Number": "77770120692039",
        "Account Type": "Salary",
        "Current Balance": "2342343.455"
    },
        {
            "Name": "Vatsal Patel",
            "Gender": "Male",
            "IFSC Code": "ICICI0007777",
            "Account Number": "01470120692039",
            "Account Type": "Saving",
            "Current Balance": "35435345.64"
        },
        {
            "Name": "Vatsal Patel",
            "Gender": "Male",
            "IFSC Code": "HDFC0007777",
            "Account Number": "02470120692039",
            "Account Type": "Saving",
            "Current Balance": "354668657.7543"
        }]

    # Return the result as a JSON object
    return json.dumps(account_details)


# In[40]:


system_prompt = """
You have an access if user's personal and account information. also, you have an information about 
products of juiter (1-app for everything money)
You will get an access to user's personal and account information through account-deatils-tool
"""

# In[41]:


tools = [get_account_details]
jupiter_info_agent = create_agent(chat_model, tools, system_prompt)

# In[42]:


import functools

jupiter_info_node = functools.partial(agent_node, agent=jupiter_info_agent, name="jupiter_info")

# In[ ]:


# In[ ]:


# ### Supervior

# In[43]:
def get_monthly_recap(state):
    return {"messages": [AIMessage(content="You can access your Money Recap using the following link:\n\n- [Money Recap Dashboard for September](https://app.jupiter.money/money-recap)")]}

# members = ["transaction_data_analyst" , "Researcher", "Coder"]
members = ["monthly_recap", "transaction_data_analyst", "Researcher", "jupiter_info"]
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

Use monthly_recap agent when question asks for Oct month summary/overview etc. If it's any other month then Current month/Oct then don't return this flow & move on to next
Use transaction_data_analyst agent when question needs an analysis on user's past transaction data
Use Research agent when questions require seearching a web to get finacial information and any general purpose query
Use jupiter_info when question require access to user's personal and account details or need an information about 
jupiter's (1-app for everythinh money) product.


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
    messages: Annotated[list, add_messages]
    intermediate_steps: Annotated[list, operator.add]
    dataframe_store: List[DataSchema]
    next: str = Field(description="next role to perform")


# In[50]:


workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_chain)

workflow.add_node("filter_generator", txn_data_fetcher.respond)
workflow.add_node("fetch_data", invoke_get_txn_data_tool)
workflow.add_node("analyse_transaction_data", analyse_transaction_data)
workflow.add_node("ask_human", human_in_loop)
workflow.add_node("monthly_recap", get_monthly_recap)

workflow.add_node("Researcher", research_node)
workflow.add_node("jupiter_info", jupiter_info_node)

workflow.set_entry_point("supervisor")

# workflow.set_entry_point("filter_generator")
path_mapping = {k: k for k in members}
path_mapping["transaction_data_analyst"] = "filter_generator"
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



