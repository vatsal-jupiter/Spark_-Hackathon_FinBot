from fastapi import FastAPI, Header, HTTPException, Request, Query
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from typing import Annotated
from db.user_dao import get_user_history, get_user_sessions_with_limit, register_message, customer_tier_info
from model import graph

app = FastAPI()

class QueryModel(BaseModel):
    query: str


def get_app_description():
    return (
        """
        A chatbot API for financial queries.\n
        Different Intent/Use cases.\n

        Analytical layer/Let’s talk money -> Filter generation is in place, these need to be integrated with mm/wealth tab APIs.\n
        User-specific information like IFSC code etc. - Banking/CC API integration\n
        Generic Financial information/ Question/Answers -> Open-ended web calls/RAG ??\n
        Category-level spends\n
        Category-level budgets/scope for spend in the current month\n
        Display upcoming transactions\n
        Investment-related basic analytics\n

        Investment-related basic analytics\n
        What are my investments in July?\n
        How much of my investments correspond to Stocks/Mutual Funds/FDs/Gold, etc.?\n
        What are the different types of mutual funds that I invested in July?\n
        What are the trends with respect to my investments across different instruments month by month since July?\n
        """
    )


# Define the root endpoint to return the app description
@app.get("/")
async def root():
    return {"message": get_app_description()}


# Define an endpoint to take customerId in header and query in JSON body
@app.post("/query/{session_id}")
async def query(request: Request, session_id: str, customer_id: str = Header(None)):
    if not customer_id:
        raise HTTPException(status_code=400, detail="customerId header missing")

    data = await request.json()
    query_text = data.get("query")

    if not query_text:
        raise HTTPException(status_code=400, detail="Query field missing in body")


    register_message(customer_id, session_id, query_text, type='query') # saved in db

    # get response

    response = graph.invoke(
        {"messages": HumanMessage(content=query_text), 'user_id': customer_id},
        config={"recursion_limit":50}
    )

    register_message(customer_id, session_id, response["messages"][-1].content, type='response') # saved response in db

    # Simulated response, replace with actual query processing logic
    return {"customer_id": customer_id, "response": response["messages"][-1].content}


class ChatHistoryModel(BaseModel):
    session_id: str
    user_id: str
    created_at: str
    title: str
    last_message: str  # Assuming you might have this or similar field for preview purposes

# Define a model for the chat preview response
class ChatPreviewResponse(BaseModel):
    customer_id: str
    chat_history: List[ChatHistoryModel]


@app.get("/chats", response_model=ChatPreviewResponse)
async def get_chat_preview(preview_size: int = 2, customer_id: Annotated[str | None, Header()] = None):
    try:
        if not customer_id or customer_id == '':
            raise HTTPException(status_code=400, detail="Invalid customer ID")

        chat_sessions = get_user_sessions_with_limit(customer_id, preview_size)

        if not chat_sessions:
            return {"customer_id": customer_id, "chat_history": []}


        chat_history = [
            ChatHistoryModel(
                session_id=session[0],
                user_id=session[1],
                created_at=session[2].strftime("%Y-%m-%d %H:%M:%S") if session[2] else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                last_message=session[4] if session[4] else '',
                title=session[5],
            )
            for session in chat_sessions
        ]

        return {"customer_id": customer_id, "chat_history": chat_history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define a model for the individual chat message
class ChatMessageModel(BaseModel):
    message_id: str
    session_id: str
    user_id: str
    message: str
    message_type: str
    generated_at: str
    images: Optional[str] = None

# Define a model for the chat data response
class ChatDataResponse(BaseModel):
    customer_id: str
    session_id: str
    messages: List[ChatMessageModel]

# Endpoint to get chat data by session_id
@app.get("/chat/{session_id}", response_model=ChatDataResponse)
async def get_chat_data(session_id: str, timestamp_start: Optional[str] = None, timestamp_end: Optional[str] = None, customer_id: str = Header(None)):
    try:
        # Call the DAO to get the chat history for the given customer_id and session_id
        chat_messages = get_user_history(customer_id, session_id, timestamp_start, timestamp_end)

        if not chat_messages:
            raise HTTPException(status_code=404, detail="No chat messages found for this session")

        print(chat_messages)
        # Transform the result from DAO to response model

        messages = [
            ChatMessageModel(
                message_id=message[0],
                session_id=message[1],
                user_id=message[2],
                message=message[3],
                message_type=message[4],
                images=message[5],
                generated_at=message[6].strftime("%Y-%m-%d %H:%M:%S") if message[6] else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            for message in chat_messages
        ]
        return {"customer_id": customer_id, "session_id": session_id, "messages": messages}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CustomerTierModel(BaseModel):
    customer_id: str
    tier: str
    coarse_grain_tier: str
    valid_till: str



# define an api /get to get the customer tier details
@app.get("/customer_tier")
async def get_customer_tier(customer_id: str = Header(None)):
    try:
        customer_tiers = customer_tier_info(customer_id)

        if not customer_tiers:
            raise HTTPException(status_code=404, detail="No tier found for this customer")

        customer_tier = [
                CustomerTierModel(
                    customer_id=customer_tier[0],
                    tier=customer_tier[1],
                    coarse_grain_tier=customer_tier[2],
                    valid_till=customer_tier[3].strftime("%Y-%m-%d %H:%M:%S") if customer_tier[3] else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                for customer_tier in customer_tiers
        ]

        return {"customer_tier": customer_tier}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))