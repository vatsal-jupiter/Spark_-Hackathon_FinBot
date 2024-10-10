from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

app = FastAPI()


# Define a model for the body of the request
class QueryModel(BaseModel):
    query: str


# Define a function to return a description of the app
def get_app_description():
    return (
        """
        Different Intent/Use cases.

        Analytical layer/Let’s talk money -> Filter generation is in place, these need to be integrated with mm/wealth tab APIs.
        User-specific information like IFSC code etc. - Banking/CC API integration
        Generic Financial information/ Question/Answers -> Open-ended web calls/RAG ??
        Category-level spends
        Category-level budgets/scope for spend in the current month
        Display upcoming transactions
        Investment-related basic analytics

        Investment-related basic analytics
        What are my investments in July?
        How much of my investments correspond to Stocks/Mutual Funds/FDs/Gold, etc.?
        What are the different types of mutual funds that I invested in July?
        What are the trends with respect to my investments across different instruments month by month since July?
        """
    )


# Define the root endpoint to return the app description
@app.get("/")
async def root():
    return {"message": get_app_description()}


# Define an endpoint to take customerId in header and query in JSON body
@app.post("/query")
async def query(request: Request, customerId: str = Header(None)):
    if not customerId:
        raise HTTPException(status_code=400, detail="customerId header missing")

    data = await request.json()
    query = data.get("query")

    if not query:
        raise HTTPException(status_code=400, detail="Query field missing in body")

    # Simulating a response based on the query, for now returning the same query.
    # You can add actual logic here to process different queries.
    return {"customerId": customerId, "response": f"Response for query: {query}"}
