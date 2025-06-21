# from typing import Union

# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

from fastapi import FastAPI
from pydantic import BaseModel
from app.api.query import query_engine_response

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_llm(data: QueryRequest):
    result = query_engine_response(data.question)
    return {"response": result}

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))