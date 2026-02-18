# -*- coding: utf-8 -*-
import os
from constant import ROOT
from src.app.model.request_seq import RequestSeq
from src.app.model.request_token import RequestToken
from src.app.lightRAG.lightrag import LightRAG, QueryParam
from src.app.lightRAG.lightrag.utils import EmbeddingFunc
from src.app.util import db_utils
import numpy as np
from dotenv import load_dotenv
import aiohttp
import logging
import shutil
import uuid
from datetime import datetime

logging.basicConfig(level=logging.INFO)

load_dotenv()

AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

BASE_DIR = "./KG/"

embedding_dimension = 3072

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    endpoint = f"YOUR_ENDPOINT" # different model instance has different endpoint url and param list, fill with the params you configured in .env

    messages = []
    
    if not system_prompt and not kwargs.get("system_prompt"):
        messages.append({"role": "system", "content": prompt})
        if history_messages:
            messages.extend(history_messages)
    else:
        messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
    
    payload = {
        "messages": messages,
        "temperature": kwargs.get("temperature", 0),    # Temperature setting: 0
        "top_p": kwargs.get("top_p", 1),
        "n": kwargs.get("n", 1),
    }
    db = next(db_utils.get_db())
    inst = RequestToken()
    inst.req_id = kwargs.get("req_id", "")
    inst.req_type = kwargs.get("req_type", "")
    inst.create_at = datetime.now()
    inst.scenario = os.getenv("scenario")

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, headers=headers, json=payload) as response:
            if response.status != 200:
                raise ValueError(
                    f"Request failed with status {response.status}: {await response.text()}"
                )
            result = await response.json()
            # Recording token consumption data.
            inst.completion_tokens = result.get("usage").get("completion_tokens")
            inst.prompt_tokens = result.get("usage").get("prompt_tokens")
            db.add(inst)
            db.commit()
            return result["choices"][0]["message"]["content"]


async def embedding_func(texts: list[str]) -> np.ndarray:
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    endpoint = f"YOUR_ENDPOINT" # different model instance has different endpoint url and param list, fill with the params you configured in .env

    payload = {"input": texts}

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, headers=headers, json=payload) as response:
            if response.status != 200:
                raise ValueError(
                    f"Request failed with status {response.status}: {await response.text()}"
                )
            result = await response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            return np.array(embeddings)

rag = LightRAG(
        working_dir=BASE_DIR + "YOUR_BASE_ENTRY", # change to your base_entry which stored in DB table: subgraph_pool_mapping
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,    # max_token_size setting
            func=embedding_func,
        ),
)

# Track and record every request, token consumption calculation is depending on this log data.      
def record_query(content: str, req_type: str = "SEARCH") -> RequestSeq:
    """
    Params:
        req_type: SEARCH | GENERATE
    """
    req_id = str(uuid.uuid4())
    db = next(db_utils.get_db())
    inst = RequestSeq()
    inst.content = content
    inst.req_id = req_id
    inst.req_type = req_type
    inst.scenario = os.getenv("scenario")
    db.add(inst)
    db.commit()
    db.refresh(inst)
    return inst

def search_public(query:str):
    record_inst = record_query(query, req_type="SEARCH")
    # Use LightRAG's "local query" as default retrieval workflow
    response_str = rag.query(query, param=QueryParam(mode="local", req_id=record_inst.req_id))
    return """
    {response_str}
""".format(response_str=response_str)
    
def del_KG_data(target_dir:str):
    if not os.path.exists(target_dir):
        return
    shutil.rmtree(target_dir, ignore_errors=True)
    
