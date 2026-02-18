from fastapi import APIRouter
from src.app.service import lightRAG_service, jigsaw_service
from pydantic import BaseModel
import os

from src.app.benchmark.dataset_exp import dataset_exp as batch_qa_exp
from src.app.benchmark.jaccard_eval import calculate_all_jaccard_scores as jaccard_exp
from src.app.benchmark.dataset_prf_evaluation import prf_eval as precision_recall_f1_eval
from src.app.benchmark.semantic_llm_judge import evaluate_dataset as llm_judge_eval

class RequestBody(BaseModel):
    query: str
    qa_id: int

router = APIRouter(
    tags=["Jigsaw_lightRAG"],
    prefix="/jigsaw"
)

# Default QA method, query from KG and summarize the actual answer.
@router.post("/search")
def search_public(requestBody: RequestBody):
    requestBody = requestBody.model_dump()
    result = lightRAG_service.search_public(requestBody.get('query'))
    return {
        "data": result
    }

# Generate Knowledge Graph, including Phase 1 - Subgraph processing and Phase 2 - Global KG aggregation.
@router.get("/genKG")
async def custom_genKG():
    await jigsaw_service.custom_genKG()

# Batch QA test, use question and ground truth answer from dataset, save actual answer to DB.
@router.get("/batch_qa_exp")
def dataset_exp_api():
    result = batch_qa_exp(dataset=os.getenv("dataset"))
    return {
        "data": result
    } 

# Get Jaccard similarity score.
@router.get("/jaccard_exp")
def call_jaccard_exp():
    result = jaccard_exp(dataset=os.getenv("dataset"), scenario=os.getenv("scenario"))
    return {
        "data": result
    }

# Evaluate batch QA test's actual answer and ground truth, generate token recall score and context recall score.
@router.get("/prf_eval")
def prf_eval():
    precision_recall_f1_eval()

# Evaluate batch QA test's actual answer and ground truth, generate semantic similarity score.
@router.get("/semantic_judge")
def semantic_judge():
    llm_judge_eval(dataset=os.getenv("dataset"), scenario=os.getenv("scenario"))
