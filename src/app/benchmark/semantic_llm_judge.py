import importlib, os
import requests
import time
import random
import re
from datetime import datetime
from pathlib import Path
from src.app.util import db_utils
from src.app.model.sampling_dataset_qa import SamplingDatasetQA
from src.app.model.qa_exp_result import QAExpResult  
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json
import statistics
from openai import AzureOpenAI

load_dotenv()
ROOT = Path(__file__).resolve().parent

# === Azure OpenAI Client ===
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")


def call_llm_judge_api(standard_answer, predicted_answer):
    """
    Call LLM API to evaluate semantic similarity between standard and predicted answers
    
    Args:
        standard_answer: The ground truth answer
        predicted_answer: The LLM's predicted answer
        service_url: URL of the API service
        
    Returns:
        int: Score from 1-5, or None if error
    """
    
    # LLM-as-a-Judge prompt for semantic similarity evaluation
    judge_prompt = f"""You are an expert evaluator tasked with assessing the semantic similarity between a standard answer and a predicted answer. Your goal is to determine how well the predicted answer aligns with the standard answer in terms of meaning and content.

**Evaluation Criteria:**
- Score 1: Completely different - The predicted answer has no semantic overlap with the standard answer
- Score 2: Mostly different - The predicted answer has minimal semantic overlap with the standard answer  
- Score 3: Partially similar - The predicted answer shares some key concepts or ideas with the standard answer
- Score 4: Mostly similar - The predicted answer captures most of the meaning and key points of the standard answer
- Score 5: Semantically equivalent - The predicted answer conveys the same meaning as the standard answer, even if worded differently

**Instructions:**
1. Carefully analyze both answers for semantic content and meaning
2. Consider conceptual overlap, factual accuracy, and completeness
3. Focus on meaning rather than exact wording or style
4. Provide only a single integer score from 1 to 5

**Standard Answer:**
{standard_answer}

**Predicted Answer:**
{predicted_answer}

**Your Score (1-5):**"""
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": judge_prompt}
            ],
            temperature=0.0
        )
        response_data = response.choices[0].message.content.strip()
        
        score_match = re.search(r'\b([1-5])\b', response_data)
        if score_match:
            score = int(score_match.group(1))
            print(f"  - Judge API successful: Score = {score}")
            return score
        else:
            print(f"  - Could not extract valid score from judge response: {response_data}")
            return None
            
    except Exception as e:
        print(f"  - Error calling judge API: {e}")
        return None

def evaluate_qa_results(dataset_name, scenario):
    """
    Evaluate the QA results using LLM-as-a-judge for semantic similarity
    
    Args:
        dataset_name: Name of the dataset to evaluate
        service_url: URL of the API service
    """
    # Get database connection
    db = next(db_utils.get_db())
    
    try:
        # Join sampling_dataset_qa with qa_exp_result to get both standard and predicted answers
        query = db.query(
            SamplingDatasetQA.id,
            SamplingDatasetQA.question,
            SamplingDatasetQA.answer.label('standard_answer'),
            QAExpResult.actual_answer.label('predicted_answer')
        ).join(
            QAExpResult, SamplingDatasetQA.id == QAExpResult.qa_id
        ).filter(
            SamplingDatasetQA.dataset == dataset_name,
            QAExpResult.scenario == scenario
        ).order_by(SamplingDatasetQA.id)
        
        results = query.all()
        print(f"Found {len(results)} QA pairs for evaluation in: {scenario}")
        
        if not results:
            print("No results found for evaluation. Please run the QA processing first.")
            return
        
        evaluation_scores = []
        detailed_results = []
        
        # Process each QA pair for evaluation
        def evaluate_single_qa(result):
            qa_id, question, standard_answer, predicted_answer = result
            
            print(f"\nEvaluating QA ID: {qa_id}")
            print(f"Question: {question[:100]}...")
            print(f"Standard Answer: {standard_answer[:100]}...")
            print(f"Predicted Answer: {predicted_answer[:100]}...")
            
            # Get semantic similarity score from LLM judge
            for attempt in range(3):  # Retry up to 3 times
                print(f"  Judge Attempt {attempt+1}/3:")
                score = call_llm_judge_api(standard_answer, predicted_answer)
                
                if score is not None:
                    evaluation_scores.append(score)
                    detailed_results.append({
                        'qa_id': qa_id,
                        'question': question,
                        'standard_answer': standard_answer,
                        'predicted_answer': predicted_answer,
                        'similarity_score': score
                    })
                    print(f"  - QA ID {qa_id} evaluated successfully. Score: {score}")
                    return score
                else:
                    print(f"  - Failed to get valid score for QA ID {qa_id}")
            
            print(f"  - All attempts failed for QA ID {qa_id}")
            return None

        # Use ThreadPoolExecutor for parallel evaluation
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(evaluate_single_qa, result)
                for result in results
            ]
            
            for future in as_completed(futures):
                score = future.result()
                if score is None:
                    print("Warning: Some evaluations failed")

        # Statistical Analysis
        if evaluation_scores:
            print("\n" + "="*80)
            print("EVALUATION RESULTS SUMMARY")
            print("="*80)
            
            # Basic statistics
            total_questions = len(evaluation_scores)
            mean_score = statistics.mean(evaluation_scores)
            median_score = statistics.median(evaluation_scores)
            mode_score = statistics.mode(evaluation_scores) if len(set(evaluation_scores)) < len(evaluation_scores) else "No mode"
            std_dev = statistics.stdev(evaluation_scores) if len(evaluation_scores) > 1 else 0
            
            print(f"Dataset: {dataset_name}")
            print(f"Scenario: {scenario}")
            print(f"Total Questions Evaluated: {total_questions}")
            print(f"Mean Similarity Score: {mean_score:.2f}")
            print(f"Median Similarity Score: {median_score}")
            print(f"Mode Similarity Score: {mode_score}")
            print(f"Standard Deviation: {std_dev:.2f}")
            print(f"Score Range: {min(evaluation_scores)} - {max(evaluation_scores)}")
            
            # Score distribution
            score_distribution = {i: evaluation_scores.count(i) for i in range(1, 6)}
            print(f"\nScore Distribution:")
            for score, count in score_distribution.items():
                percentage = (count / total_questions) * 100
                print(f"  Score {score}: {count} questions ({percentage:.1f}%)")
            
            # Performance categories
            excellent = sum(1 for score in evaluation_scores if score == 5)
            good = sum(1 for score in evaluation_scores if score == 4)
            fair = sum(1 for score in evaluation_scores if score == 3)
            poor = sum(1 for score in evaluation_scores if score <= 2)
            
            print(f"\nPerformance Categories:")
            print(f"  Excellent (Score 5): {excellent} ({excellent/total_questions*100:.1f}%)")
            print(f"  Good (Score 4): {good} ({good/total_questions*100:.1f}%)")
            print(f"  Fair (Score 3): {fair} ({fair/total_questions*100:.1f}%)")
            print(f"  Poor (Score 1-2): {poor} ({poor/total_questions*100:.1f}%)")
                    
        else:
            print("\nNo valid evaluation scores obtained. Please check the evaluation process.")
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
    finally:
        db.close()


def evaluate_dataset(dataset: str, scenario: str):
    """Main function to run the evaluation""" 
    print(f"Starting QA dataset evaluation...")
    evaluate_qa_results(dataset, scenario)
    print("QA dataset evaluation completed!")


