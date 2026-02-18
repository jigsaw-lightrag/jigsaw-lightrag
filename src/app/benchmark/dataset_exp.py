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

load_dotenv()

ROOT = Path(__file__).resolve().parent

def extract_file_list_and_answer(response_data):
    """Extract file list and answer from the response data"""
    data = response_data.get('data', '')
    
    # Extract file list using regex
    file_list_match = re.search(r'<FileNameList>\[(.*?)\]</FileNameList>', data)
    file_list_str = file_list_match.group(1) if file_list_match else ''
    file_list = file_list_str.replace("'", "").replace(" ", "").split(',')

    # Get the answer (text after the FileNameList tag)
    answer = re.sub(r'<FileNameList>\[.*?\]</FileNameList>', '', data).strip()

    return file_list, answer

def call_api(question, service_url):
    """
    Call the API with the given question
    
    Args:
        question: The question to ask
        service_url: URL of the API service
        
    Returns:
        tuple: (file_list, answer) or (None, None) if error
    """
    # Prepare request payload
    payload = {
        "query": question
    }
    
    # Add headers
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    try:
        # Send request to the service
        response = requests.post(service_url, json=payload, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse response
        response_data = response.json()
        file_list, answer = extract_file_list_and_answer(response_data)
        
        print(f"  - API call successful: Response time: {response.elapsed.total_seconds():.2f}s")
        return file_list, answer
        
    except Exception as e:
        print(f"  - Error calling API: {e}")
        return None, None

def save_result_to_db(qa_id, actual_answer, actual_filelist):
    """
    Save the result to qa_exp_result table
    
    Args:
        db: Database session
        qa_id: ID from sampling_dataset_qa table
        actual_answer: The actual answer from API
        actual_filelist: The actual file list from API
    """
    db = next(db_utils.get_db())
    try:
        # Create new result record
        result = QAExpResult(
            qa_id=qa_id,
            actual_answer=actual_answer,
            actual_filelist=', '.join(actual_filelist),
            scenario = os.getenv("scenario")
        )
        
        # Add to database
        db.add(result)
        db.commit()
        print(f"  - Result saved to database for qa_id: {qa_id}")
        
    except Exception as e:
        print(f"  - Error saving to database: {e}")
        db.rollback()

def process_qa_dataset(dataset_name, service_url, qa_id=None):
    """
    Process all QA data for the specified dataset
    
    Args:
        dataset_name: Name of the dataset to process
        service_url: URL of the API service
    """
    # Get database connection
    db = next(db_utils.get_db())
    
    try:
        # Read data from sampling_dataset_qa table
        if qa_id == None:
            datas = db.query(SamplingDatasetQA).order_by(SamplingDatasetQA.id).filter(
                SamplingDatasetQA.dataset == dataset_name
            ).all()
        else:
            datas = db.query(SamplingDatasetQA).order_by(SamplingDatasetQA.id).filter(
                SamplingDatasetQA.dataset == dataset_name,
                SamplingDatasetQA.id == qa_id
            ).all()        
        print(f"Found {len(datas)} records for dataset: {dataset_name}")
        
        # Process each record
        def process_question(data, service_url):
            print(f"\nProcessing record ID: {data.id}")
            print(f"Question: {data.question}")
            for attempt in range(1):
                print(f"  Attempt {attempt+1}/3:")
                file_list, answer = call_api(data.question, service_url)
                if file_list is not None and answer is not None:
                    if qa_id is not None:
                        print(f"file_list: {file_list}, answer: {answer}")
                    save_result_to_db(data.id, answer, file_list)
                    print(f"ID {data.id}-{attempt+1} processed successfully.")
                else:
                    print(f"  - Failed to get valid response from API")
            return f"ID {data.id} processed successfully."

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(process_question, data, service_url)
                for data in datas
            ]
            for future in as_completed(futures):
                print(future.result())

        print(f"\nCompleted processing all {len(datas)} records for dataset: {dataset_name}") 
    except Exception as e:
        print(f"Error processing dataset: {e}")
    finally:
        db.close()

def dataset_exp(dataset:str):
    """Main function to run the QA processing"""
    # Configuration
    DATASET_NAME = dataset 
    SERVICE_URL = "http://127.0.0.1:8000/jigsaw/search"  
    
    print(f"Starting QA dataset processing...")
    print(f"Dataset: {DATASET_NAME}")
    print(f"API URL: {SERVICE_URL}")
    
    # Process the dataset
    process_qa_dataset(DATASET_NAME, SERVICE_URL)
    
    print("QA dataset processing completed!")

