import os
import difflib
import numpy as np
from dotenv import load_dotenv
from src.app.util.db_utils import get_db
from src.app.model.qa_exp_result import QAExpResult  
from src.app.model.sampling_dataset_qa import SamplingDatasetQA
import tiktoken

load_dotenv()

def token_f1(prediction, ground_truth, model_name="gpt-4"): # you might need to change this model_name to align with your actual LLM model
    enc = tiktoken.encoding_for_model(model_name)
    pred_tokens = set(enc.encode(prediction))
    gt_tokens = set(enc.encode(ground_truth))
    if not pred_tokens and not gt_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0, 0.0, 0.0
    tp = len(pred_tokens & gt_tokens)
    precision = tp / len(pred_tokens) if pred_tokens else 0.0
    recall = tp / len(gt_tokens) if gt_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1

def calculate_filename_score_improved(actual_files, expected_files, file_weights=None):
    if not expected_files:
        return 1.0, 1.0, 1.0
    if not file_weights:
        file_weights = {file: 1.0 for file in expected_files}
    total_weight = sum(file_weights.values())
    score = 0
    matched = 0
    for exp_file in expected_files:
        weight = file_weights.get(exp_file, 1.0)
        if exp_file in actual_files:
            score += weight
            matched += 1
        else:
            best_partial_match = 0
            for act_file in actual_files:
                similarity = difflib.SequenceMatcher(None, exp_file, act_file).ratio()
                best_partial_match = max(best_partial_match, similarity * 0.5)
            score += weight * best_partial_match
    if len(actual_files) > len(expected_files):
        excess_penalty = 0.1 * (len(actual_files) - len(expected_files))
        score = max(0, score - excess_penalty * total_weight)
    p = matched / len(actual_files) if actual_files else 0.0
    r = matched / len(expected_files) if expected_files else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    return p, r, f1

def prf_eval():
    db = next(get_db())
    try:
        qa_rows = db.query(SamplingDatasetQA).filter(
            SamplingDatasetQA.dataset == os.getenv("dataset")
            ).order_by(SamplingDatasetQA.id).all()
        qa_ids = [qa.id for qa in qa_rows]

        exp_rows = db.query(QAExpResult).filter(
            QAExpResult.qa_id.in_(qa_ids),
            QAExpResult.scenario == os.getenv("scenario")
            ).order_by(QAExpResult.qa_id, QAExpResult.create_at.desc()).all()
        exp_map = {}
        for row in exp_rows:
            if row.qa_id not in exp_map:
                exp_map[row.qa_id] = row

        print("=== token recall evaluation ===")
        all_precisions, all_recalls, all_f1s = [], [], []
        for qa in qa_rows:
            qa_id = qa.id
            q = qa.question
            gt = qa.answer.strip() if qa.answer else ""
            exp_row = exp_map.get(qa_id)
            if not exp_row:
                print(f"Q{qa_id}: no result, skipped.")
                continue
            pred = exp_row.actual_answer.strip() if exp_row.actual_answer else ""
            p, r, f = token_f1(pred, gt)
            all_precisions.append(p)
            all_recalls.append(r)
            all_f1s.append(f)
            print(f"Q{qa_id}:")
            print(f"  Question: {q}")
            print(f"  Model Answer: {pred[:100]}...")
            print(f"  Gold Answer: {gt[:100]}...")
            print(f"  Precision: {p:.4f}  Recall: {r:.4f}  F1: {f:.4f}")
        print(f"[Answer] Macro Precision: {np.mean(all_precisions):.4f}")
        print(f"[Answer] Token Recall: {np.mean(all_recalls):.4f}") # token recall used in ED3 evaluation
        print(f"[Answer] Macro F1: {np.mean(all_f1s):.4f}")

        print("\n=== Context recall evaluation ===")
        ctx_precisions, ctx_recalls, ctx_f1s = [], [], []
        for qa in qa_rows:
            qa_id = qa.id
            exp_row = exp_map.get(qa_id)
            if not exp_row:
                continue
            gold_ctx = [x.strip() for x in (qa.filelist or "").split(',') if x.strip()]
            actual_ctx = [x.strip() for x in (exp_row.actual_filelist or "").split(',') if x.strip()]
            p, r, f = calculate_filename_score_improved(actual_ctx, gold_ctx)
            ctx_precisions.append(p)
            ctx_recalls.append(r)
            ctx_f1s.append(f)
            print(f"Q{qa_id}:")
            print(f"  Retrieved Context Chunks: {actual_ctx}")
            print(f"  Gold Context Chunks: {gold_ctx}")
            print(f"  Precision: {p:.4f}  Recall: {r:.4f}  F1: {f:.4f}")
        print(f"[Context] Macro Precision: {np.mean(ctx_precisions):.4f}")
        print(f"[Context] Context Recall: {np.mean(ctx_recalls):.4f}") # context recall used in ED3 evaluation
        print(f"[Context] Macro F1: {np.mean(ctx_f1s):.4f}")

    finally:
        db.close()

