# README
This code is based on open-source framework LightRAG V1.0.1. 
It implemented the critical algorithm logic of our paper: 
"Jigsaw-like knowledge graph generation: A study on generalization patterns with a LightRAG implementation".

## Requirements
- LightRAG V1.0.1 and relevant dependencies, need to be installed from source (pip install -e .)
- Python env. (>=3.10.x)
- requirements.txt: Python library dependencies of our implementation.
- RMDB (Obtain the lifecycle data of the original corpus, or the document collection data corresponding to the effectively maintained external subgraph pool, and other relevant data need to be saved from experiments.)
- Large Language Model (e.g., OpenAI GPT series LLM models)
- Compatible Embedding Model (e.g. text-embedding-3 series embedding models)
- .env: general system params.
- Dataset: the dataset you select to finish the experiment.

## Core algorithm logic
- Jigsaw_service: custom_genKG() method implemented Phase 1 (single subgraph generation) and Phase 2 (subgraph pool aggregation), follow the same deduplication logic: strict string-matching.
- lightRAG_service: search_public(query:str) method inherited from vanilla LightRAG framework, used for retrieving answers from KG.
- Other entry methods are listed in jigsaw_api.py

## Experiment workflow
- For Jigsaw-LightRAG, you can finish the major ED1, ED2, ED3 experiments as below:
    1. Get your environment ready following Section Requirements.
    2. Get your exp. dataset sampling subset files ready, manually generate projection data into DB table subgraph_pool_mapping, then extract dataset content into 
    DB table sampling_dataset_qa.
    3. Configure your .env params to real LLM / Embedding model endpoints, real DB connection, dataset, and exp. scenario info.
    4. Start this whole FastAPI application by: uvicorn src.app.main:app
    5. Access this instance through web browser such as Chrome: http://127.0.0.1:8000/docs#
    6. Call /jigsaw/genKG to generate the KG of your current exp. dataset.
    7. After step. 6, you will get token consumption log data in DB table: request_seq and request_token, collect data by scenario to get ED1 result, then collect entity and relationship quantity from command line record, this are ED2 entity and relationship quantity results.
    8. Call /jigsaw/jaccard_exp to get Jaccard similarity result in ED2.
    9. Call /jigsaw/batch_qa_exp to generate batch QA test results based on sampling_dataset_qa, the results will be saved into DB table qa_exp_result.
    10. Call /jigsaw/prf_eval to get token recall and context recall score, this are ED3 token recall and context recall results.
    11. Call /jigsaw/semantic_judge to get semantic similarity score, this is ED3 semantic similarity result.
    12. You can change dataset and scenario to repeat step. 6 to step. 11, then finish new iteration of your experiment.

- For Vanilla-LightRAG and GraphRAG, you can refer to Jigsaw-LightRAG's workflow and finish the baseline experiments.

## Exclusion
Althrough LightRAG is an open-source KG-based RAG framework, and it follows MIT license, but for safety and exemption considerations, we did not include LightRAG framework in our supplementary code package, you can access the official LightRAG GitHub site to get proper codes and libraries：
[LightRAG base](https://github.com/HKUDS/LightRAG)