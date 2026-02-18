import json
import os
from pathlib import Path
from typing import List
from bs4 import BeautifulSoup, Tag
from pathlib import Path

from src.app.model.subgraph_pool_mapping import SubgraphPoolMapping
from src.app.util import db_utils
from src.app.lightRAG.lightrag.utils import compute_mdhash_id, clean_text
from src.app.service.lightRAG_service import (
    LightRAG,
    EmbeddingFunc,
    llm_model_func,
    embedding_dimension,
    embedding_func,
    del_KG_data,
    record_query,
)
from dotenv import load_dotenv
import shutil
from constant import ROOT

load_dotenv()

def create_single_json(rag_workspace: Path = ROOT / "single_kg"):
    s_doc = {"chunks": [], "entities": [], "relationships": []}
    # get content
    content_json_file = rag_workspace / "kv_store_full_docs.json"
    doc_content = get_doc_content(content_json_file)
    for _id in doc_content:
        s_doc["source_id"] = _id
        s_doc.update(doc_content.get(_id))

    doc_chunks = get_doc_chunks(rag_workspace / "kv_store_text_chunks.json")
    for _id in doc_chunks:
        chunk = {}
        chunk["source_id"] = _id
        chunk.update(doc_chunks.get(_id))
        s_doc["chunks"].append(chunk)

    graphml_file = rag_workspace / "graph_chunk_entity_relation.graphml"
    xml_content = get_graphml(graphml_file)
    soup = BeautifulSoup(xml_content, "xml")
    s_doc["entities"] = get_entity(soup.find_all("node"))
    s_doc["relationships"] = get_edges(soup.find_all("edge"))

    json_file_dir = ROOT / "json_dir"
    json_file_dir.mkdir(parents=True, exist_ok=True)
    file_name = s_doc.get("source_id")[4:]
    json_file = json_file_dir / Path(file_name + ".json")
    with open(json_file, "w", encoding="utf-8") as file:
        json.dump(s_doc, file, indent=4, ensure_ascii=False)
    return file_name


def get_doc_content(file_path) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        content = json.load(file)
        return content


def get_doc_chunks(file_path) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        content = json.load(file)
        return content


def get_entity(nodes: List[Tag]):
    entity_list = []
    for node in nodes:
        entity_list.append(
            {
                "entity_name": node.get("id"),
                "entity_type": node.find("data", {"key": "d0"}).text,
                "description": node.find("data", {"key": "d1"}).text,
                "source_id": node.find("data", {"key": "d2"}).text,
            }
        )
    return entity_list


def get_edges(nodes: List[Tag]):
    edge_list = []
    for node in nodes:
        edge_list.append(
            {
                "src_id": node.get("source"),
                "tgt_id": node.get("target"),
                "weight": node.find("data", {"key": "d3"}).text,
                "description": node.find("data", {"key": "d4"}).text,
                "keywords": node.find("data", {"key": "d5"}).text,
                "source_id": node.find("data", {"key": "d6"}).text,
            }
        )
    return edge_list

def get_graphml(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

async def custom_insert(
    working_dir: str | Path = "./custom_kg/",
    files: List[str] = [],
):
    if working_dir.exists():
        shutil.rmtree(working_dir)
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    pipeline_rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8196,
            func=embedding_func,
        ),
    )
    all_entities_map: dict[str, dict] = {}
    all_relationships_map: dict[str, dict] = {}
    for file in files:
        await pipeline_rag.ainsert_custom_kg(
            get_custom_kg_dict(file_path=file),
            all_entities_map=all_entities_map,
            all_relationships_map=all_relationships_map,
        )
    base_dir = Path(working_dir) / "graph_chunk_entity_relation.graphml"
    xml_content = get_graphml(base_dir)
    soup = BeautifulSoup(xml_content, "xml")
    all_entities_data = get_entity(soup.find_all("node"))

    data_for_vdb = {
        compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
            "content": dp["entity_name"] + dp["description"],
            "entity_name": dp["entity_name"],
        }
        for dp in all_entities_data
    }
    await pipeline_rag.entities_vdb.upsert(data_for_vdb)
    chunks_dir = Path(working_dir) / "kv_store_text_chunks.json"
    s_doc = {"chunks": []}
    doc_chunks = get_doc_chunks(chunks_dir)
    for _id in doc_chunks:
        chunk = {}
        chunk["source_id"] = _id
        chunk.update(doc_chunks.get(_id))
        s_doc["chunks"].append(chunk)
    await pipeline_rag.chunks_vdb.upsert(s_doc["chunks"]) 
    await pipeline_rag._insert_done() 

def get_custom_kg_dict(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# Phase 2: Global KG aggregation
async def merge_kg():
    base_kg_dir = ROOT / "KG_NEW"  
    json_file_dir = ROOT / "json_dir"
    json_file_dir.mkdir(parents=True, exist_ok=True)

    db = next(db_utils.get_db())
    '''
    Aggregate all Persistent documents' subgraphs from your dataset into global KG. 
    For Deleted document(s), the cur_status should be marked as 'Deleted' manually in DB.
    '''
    datas = db.query(SubgraphPoolMapping).filter(SubgraphPoolMapping.cur_status == 'Persistent').all()
    base_entrys: dict[str, List[SubgraphPoolMapping]] = {}
    for inst in datas:
        base_entry = str(inst.base_entry).strip()
        if not inst.base_entry or not base_entry:
            continue
        if not base_entrys.get(base_entry):
            base_entrys[base_entry] = []
        base_entrys[base_entry].append(inst)
    for b in base_entrys:
        working_dir: Path = base_kg_dir / b
        await custom_insert(
            working_dir=working_dir,
            files=list(
                set(map(lambda x: json_file_dir / f"{str(x.md5)}.json", base_entrys[b]))
            ),
        )
        for inst in base_entrys[b]:
            db.commit()

# Phase 1: Subgraph processing
async def single_genKG():
    txt_dir = ROOT / "../test"

    json_file_dir = ROOT / "json_dir"
    json_file_dir.mkdir(parents=True, exist_ok=True)

    db = next(db_utils.get_db())
    ''' 
    Mark 'New' and 'Modified' documents from your dataset in DB projection data, you can organize the documents by simulating 
    all New, Modified, Persistent, Deleted lifecycle status.
    '''
    datas = db.query(SubgraphPoolMapping).filter(SubgraphPoolMapping.cur_status.in_(['New', 'Modified'])).all() 
    response = 1
    for inst in datas:
        text_file_path = txt_dir / inst.filepath
        with open(text_file_path, "r", encoding="utf-8") as file:
            content: str = file.read()
        working_dir = ROOT / "single_kg"
        if working_dir.exists():
            shutil.rmtree(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)
        try:
            p_rag = LightRAG(
                working_dir=working_dir,
                llm_model_func=llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=embedding_dimension,
                    max_token_size=8196,
                    func=embedding_func,
                ),
            )
            record_inst = record_query(content=str(inst.filename), req_type="GENERATE")
            await p_rag.ainsert(
                string_or_strings=content,
                file_or_files=str(inst.filename),
                req_id=record_inst.req_id,
            )
            kg_file_md5 = create_single_json()
            inst.md5 = kg_file_md5
            inst.cur_status = "Persistent"
            db.commit()
        except Exception as e:
            print(e)
            print(f"inst.id: {inst.id} , inst.filename: {inst.filename}")
            response = 0
    return response

async def custom_genKG():
    try_times = 3
    ret = 0
    while try_times > 0 and ret != 1:
        try:
            ret = await single_genKG()
            try_times -= 1
        except Exception as e:
            print(e)
            try_times -= 1
    if ret != 1:
        return "FAILED"
    try_times = 3
    ret = 0
    while try_times > 0 and ret != 1:
        try:
            await merge_kg()
            ret = 1
        except Exception as e:
            print(e)
            ret = 0
        try_times -= 1
    if ret == 0:
        return "FAILED"
    dir2 = ROOT / "KG_NEW"
    dir1 = ROOT / "KG"
    del_KG_data(dir1)
    if os.path.exists(dir1):
        shutil.rmtree(dir1)
    shutil.move(dir2, dir1)
    return "SUCCESS"

