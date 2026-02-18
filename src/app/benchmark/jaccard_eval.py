from pathlib import Path
import networkx as nx
import pandas as pd
from itertools import combinations
import json

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
GRAPHRAG_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent

DATASETS = ["PubMedQA", "QASPER", "LongBench"]
FRAMEWORKS = ["Jigsaw_LightRAG", "Vanilla_LightRAG", "GRAPHRAG"]
SCENARIOS = {
    "Jigsaw_LightRAG": ["ADD", "MODIFY", "DELETE"],
    "Vanilla_LightRAG": ["ADD"],
    "GRAPHRAG": ["ADD"]
}

VERSION_MAPPING = {
    "PubMedQA": ["1", "2", "3"],
    "QASPER": ["1", "2", "3"],
    "LongBench": ["1", "2", "3"]
}

def jaccard_lightrag(graph1_path, graph2_path):
    try:
        g1 = nx.read_graphml(graph1_path)
        g2 = nx.read_graphml(graph2_path)
        
        nodes1 = set(g1.nodes())
        nodes2 = set(g2.nodes())
        node_jaccard = len(nodes1 & nodes2) / len(nodes1 | nodes2) if nodes1 or nodes2 else 0.0
        
        edges1 = set(
            (u, v, d.get("relationship_type", "")) 
            for u, v, d in g1.edges(data=True)
        )
        edges2 = set(
            (u, v, d.get("relationship_type", "")) 
            for u, v, d in g2.edges(data=True)
        )
        edge_jaccard = len(edges1 & edges2) / len(edges1 | edges2) if edges1 or edges2 else 0.0
        
        return {"node_jaccard": node_jaccard, "edge_jaccard": edge_jaccard}
    except Exception as e:
        print(f"Error processing LightRAG files: {graph1_path}, {graph2_path}")
        print(f"Error: {e}")
        return {"node_jaccard": 0.0, "edge_jaccard": 0.0}

def jaccard_graphrag(kg1_entities_path, kg1_relations_path, kg2_entities_path, kg2_relations_path):
    try:
        kg1_entities = pd.read_parquet(kg1_entities_path)
        kg1_relations = pd.read_parquet(kg1_relations_path)
        kg2_entities = pd.read_parquet(kg2_entities_path)
        kg2_relations = pd.read_parquet(kg2_relations_path)
        
        ents1 = set(kg1_entities["id"])
        ents2 = set(kg2_entities["id"])
        entity_jaccard = len(ents1 & ents2) / len(ents1 | ents2) if ents1 or ents2 else 0.0
        
        def make_relation_fingerprint(df):
            return set(
                (str(row["source"]), str(row["target"]), str(row.get("relationship_type", "")))
                for _, row in df.iterrows()
            )
        
        rels1 = make_relation_fingerprint(kg1_relations)
        rels2 = make_relation_fingerprint(kg2_relations)
        relation_jaccard = len(rels1 & rels2) / len(rels1 | rels2) if rels1 or rels2 else 0.0
        
        return {"entity_jaccard": entity_jaccard, "relation_jaccard": relation_jaccard}
    except Exception as e:
        print(f"Error processing GraphRAG files: {kg1_entities_path}, {kg2_entities_path}")
        print(f"Error: {e}")
        return {"entity_jaccard": 0.0, "relation_jaccard": 0.0}

def find_lightrag_paths():
    paths = {}
    for dataset in DATASETS:
        paths[dataset] = {}
        for framework in ["Jigsaw_LightRAG", "Vanilla_LightRAG"]:
            paths[dataset][framework] = {}
            for scenario in SCENARIOS[framework]:
                paths[dataset][framework][scenario] = {}
                for version in VERSION_MAPPING[dataset]:
                    dir_name = f"{framework}_{dataset}_{scenario}_{version}"
                    graph_path = ROOT / dir_name / "PUBLIC" / "AI_KG" / "graph_chunk_entity_relation.graphml"
                    if graph_path.exists():
                        paths[dataset][framework][scenario][version] = graph_path
    return paths

def find_graphrag_paths():
    paths = {}
    for dataset in DATASETS:
        dataset_lower = dataset.lower()
        paths[dataset] = {}
        for scenario in SCENARIOS["GRAPHRAG"]:
            paths[dataset][scenario] = {}
            for version in VERSION_MAPPING[dataset]:
                dir_name = f"{dataset_lower}_{scenario.lower()}_{version}"
                entities_path = GRAPHRAG_ROOT / "gpt_poc" / dir_name / "output" / "create_final_entities.parquet"
                relations_path = GRAPHRAG_ROOT / "gpt_poc" / dir_name / "output" / "create_final_relationships.parquet"
                if entities_path.exists() and relations_path.exists():
                    paths[dataset][scenario][version] = (entities_path, relations_path)
    return paths

def calculate_all_jaccard_scores(dataset, scenario):
    results = []
    
    # Jigsaw-LightRAG and Vanilla-LightRAG
    lightrag_paths = find_lightrag_paths()
    for dataset in lightrag_paths:
        for framework in lightrag_paths[dataset]:
            for scenario in lightrag_paths[dataset][framework]:
                versions = list(lightrag_paths[dataset][framework][scenario].keys())
                
                for ver1, ver2 in combinations(versions, 2):
                    path1 = lightrag_paths[dataset][framework][scenario][ver1]
                    path2 = lightrag_paths[dataset][framework][scenario][ver2]
                    
                    scores = jaccard_lightrag(path1, path2)
                    
                    results.append({
                        "dataset": dataset,
                        "framework": framework,
                        "scenario": scenario,
                        "comparison": f"{ver1}-vs-{ver2}",
                        "node_jaccard": scores["node_jaccard"],
                        "edge_jaccard": scores["edge_jaccard"],
                        "entity_jaccard": None,
                        "relation_jaccard": None
                    })
    
    # GraphRAG
    graphrag_paths = find_graphrag_paths()
    for dataset in graphrag_paths:
        for scenario in graphrag_paths[dataset]:
            versions = list(graphrag_paths[dataset][scenario].keys())
            
            for ver1, ver2 in combinations(versions, 2):
                path1 = graphrag_paths[dataset][scenario][ver1]
                path2 = graphrag_paths[dataset][scenario][ver2]
                
                scores = jaccard_graphrag(path1[0], path1[1], path2[0], path2[1])
                
                results.append({
                    "dataset": dataset,
                    "framework": "GRAPHRAG",
                    "scenario": scenario,
                    "comparison": f"{ver1}-vs-{ver2}",
                    "node_jaccard": None,
                    "edge_jaccard": None,
                    "entity_jaccard": scores["entity_jaccard"],
                    "relation_jaccard": scores["relation_jaccard"]
                })
    
    return results

