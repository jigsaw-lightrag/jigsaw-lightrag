'''
The corresponding source code in LightRAG V1.0.1 is omitted here. 
You can download and compile the framework yourself and then integrate the ainsert_custom_kg method provided by this method.
'''

# Highlight the Phase 2 global KG aggregation logic, need a full version of LightRAG V1.0.1 to enable this method.
async def ainsert_custom_kg(
        self,
        custom_kg: dict[str, Any],
        all_entities_map: dict[str, dict],
        all_relationships_map: dict[str, dict],
        full_doc_id: str = None,
        file_path: str = "custom_kg",
    ) -> None:
        try:
            # Insert chunks into vector storage
            all_chunks_data: dict[str, dict[str, str]] = {}
            chunk_to_source_map: dict[str, str] = {}
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = clean_text(chunk_data["content"])
                source_id = chunk_data["source_id"]
                tokens = len(
                    encode_string_by_tiktoken(
                        chunk_content, model_name=self.tiktoken_model_name
                    )
                )
                chunk_order_index = (
                    0
                    if "chunk_order_index" not in chunk_data.keys()
                    else chunk_data["chunk_order_index"]
                )
                chunk_id = compute_mdhash_id(chunk_content, prefix="chunk-")
                chunk_entry = {}
                chunk_entry.update(chunk_data)
                chunk_entry.update(
                    {
                        "content": chunk_content,
                        "source_id": source_id,
                        "tokens": tokens,
                        "chunk_order_index": chunk_order_index,
                        "full_doc_id": (
                            full_doc_id if full_doc_id is not None else source_id
                        ),
                        "file_path": file_path,  # Add file path
                    }
                )
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id

            if all_chunks_data:
                await asyncio.gather(
                    self.text_chunks.upsert(all_chunks_data),
                )

            # Insert entities into knowledge graph
            for entity_data in custom_kg.get("entities", []):
                entity_name = entity_data["entity_name"]
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                node_data: dict[str, str] = {
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_chunk_id,
                }
                if entity_name not in all_entities_map:
                    all_entities_map[entity_name] = node_data
                else:
                    old_node_data = all_entities_map[entity_name]
                    if entity_type != old_node_data.get("entity_type"):
                        entity_type = (
                            f"{entity_type}<SEP>{old_node_data.get('entity_type')}"
                        )
                    all_entities_map[entity_name] = {
                        "entity_type": entity_type,
                        "description": f"{old_node_data.get('description')}<SEP>{description}",
                        "source_id": f"{old_node_data.get('source_id')}<SEP>{source_chunk_id}",
                    }
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=all_entities_map[entity_name]
                )

            for relationship_data in custom_kg.get("relationships", []):
                src_id = relationship_data["src_id"]
                tgt_id = relationship_data["tgt_id"]
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)
                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")

                edge_data: dict[str, str] = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                    "source_id": source_id,
                    "weight": weight,
                }
                relationship_key = f"{src_id}######{tgt_id}"
                if relationship_key not in all_relationships_map:
                    all_relationships_map[relationship_key] = edge_data
                else:
                    old_edge_data = all_relationships_map[relationship_key]
                    all_relationships_map[relationship_key] = {
                        "weight": weight,
                        "keywords": f"{old_edge_data.get('keywords')}<SEP>{keywords}",
                        "description": f"{old_edge_data.get('description')}<SEP>{description}",
                        "source_id": f"{old_edge_data.get('source_id')}<SEP>{source_chunk_id}",
                    }
                # Insert edge into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data=all_relationships_map[relationship_key],
                )

            new_docs = {
                custom_kg.get("source_id"): {
                    k: v
                    for k, v in custom_kg.items()
                    if isinstance(v, (str, int, float))
                }
            }
            await self.full_docs.upsert(new_docs)

        except Exception as e:
            print(f"Error in ainsert_custom_kg: {e}")
            raise