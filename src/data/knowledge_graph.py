"""
Knowledge graph construction for TCM target prioritization system.
"""
import os
import json
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Union
from collections import defaultdict
import torch
from torch_geometric.data import Data, HeteroData
from src.config import DataConfig

class KnowledgeGraph:
    """Knowledge graph for TCM target prioritization."""
    
    def __init__(self, config: DataConfig):
        """
        Initialize knowledge graph.
        
        Args:
            config: Data configuration.
        """
        self.config = config
        self.entity_types = config.entity_types
        self.relation_types = config.relation_types
        
        # Entity and relation mappings
        self.entity_to_idx = {entity_type: {} for entity_type in self.entity_types}
        self.idx_to_entity = {entity_type: {} for entity_type in self.entity_types}
        self.relation_to_idx = {rel: idx for idx, rel in enumerate(self.relation_types)}
        self.idx_to_relation = {idx: rel for idx, rel in enumerate(self.relation_types)}
        
        # Entity counts
        self.entity_counts = {entity_type: 0 for entity_type in self.entity_types}
        
        # Graph representation
        self.nx_graph = nx.MultiDiGraph()
        self.triplets = []
        self.confidence_scores = {}
        
        # Store compound-target pairs for model training
        self.compound_target_pairs = []
    
    def add_compound_target_pair(self, compound_id: str, target_id: str) -> None:
        """
        Add compound-target pair information to the graph.
        
        Args:
            compound_id: Compound identifier.
            target_id: Target identifier.
        """
        if not hasattr(self, 'compound_target_pairs'):
            self.compound_target_pairs = []
        
        self.compound_target_pairs.append((compound_id, target_id))
        
    def add_entity(self, entity_id: str, entity_type: str) -> int:
        """
        Add entity to knowledge graph.
        
        Args:
            entity_id: Entity identifier.
            entity_type: Entity type.
            
        Returns:
            Entity index.
        """
        if entity_type not in self.entity_types:
            raise ValueError(f"Unknown entity type: {entity_type}")
        
        if entity_id not in self.entity_to_idx[entity_type]:
            idx = self.entity_counts[entity_type]
            self.entity_to_idx[entity_type][entity_id] = idx
            self.idx_to_entity[entity_type][idx] = entity_id
            self.entity_counts[entity_type] += 1
            
            # Add to NetworkX graph
            self.nx_graph.add_node(
                (entity_type, idx),
                entity_id=entity_id,
                entity_type=entity_type
            )
        
        return self.entity_to_idx[entity_type][entity_id]
    
    def add_triplet(
        self,
        head_id: str,
        head_type: str,
        relation: str,
        tail_id: str,
        tail_type: str,
        confidence: float = 1.0
    ) -> None:
        """
        Add triplet to knowledge graph.
        
        Args:
            head_id: Head entity identifier.
            head_type: Head entity type.
            relation: Relation type.
            tail_id: Tail entity identifier.
            tail_type: Tail entity type.
            confidence: Confidence score of the triplet.
        """
        if relation not in self.relation_types:
            raise ValueError(f"Unknown relation type: {relation}")
        
        head_idx = self.add_entity(head_id, head_type)
        tail_idx = self.add_entity(tail_id, tail_type)
        relation_idx = self.relation_to_idx[relation]
        
        triplet = (
            (head_type, head_idx),
            relation,
            (tail_type, tail_idx)
        )
        self.triplets.append(triplet)
        self.confidence_scores[triplet] = confidence
        
        # Add to NetworkX graph
        self.nx_graph.add_edge(
            (head_type, head_idx),
            (tail_type, tail_idx),
            relation=relation,
            confidence=confidence
        )
    
    def add_triplets_from_file(self, file_path: str, format: str = "csv") -> None:
        """
        Add triplets from file.
        
        Args:
            file_path: Path to the file containing triplets.
            format: File format ("csv", "tsv", "json").
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if format.lower() == "csv":
            df = pd.read_csv(file_path)
        elif format.lower() == "tsv":
            df = pd.read_csv(file_path, sep="\t")
        elif format.lower() == "json":
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                raise ValueError("JSON format should be a list of triplets")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Validate required columns
        required_columns = ["head_id", "head_type", "relation", "tail_id", "tail_type"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        # Add confidence column if not present
        if "confidence" not in df.columns:
            df["confidence"] = 1.0
        
        # Add triplets
        for _, row in df.iterrows():
            self.add_triplet(
                head_id=row["head_id"],
                head_type=row["head_type"],
                relation=row["relation"],
                tail_id=row["tail_id"],
                tail_type=row["tail_type"],
                confidence=row["confidence"]
            )
    
    def merge_from_multiple_sources(self, file_paths: List[str], formats: List[str]) -> None:
        """
        Merge triplets from multiple sources.
        
        Args:
            file_paths: Paths to the files containing triplets.
            formats: File formats.
        """
        if len(file_paths) != len(formats):
            raise ValueError("Number of file paths must match number of formats")
        
        for file_path, format in zip(file_paths, formats):
            self.add_triplets_from_file(file_path, format)
    
    def create_unified_identifiers(self) -> None:
        """Create unified identifiers for entities."""
        # Create mapping of equivalent entities
        equivalence_map = defaultdict(set)
        
        # Find equivalence relationships in the graph
        for head, relation, tail in self.triplets:
            if relation == "same_as":
                head_id = self.idx_to_entity[head[0]][head[1]]
                tail_id = self.idx_to_entity[tail[0]][tail[1]]
                equivalence_map[head_id].add(tail_id)
                equivalence_map[tail_id].add(head_id)
        
        # Ensure transitive closure of equivalence relation
        changed = True
        while changed:
            changed = False
            for entity_id, equiv_set in equivalence_map.items():
                for equiv_id in list(equiv_set):
                    if equiv_id in equivalence_map:
                        for transitive_equiv in equivalence_map[equiv_id]:
                            if transitive_equiv not in equiv_set and transitive_equiv != entity_id:
                                equiv_set.add(transitive_equiv)
                                changed = True
        
        # Create unified identifiers
        unified_ids = {}
        for entity_id, equiv_set in equivalence_map.items():
            # Use the shortest ID as the unified ID
            unified_id = min([entity_id] + list(equiv_set), key=len)
            unified_ids[entity_id] = unified_id
            for equiv_id in equiv_set:
                unified_ids[equiv_id] = unified_id
        
        # Update entity mappings and graph
        for entity_type in self.entity_types:
            for entity_id, idx in list(self.entity_to_idx[entity_type].items()):
                if entity_id in unified_ids:
                    unified_id = unified_ids[entity_id]
                    if unified_id != entity_id:
                        # Update mapping
                        self.entity_to_idx[entity_type][unified_id] = idx
                        self.idx_to_entity[entity_type][idx] = unified_id
                        # Remove old mapping
                        del self.entity_to_idx[entity_type][entity_id]
        
        # Update triplets
        updated_triplets = []
        updated_confidence_scores = {}
        
        for triplet in self.triplets:
            head, relation, tail = triplet
            head_type, head_idx = head
            tail_type, tail_idx = tail
            
            # Skip "same_as" relations
            if relation == "same_as":
                continue
            
            head_id = self.idx_to_entity[head_type][head_idx]
            tail_id = self.idx_to_entity[tail_type][tail_idx]
            
            updated_triplet = (
                (head_type, head_idx),
                relation,
                (tail_type, tail_idx)
            )
            
            updated_triplets.append(updated_triplet)
            updated_confidence_scores[updated_triplet] = self.confidence_scores[triplet]
        
        self.triplets = updated_triplets
        self.confidence_scores = updated_confidence_scores
        
        # Rebuild NetworkX graph
        self.nx_graph = nx.MultiDiGraph()
        for entity_type in self.entity_types:
            for idx, entity_id in self.idx_to_entity[entity_type].items():
                self.nx_graph.add_node(
                    (entity_type, idx),
                    entity_id=entity_id,
                    entity_type=entity_type
                )
        
        for triplet in self.triplets:
            head, relation, tail = triplet
            self.nx_graph.add_edge(
                head,
                tail,
                relation=relation,
                confidence=self.confidence_scores[triplet]
            )
    
    def to_torch_geometric(self, node_features: Dict[str, torch.Tensor] = None) -> HeteroData:
        """
        Convert knowledge graph to PyTorch Geometric HeteroData.
        
        Args:
            node_features: Node features for each entity type.
            
        Returns:
            PyTorch Geometric HeteroData.
        """
        data = HeteroData()
        
        # Add node features
        if node_features is not None:
            for entity_type in self.entity_types:
                if entity_type in node_features:
                    data[entity_type].x = node_features[entity_type]
                else:
                    # Create dummy features if not provided
                    num_entities = self.entity_counts[entity_type]
                    data[entity_type].x = torch.zeros(num_entities, 1)
        else:
            # Create dummy features if not provided
            for entity_type in self.entity_types:
                num_entities = self.entity_counts[entity_type]
                data[entity_type].x = torch.zeros(num_entities, 1)
        
        # Add edges
        for relation in self.relation_types:
            for head_type in self.entity_types:
                for tail_type in self.entity_types:
                    # Collect edges of this type
                    edge_indices = []
                    edge_weights = []
                    
                    for triplet in self.triplets:
                        (h_type, h_idx), rel, (t_type, t_idx) = triplet
                        if h_type == head_type and t_type == tail_type and rel == relation:
                            edge_indices.append((h_idx, t_idx))
                            edge_weights.append(self.confidence_scores[triplet])
                    
                    if edge_indices:
                        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
                        edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
                        
                        data[head_type, relation, tail_type].edge_index = edge_index
                        data[head_type, relation, tail_type].edge_attr = edge_attr
        
        return data
    
    def get_entity_neighbors(self, entity_id: str, entity_type: str) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Get neighbors of an entity.
        
        Args:
            entity_id: Entity identifier.
            entity_type: Entity type.
            
        Returns:
            Dictionary mapping relation types to lists of (neighbor_id, neighbor_type, confidence) tuples.
        """
        if entity_type not in self.entity_types:
            raise ValueError(f"Unknown entity type: {entity_type}")
        
        if entity_id not in self.entity_to_idx[entity_type]:
            raise ValueError(f"Unknown entity: {entity_id}")
        
        entity_idx = self.entity_to_idx[entity_type][entity_id]
        entity_node = (entity_type, entity_idx)
        
        neighbors = defaultdict(list)
        
        # Outgoing edges
        for _, target, data in self.nx_graph.out_edges(entity_node, data=True):
            target_type, target_idx = target
            target_id = self.idx_to_entity[target_type][target_idx]
            relation = data["relation"]
            confidence = data["confidence"]
            
            neighbors[relation].append((target_id, target_type, confidence))
        
        # Incoming edges
        for source, _, data in self.nx_graph.in_edges(entity_node, data=True):
            source_type, source_idx = source
            source_id = self.idx_to_entity[source_type][source_idx]
            relation = data["relation"]
            confidence = data["confidence"]
            
            # Add with inverse relation
            inverse_relation = f"inverse_{relation}"
            neighbors[inverse_relation].append((source_id, source_type, confidence))
        
        return dict(neighbors)
    
    def get_entity_subgraph(
        self,
        entity_ids: List[str],
        entity_types: List[str],
        max_hops: int = 2
    ) -> "KnowledgeGraph":
        """
        Get subgraph around entities.
        
        Args:
            entity_ids: Entity identifiers.
            entity_types: Entity types.
            max_hops: Maximum number of hops from the entities.
            
        Returns:
            Subgraph as a new KnowledgeGraph.
        """
        if len(entity_ids) != len(entity_types):
            raise ValueError("Number of entity IDs must match number of entity types")
        
        # Create new knowledge graph
        subgraph = KnowledgeGraph(self.config)
        
        # Find nodes within max_hops
        seed_nodes = []
        for entity_id, entity_type in zip(entity_ids, entity_types):
            if entity_id in self.entity_to_idx[entity_type]:
                entity_idx = self.entity_to_idx[entity_type][entity_id]
                seed_nodes.append((entity_type, entity_idx))
        
        # Use NetworkX to get subgraph
        nodes = set(seed_nodes)
        for _ in range(max_hops):
            new_nodes = set()
            for node in nodes:
                # Add neighbors
                for neighbor in self.nx_graph.successors(node):
                    new_nodes.add(neighbor)
                for neighbor in self.nx_graph.predecessors(node):
                    new_nodes.add(neighbor)
            nodes |= new_nodes
        
        # Add nodes and edges to subgraph
        for node in nodes:
            node_type, node_idx = node
            entity_id = self.idx_to_entity[node_type][node_idx]
            subgraph.add_entity(entity_id, node_type)
        
        for triplet in self.triplets:
            head, relation, tail = triplet
            if head in nodes and tail in nodes:
                head_type, head_idx = head
                tail_type, tail_idx = tail
                
                head_id = self.idx_to_entity[head_type][head_idx]
                tail_id = self.idx_to_entity[tail_type][tail_idx]
                
                subgraph.add_triplet(
                    head_id=head_id,
                    head_type=head_type,
                    relation=relation,
                    tail_id=tail_id,
                    tail_type=tail_type,
                    confidence=self.confidence_scores[triplet]
                )
        
        # Copy compound-target pairs that are within the subgraph
        for compound_id, target_id in self.compound_target_pairs:
            if (compound_id in subgraph.entity_to_idx.get("compound", {}) and 
                target_id in subgraph.entity_to_idx.get("target", {})):
                subgraph.add_compound_target_pair(compound_id, target_id)
        
        return subgraph
    
    def calculate_centrality(self, centrality_method: str = "pagerank") -> Dict[Tuple[str, int], float]:
        """
        Calculate centrality of nodes in the graph.
        
        Args:
            centrality_method: Centrality method ("degree", "closeness", "betweenness", "pagerank").
            
        Returns:
            Dictionary mapping (entity_type, entity_idx) tuples to centrality scores.
        """
        if centrality_method == "degree":
            centrality = nx.degree_centrality(self.nx_graph)
        elif centrality_method == "closeness":
            centrality = nx.closeness_centrality(self.nx_graph)
        elif centrality_method == "betweenness":
            centrality = nx.betweenness_centrality(self.nx_graph)
        elif centrality_method == "pagerank":
            centrality = nx.pagerank(self.nx_graph)
        else:
            raise ValueError(f"Unknown centrality method: {centrality_method}")
        
        return centrality
    
    def calculate_node_similarity(
        self,
        entity1_id: str,
        entity1_type: str,
        entity2_id: str,
        entity2_type: str,
        method: str = "jaccard"
    ) -> float:
        """
        Calculate similarity between two nodes.
        
        Args:
            entity1_id: First entity identifier.
            entity1_type: First entity type.
            entity2_id: Second entity identifier.
            entity2_type: Second entity type.
            method: Similarity method ("jaccard", "cosine").
            
        Returns:
            Similarity score.
        """
        if entity1_type not in self.entity_types or entity2_type not in self.entity_types:
            raise ValueError(f"Unknown entity type: {entity1_type} or {entity2_type}")
        
        if entity1_id not in self.entity_to_idx[entity1_type] or entity2_id not in self.entity_to_idx[entity2_type]:
            raise ValueError(f"Unknown entity: {entity1_id} or {entity2_id}")
        
        entity1_idx = self.entity_to_idx[entity1_type][entity1_id]
        entity2_idx = self.entity_to_idx[entity2_type][entity2_id]
        
        entity1_node = (entity1_type, entity1_idx)
        entity2_node = (entity2_type, entity2_idx)
        
        # Get neighbors
        entity1_neighbors = set(self.nx_graph.successors(entity1_node)) | set(self.nx_graph.predecessors(entity1_node))
        entity2_neighbors = set(self.nx_graph.successors(entity2_node)) | set(self.nx_graph.predecessors(entity2_node))
        
        if method == "jaccard":
            # Jaccard similarity
            if not entity1_neighbors and not entity2_neighbors:
                return 0.0
            
            intersection = len(entity1_neighbors & entity2_neighbors)
            union = len(entity1_neighbors | entity2_neighbors)
            
            return intersection / union
        elif method == "cosine":
            # Cosine similarity using adjacency vectors
            all_nodes = sorted(list(set(self.nx_graph.nodes())))
            
            # Create adjacency vectors
            entity1_vector = [1 if node in entity1_neighbors else 0 for node in all_nodes]
            entity2_vector = [1 if node in entity2_neighbors else 0 for node in all_nodes]
            
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(entity1_vector, entity2_vector))
            magnitude1 = sum(a * a for a in entity1_vector) ** 0.5
            magnitude2 = sum(b * b for b in entity2_vector) ** 0.5
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def save(self, file_path: str) -> None:
        """
        Save knowledge graph to file.
        
        Args:
            file_path: Path to the output file.
        """
        data = {
            "entity_types": self.entity_types,
            "relation_types": self.relation_types,
            "entity_to_idx": self.entity_to_idx,
            "idx_to_entity": {k: {str(k2): v2 for k2, v2 in v.items()} for k, v in self.idx_to_entity.items()},
            "entity_counts": self.entity_counts,
            "triplets": [
                [list(h), r, list(t)] for h, r, t in self.triplets
            ],
            "compound_target_pairs": getattr(self, 'compound_target_pairs', [])
        }
        
        # Convert confidence scores to a serializable format
        confidence_scores_serialized = {}
        for triplet, score in self.confidence_scores.items():
            head, relation, tail = triplet
            key = f"{head}|{relation}|{tail}"
            confidence_scores_serialized[key] = score
        
        data["confidence_scores"] = confidence_scores_serialized
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, file_path: str, config: DataConfig) -> "KnowledgeGraph":
        """
        Load knowledge graph from file.
        
        Args:
            file_path: Path to the input file.
            config: Data configuration.
            
        Returns:
            Loaded knowledge graph.
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        
        kg = cls(config)
        kg.entity_types = data["entity_types"]
        kg.relation_types = data["relation_types"]
        kg.entity_to_idx = data["entity_to_idx"]
        kg.idx_to_entity = {k: {int(k2): v2 for k2, v2 in v.items()} for k, v in data["idx_to_entity"].items()}
        kg.entity_counts = data["entity_counts"]
        
        # Convert triplets
        kg.triplets = []
        for h, r, t in data["triplets"]:
            head = (h[0], h[1])
            relation = r
            tail = (t[0], t[1])
            kg.triplets.append((head, relation, tail))
        
        # Initialize confidence scores with default value of 1.0
        kg.confidence_scores = {triplet: 1.0 for triplet in kg.triplets}
        
        # Load compound-target pairs if available
        if "compound_target_pairs" in data:
            kg.compound_target_pairs = data["compound_target_pairs"]
        else:
            kg.compound_target_pairs = []
        
        # Rebuild NetworkX graph
        kg.nx_graph = nx.MultiDiGraph()
        for entity_type in kg.entity_types:
            for idx, entity_id in kg.idx_to_entity[entity_type].items():
                kg.nx_graph.add_node(
                    (entity_type, idx),
                    entity_id=entity_id,
                    entity_type=entity_type
                )
        
        for triplet in kg.triplets:
            head, relation, tail = triplet
            kg.nx_graph.add_edge(
                head,
                tail,
                relation=relation,
                confidence=kg.confidence_scores[triplet]
            )
        
        return kg
