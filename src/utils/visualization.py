"""
Visualization utilities for TCM target prioritization system.
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Set
from src.data.knowledge_graph import KnowledgeGraph
from src.models.attention_rgcn import AttentionRGCN
from src.ranking.target_ranker import TargetRanker

class Visualizer:
    """Visualizer for TCM target prioritization."""
    
    def __init__(
        self,
        kg: KnowledgeGraph,
        model: Optional[AttentionRGCN] = None,
        ranker: Optional[TargetRanker] = None
    ):
        """
        Initialize visualizer.
        
        Args:
            kg: Knowledge graph.
            model: AttentionRGCN model.
            ranker: Target ranker.
        """
        self.kg = kg
        self.model = model
        self.ranker = ranker
    
    def plot_knowledge_graph(
        self,
        output_file: Optional[str] = None,
        max_nodes: int = 100,
        include_edge_labels: bool = False
    ) -> None:
        """
        Plot knowledge graph.
        
        Args:
            output_file: Output file path. If None, display the plot.
            max_nodes: Maximum number of nodes to include.
            include_edge_labels: Whether to include edge labels.
        """
        # Extract subgraph if necessary
        if self.kg.nx_graph.number_of_nodes() > max_nodes:
            # Select random seed node
            seed_node = np.random.choice(list(self.kg.nx_graph.nodes()))
            
            # Extract k-hop subgraph
            subgraph_nodes = {seed_node}
            current_nodes = {seed_node}
            
            while len(subgraph_nodes) < max_nodes:
                new_nodes = set()
                for node in current_nodes:
                    neighbors = set(self.kg.nx_graph.successors(node)) | set(self.kg.nx_graph.predecessors(node))
                    new_nodes.update(neighbors - subgraph_nodes)
                    if len(subgraph_nodes | new_nodes) >= max_nodes:
                        new_nodes = list(new_nodes)[:max_nodes - len(subgraph_nodes)]
                        break
                
                subgraph_nodes.update(new_nodes)
                current_nodes = new_nodes
                
                if not current_nodes:
                    break
            
            # Extract subgraph
            subgraph = self.kg.nx_graph.subgraph(subgraph_nodes)
        else:
            subgraph = self.kg.nx_graph
        
        # Create figure
        plt.figure(figsize=(20, 16))
        
        # Define node colors by type
        node_colors = []
        node_sizes = []
        
        color_map = {
            "tcm": "#3498db",        # Blue
            "compound": "#2ecc71",   # Green
            "target": "#e74c3c",     # Red
            "disease": "#f39c12"     # Orange
        }
        
        size_map = {
            "tcm": 300,
            "compound": 200,
            "target": 250,
            "disease": 300
        }
        
        for node in subgraph.nodes():
            node_type = node[0]
            node_colors.append(color_map.get(node_type, "#95a5a6"))  # Gray for unknown types
            node_sizes.append(size_map.get(node_type, 100))
        
        # Define edge colors by type
        edge_colors = []
        
        color_map = {
            "contains": "#3498db",        # Blue
            "binds": "#2ecc71",           # Green
            "inhibits": "#e74c3c",        # Red
            "activates": "#f39c12",       # Orange
            "associated_with": "#9b59b6", # Purple
            "linked_with": "#34495e"      # Dark gray
        }
        
        for u, v, data in subgraph.edges(data=True):
            edge_colors.append(color_map.get(data["relation"], "#95a5a6"))  # Gray for unknown types
        
        # Create layout
        layout = nx.spring_layout(subgraph, k=0.3, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            subgraph,
            pos=layout,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            subgraph,
            pos=layout,
            edge_color=edge_colors,
            width=1.0,
            alpha=0.5,
            arrows=True,
            arrowsize=10,
            arrowstyle="->"
        )
        
        # Draw node labels
        node_labels = {}
        for node in subgraph.nodes():
            node_type, node_idx = node
            entity_id = self.kg.idx_to_entity[node_type][node_idx]
            node_labels[node] = f"{entity_id[:10]}..." if len(entity_id) > 10 else entity_id
        
        nx.draw_networkx_labels(
            subgraph,
            pos=layout,
            labels=node_labels,
            font_size=8,
            font_family="sans-serif"
        )
        
        # Draw edge labels if requested
        if include_edge_labels:
            edge_labels = {}
            for u, v, data in subgraph.edges(data=True):
                edge_labels[(u, v)] = data["relation"]
            
            nx.draw_networkx_edge_labels(
                subgraph,
                pos=layout,
                edge_labels=edge_labels,
                font_size=6,
                font_family="sans-serif"
            )
        
        # Create legend
        legend_elements = []
        
        # Node legend
        for node_type, color in color_map.items():
            legend_elements.append(
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                          markersize=10, label=f"{node_type}")
            )
        
        # Edge legend
        for relation, color in color_map.items():
            legend_elements.append(
                plt.Line2D([0], [0], color=color, lw=2, label=f"{relation}")
            )
        
        plt.legend(handles=legend_elements, loc="upper right")
        
        plt.axis("off")
        plt.title("Knowledge Graph Visualization")
        
        # Save or display the plot
        if output_file is not None:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()
    
    def plot_ranked_targets(
        self,
        compound_id: str,
        disease_id: Optional[str] = None,
        top_k: int = 20,
        output_file: Optional[str] = None
    ) -> None:
        """
        Plot ranked targets for a compound.
        
        Args:
            compound_id: Compound identifier.
            disease_id: Disease identifier. If None, disease priority is averaged.
            top_k: Number of top targets to include.
            output_file: Output file path. If None, display the plot.
        """
        if self.ranker is None:
            raise ValueError("Target ranker is required for this visualization")
        
        # Rank targets
        ranked_targets = self.ranker.rank_targets(
            compound_id=compound_id,
            disease_id=disease_id,
            top_k=top_k
        )
        
        # Check if any targets were found
        if ranked_targets.empty:
            print(f"No targets found for compound {compound_id}")
            return
        
        # Extract data
        target_ids = ranked_targets["target_id"].tolist()
        scores = ranked_targets["weighted_score"].tolist()
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart
        plt.barh(range(len(target_ids)), scores, color="#3498db", alpha=0.8)
        
        # Set y-ticks
        plt.yticks(range(len(target_ids)), target_ids)
        
        # Add labels and title
        plt.xlabel("Score")
        plt.ylabel("Target")
        plt.title(f"Top {top_k} Targets for Compound {compound_id}")
        
        # Add grid
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display the plot
        if output_file is not None:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()
    
    def plot_feature_contributions(
        self,
        compound_id: str,
        target_ids: List[str],
        disease_id: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> None:
        """
        Plot feature contributions for compound-target pairs.
        
        Args:
            compound_id: Compound identifier.
            target_ids: List of target identifiers.
            disease_id: Disease identifier. If None, disease priority is averaged.
            output_file: Output file path. If None, display the plot.
        """
        if self.ranker is None:
            raise ValueError("Target ranker is required for this visualization")
        
        # Convert knowledge graph to PyTorch Geometric format
        pyg_data = self.kg.to_torch_geometric()
        
        # Extract node features
        node_features = {}
        for entity_type in self.kg.entity_types:
            if entity_type in pyg_data:
                node_features[entity_type] = pyg_data[entity_type].x
        
        # Extract edge indices and attributes
        edge_index_dict = {}
        edge_attr_dict = {}
        
        for edge_type, edge_info in pyg_data.edge_items():
            edge_index_dict[edge_type] = edge_info.edge_index
            edge_attr_dict[edge_type] = edge_info.edge_attr
        
        # Compute node embeddings
        with torch.no_grad():
            node_embeddings = self.model(node_features, edge_index_dict, edge_attr_dict)
        
        # Calculate feature contributions for each target
        feature_data = []
        
        for target_id in target_ids:
            if target_id in self.kg.entity_to_idx["target"]:
                # Calculate similarity score
                similarity = self.ranker.predict_score(compound_id, target_id, node_embeddings)
                
                # Calculate disease priority
                disease_priority = self.ranker.calculate_disease_priority(target_id, disease_id)
                
                # Calculate centrality
                centrality = self.ranker.calculate_centrality(target_id)
                
                # Calculate weighted contributions
                similarity_contrib = self.ranker.similarity_weight * similarity
                disease_contrib = self.ranker.disease_weight * disease_priority
                centrality_contrib = self.ranker.centrality_weight * centrality
                
                # Calculate total score
                total_score = similarity_contrib + disease_contrib + centrality_contrib
                
                # Add to feature data
                feature_data.append({
                    "target_id": target_id,
                    "similarity": similarity_contrib,
                    "disease_priority": disease_contrib,
                    "centrality": centrality_contrib,
                    "total_score": total_score
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(feature_data)
        
        # Sort by total score
        df = df.sort_values("total_score", ascending=False)
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Create stacked bar chart
        ind = range(len(df))
        
        p1 = plt.bar(ind, df["similarity"], color="#3498db", alpha=0.8)
        p2 = plt.bar(ind, df["disease_priority"], bottom=df["similarity"], color="#2ecc71", alpha=0.8)
        p3 = plt.bar(ind, df["centrality"], bottom=df["similarity"] + df["disease_priority"], color="#e74c3c", alpha=0.8)
        
        # Set x-ticks
        plt.xticks(ind, df["target_id"], rotation=45, ha="right")
        
        # Add labels and title
        plt.xlabel("Target")
        plt.ylabel("Score Contribution")
        plt.title(f"Feature Contributions for Compound {compound_id}")
        
        # Add legend
        plt.legend((p1[0], p2[0], p3[0]), ("Similarity", "Disease Priority", "Centrality"))
        
        # Add grid
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display the plot
        if output_file is not None:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()
    
    def plot_embedding_space(
        self,
        entity_type: str,
        output_file: Optional[str] = None,
        num_entities: int = 100,
        use_tsne: bool = True
    ) -> None:
        """
        Plot embedding space for entities.
        
        Args:
            entity_type: Entity type.
            output_file: Output file path. If None, display the plot.
            num_entities: Number of entities to include.
            use_tsne: Whether to use t-SNE instead of PCA.
        """
        if self.model is None:
            raise ValueError("RGCN model is required for this visualization")
        
        # Convert knowledge graph to PyTorch Geometric format
        pyg_data = self.kg.to_torch_geometric()
        
        # Extract node features
        node_features = {}
        for etype in self.kg.entity_types:
            if etype in pyg_data:
                node_features[etype] = pyg_data[etype].x
        
        # Extract edge indices and attributes
        edge_index_dict = {}
        edge_attr_dict = {}
        
        for edge_type, edge_info in pyg_data.edge_items():
            edge_index_dict[edge_type] = edge_info.edge_index
            edge_attr_dict[edge_type] = edge_info.edge_attr
        
        # Compute node embeddings
        with torch.no_grad():
            node_embeddings = self.model(node_features, edge_index_dict, edge_attr_dict)
        
        # Extract embeddings for the specified entity type
        if entity_type not in node_embeddings:
            raise ValueError(f"Entity type not found: {entity_type}")
        
        # Get entity IDs
        entity_ids = list(self.kg.entity_to_idx[entity_type].keys())
        
        # Sample entities if there are too many
        if len(entity_ids) > num_entities:
            entity_ids = np.random.choice(entity_ids, num_entities, replace=False)
        
        # Get entity indices
        entity_indices = [self.kg.entity_to_idx[entity_type][eid] for eid in entity_ids]
        
        # Extract embeddings
        embeddings = node_embeddings[entity_type][entity_indices].cpu().numpy()
        
        # Reduce dimensionality for visualization
        if use_tsne:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        else:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        
        # Apply dimensionality reduction
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot
        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c="#3498db",
            alpha=0.8,
            s=100
        )
        
        # Add labels
        for i, entity_id in enumerate(entity_ids):
            plt.annotate(
                entity_id[:10] + "..." if len(entity_id) > 10 else entity_id,
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=8,
                alpha=0.8
            )
        
        # Add labels and title
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.title(f"Embedding Space for {entity_type} Entities")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display the plot
        if output_file is not None:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()
