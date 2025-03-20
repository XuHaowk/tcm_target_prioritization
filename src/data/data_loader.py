"""
Data loading utilities for TCM target prioritization system.
"""
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from src.config import DataConfig
from src.data.knowledge_graph import KnowledgeGraph

class DataLoader:
    """Data loader for TCM target prioritization."""
    
    def __init__(self, config: DataConfig):
        """
        Initialize data loader.
        
        Args:
            config: Data configuration.
        """
        self.config = config
        self.data_dir = config.raw_data_dir
    
    def load_drug_target_data(self, sources: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load drug-target interaction data.
        
        Args:
            sources: List of data sources to load. If None, load all sources.
            
        Returns:
            DataFrame with drug-target interaction data.
        """
        if sources is None:
            sources = self.config.drug_target_sources
        
        dfs = []
        for source in sources:
            file_path = os.path.join(self.data_dir, f"{source.lower()}_drug_target.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Add source column if not present
                if "source" not in df.columns:
                    df["source"] = source
                dfs.append(df)
            else:
                print(f"Warning: Drug-target data file not found: {file_path}")
        
        if not dfs:
            raise FileNotFoundError("No drug-target data files found")
        
        # Concatenate DataFrames
        return pd.concat(dfs, ignore_index=True)
    
    def load_disease_target_data(self, sources: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load disease-target data.
        
        Args:
            sources: List of data sources to load. If None, load all sources.
            
        Returns:
            DataFrame with disease-target data.
        """
        if sources is None:
            sources = self.config.disease_target_sources
        
        dfs = []
        for source in sources:
            file_path = os.path.join(self.data_dir, f"{source.lower()}_disease_target.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Add source column if not present
                if "source" not in df.columns:
                    df["source"] = source
                dfs.append(df)
            else:
                print(f"Warning: Disease-target data file not found: {file_path}")
        
        if not dfs:
            raise FileNotFoundError("No disease-target data files found")
        
        # Concatenate DataFrames
        return pd.concat(dfs, ignore_index=True)
    
    def load_tcm_compound_data(self) -> pd.DataFrame:
        """
        Load TCM compound data.
        
        Returns:
            DataFrame with TCM compound data.
        """
        file_path = os.path.join(self.data_dir, "tcm_compounds.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"TCM compound data file not found: {file_path}")
        
        return pd.read_csv(file_path)
    
    def load_compound_structures(self) -> Dict[str, str]:
        """
        Load compound structures.
        
        Returns:
            Dictionary mapping compound IDs to SMILES strings.
        """
        file_path = os.path.join(self.data_dir, "compound_structures.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Compound structures file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        return dict(zip(df["compound_id"], df["smiles"]))
    
    def load_target_sequences(self) -> Dict[str, str]:
        """
        Load target protein sequences.
        
        Returns:
            Dictionary mapping target IDs to protein sequences.
        """
        file_path = os.path.join(self.data_dir, "target_sequences.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target sequences file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        return dict(zip(df["target_id"], df["sequence"]))
    
    def load_disease_ontology(self) -> Dict[str, Dict[str, str]]:
        """
        Load disease ontology data.
        
        Returns:
            Dictionary mapping disease IDs to ontology information.
        """
        file_path = os.path.join(self.data_dir, "disease_ontology.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Disease ontology file not found: {file_path}")
        
        with open(file_path, "r") as f:
            return json.load(f)
    
    def load_entity_metadata(self, entity_type: str) -> Dict[str, Dict[str, str]]:
        """
        Load entity metadata.
        
        Args:
            entity_type: Entity type ("tcm", "compound", "target", "disease").
            
        Returns:
            Dictionary mapping entity IDs to metadata.
        """
        file_path = os.path.join(self.data_dir, f"{entity_type}_metadata.json")
        if not os.path.exists(file_path):
            print(f"Warning: Entity metadata file not found: {file_path}")
            return {}
        
        with open(file_path, "r") as f:
            return json.load(f)
    
    def load_or_create_knowledge_graph(self) -> KnowledgeGraph:
        """
        Load or create knowledge graph.
        
        Returns:
            Knowledge graph.
        """
        kg_file_path = os.path.join(self.config.processed_data_dir, "knowledge_graph.json")
        if os.path.exists(kg_file_path):
            return KnowledgeGraph.load(kg_file_path, self.config)
        
        print("Knowledge graph file not found. Creating new knowledge graph...")
        
        # Create knowledge graph
        kg = KnowledgeGraph(self.config)
        
        # Load TCM compound data
        tcm_compounds = self.load_tcm_compound_data()
        for _, row in tcm_compounds.iterrows():
            tcm_id = row["tcm_id"]
            compound_id = row["compound_id"]
            
            # Add TCM-compound relationship
            kg.add_triplet(
                head_id=tcm_id,
                head_type="tcm",
                relation="contains",
                tail_id=compound_id,
                tail_type="compound",
                confidence=row.get("confidence", 1.0)
            )
        
        # Load drug-target data
        drug_target_data = self.load_drug_target_data()
        for _, row in drug_target_data.iterrows():
            compound_id = row["compound_id"]
            target_id = row["target_id"]
            relation = row.get("relation", "binds")
            
            # Add compound-target relationship
            kg.add_triplet(
                head_id=compound_id,
                head_type="compound",
                relation=relation,
                tail_id=target_id,
                tail_type="target",
                confidence=row.get("confidence", 1.0)
            )
        
        # Load disease-target data
        disease_target_data = self.load_disease_target_data()
        for _, row in disease_target_data.iterrows():
            target_id = row["target_id"]
            disease_id = row["disease_id"]
            relation = row.get("relation", "associated_with")
            
            # Add target-disease relationship
            kg.add_triplet(
                head_id=target_id,
                head_type="target",
                relation=relation,
                tail_id=disease_id,
                tail_type="disease",
                confidence=row.get("confidence", 1.0)
            )
        
        # Create unified identifiers
        kg.create_unified_identifiers()
        
        # Save knowledge graph
        os.makedirs(self.config.processed_data_dir, exist_ok=True)
        kg.save(kg_file_path)
        
        return kg
    
    def load_training_data(self) -> Tuple[List[Tuple[str, str]], List[float]]:
        """
        Load training data for target prioritization.
        
        Returns:
            Tuple of (compound_id, target_id) pairs and corresponding labels.
        """
        file_path = os.path.join(self.data_dir, "training_data.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        pairs = list(zip(df["compound_id"], df["target_id"]))
        labels = df["label"].values
        
        return pairs, labels
    
    def load_validation_data(self) -> Tuple[List[Tuple[str, str]], List[float]]:
        """
        Load validation data for target prioritization.
        
        Returns:
            Tuple of (compound_id, target_id) pairs and corresponding labels.
        """
        file_path = os.path.join(self.data_dir, "validation_data.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Validation data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        pairs = list(zip(df["compound_id"], df["target_id"]))
        labels = df["label"].values
        
        return pairs, labels
    
    def load_test_data(self) -> Tuple[List[Tuple[str, str]], List[float]]:
        """
        Load test data for target prioritization.
        
        Returns:
            Tuple of (compound_id, target_id) pairs and corresponding labels.
        """
        file_path = os.path.join(self.data_dir, "test_data.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Test data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        pairs = list(zip(df["compound_id"], df["target_id"]))
        labels = df["label"].values
        
        return pairs, labels
    
    def load_compound_pair(self, compound_id: str) -> Dict[str, str]:
        """
        Load specific compound data for inference.
        
        Args:
            compound_id: Compound identifier.
            
        Returns:
            Dictionary with compound data.
        """
        file_path = os.path.join(self.data_dir, "compounds.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Compounds file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        row = df[df["compound_id"] == compound_id]
        
        if row.empty:
            raise ValueError(f"Compound not found: {compound_id}")
        
        return row.iloc[0].to_dict()
    
    def save_results(self, results: pd.DataFrame, output_file: str) -> None:
        """
        Save results to file.
        
        Args:
            results: Results DataFrame.
            output_file: Output file path.
        """
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        results.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
