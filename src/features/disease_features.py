"""
Disease feature extraction for TCM target prioritization system.
"""
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from collections import Counter
import torch
from src.config import FeatureConfig

class DiseaseFeatureExtractor:
    """Disease feature extractor for TCM target prioritization."""
    
    def __init__(self, config: FeatureConfig):
        """
        Initialize disease feature extractor.
        
        Args:
            config: Feature configuration.
        """
        self.config = config
        self.use_ontology = config.use_ontology
        
        # Define default disease categories
        self.disease_categories = [
            "infectious",
            "cardiovascular",
            "respiratory",
            "digestive",
            "neurological",
            "psychiatric",
            "endocrine",
            "metabolic",
            "musculoskeletal",
            "dermatological",
            "hematological",
            "immunological",
            "cancer",
            "genetic",
            "developmental",
            "congenital",
            "inflammatory",
            "autoimmune",
            "allergic",
            "nutritional"
        ]
        
        # Create category to index mapping
        self.category_to_idx = {cat: i for i, cat in enumerate(self.disease_categories)}
    
    def extract_category_encoding(self, categories: List[str]) -> np.ndarray:
        """
        Extract one-hot encoding of disease categories.
        
        Args:
            categories: List of disease categories.
            
        Returns:
            One-hot encoding of disease categories.
        """
        encoding = np.zeros(len(self.disease_categories), dtype=np.float32)
        
        for category in categories:
            if category.lower() in self.category_to_idx:
                encoding[self.category_to_idx[category.lower()]] = 1.0
        
        return encoding
    
    def extract_ontology_features(self, ontology_entry: Dict[str, str]) -> np.ndarray:
        """
        Extract ontology features.
        
        Args:
            ontology_entry: Disease ontology entry.
            
        Returns:
            Ontology features.
        """
        # Extract categories
        categories = ontology_entry.get("categories", [])
        category_encoding = self.extract_category_encoding(categories)
        
        # Extract broader terms
        broader_terms = ontology_entry.get("broader", [])
        broader_encoding = np.zeros(len(self.disease_categories), dtype=np.float32)
        
        for term in broader_terms:
            if "categories" in term:
                term_categories = term["categories"]
                for category in term_categories:
                    if category.lower() in self.category_to_idx:
                        broader_encoding[self.category_to_idx[category.lower()]] = 1.0
        
        # Extract narrower terms
        narrower_terms = ontology_entry.get("narrower", [])
        narrower_encoding = np.zeros(len(self.disease_categories), dtype=np.float32)
        
        for term in narrower_terms:
            if "categories" in term:
                term_categories = term["categories"]
                for category in term_categories:
                    if category.lower() in self.category_to_idx:
                        narrower_encoding[self.category_to_idx[category.lower()]] = 1.0
        
        # Combine ontology features
        ontology_features = np.concatenate([
            category_encoding,
            broader_encoding,
            narrower_encoding
        ])
        
        return ontology_features
    
    def extract_disease_properties(self, ontology_entry: Dict[str, str]) -> np.ndarray:
        """
        Extract disease properties.
        
        Args:
            ontology_entry: Disease ontology entry.
            
        Returns:
            Disease properties.
        """
        # Extract properties
        properties = []
        
        # Prevalence (0-1)
        prevalence = float(ontology_entry.get("prevalence", 0))
        properties.append(prevalence)
        
        # Chronicity (0-1)
        chronicity = float(ontology_entry.get("chronicity", 0))
        properties.append(chronicity)
        
        # Severity (0-1)
        severity = float(ontology_entry.get("severity", 0))
        properties.append(severity)
        
        # Treatability (0-1)
        treatability = float(ontology_entry.get("treatability", 0))
        properties.append(treatability)
        
        # Age of onset (normalized to 0-1)
        age_of_onset = float(ontology_entry.get("age_of_onset", 0)) / 100
        properties.append(age_of_onset)
        
        return np.array(properties, dtype=np.float32)
    
    def extract_disease_associations(self, disease_id: str, kg_data: Dict[str, List[str]]) -> np.ndarray:
        """
        Extract disease associations from knowledge graph.
        
        Args:
            disease_id: Disease identifier.
            kg_data: Knowledge graph data.
            
        Returns:
            Disease association features.
        """
        # Initialize association counts
        target_count = 0
        compound_count = 0
        pathway_count = 0
        
        # Count associations
        if "targets" in kg_data and disease_id in kg_data["targets"]:
            target_count = len(kg_data["targets"][disease_id])
        
        if "compounds" in kg_data and disease_id in kg_data["compounds"]:
            compound_count = len(kg_data["compounds"][disease_id])
        
        if "pathways" in kg_data and disease_id in kg_data["pathways"]:
            pathway_count = len(kg_data["pathways"][disease_id])
        
        # Create association features
        associations = np.array([
            target_count,
            compound_count,
            pathway_count
        ], dtype=np.float32)
        
        # Normalize by maximum values
        max_values = np.array([100, 1000, 100], dtype=np.float32)
        normalized_associations = associations / max_values
        
        return normalized_associations
    
    def extract_features(
        self,
        disease_id: str,
        ontology_data: Dict[str, Dict[str, str]],
        kg_data: Optional[Dict[str, List[str]]] = None
    ) -> np.ndarray:
        """
        Extract disease features.
        
        Args:
            disease_id: Disease identifier.
            ontology_data: Disease ontology data.
            kg_data: Knowledge graph data.
            
        Returns:
            Disease features.
        """
        # Get ontology entry
        if disease_id not in ontology_data:
            # Create default ontology entry
            ontology_entry = {
                "categories": [],
                "broader": [],
                "narrower": [],
                "prevalence": 0,
                "chronicity": 0,
                "severity": 0,
                "treatability": 0,
                "age_of_onset": 0
            }
        else:
            ontology_entry = ontology_data[disease_id]
        
        # Extract ontology features if configured
        if self.use_ontology:
            ontology_features = self.extract_ontology_features(ontology_entry)
        else:
            # Use simplified category encoding
            categories = ontology_entry.get("categories", [])
            ontology_features = self.extract_category_encoding(categories)
        
        # Extract disease properties
        disease_properties = self.extract_disease_properties(ontology_entry)
        
        # Extract disease associations if knowledge graph data is provided
        if kg_data is not None:
            association_features = self.extract_disease_associations(disease_id, kg_data)
            features = np.concatenate([ontology_features, disease_properties, association_features])
        else:
            features = np.concatenate([ontology_features, disease_properties])
        
        return features
    
    def extract_batch_features(
        self,
        disease_ids: List[str],
        ontology_data: Dict[str, Dict[str, str]],
        kg_data: Optional[Dict[str, List[str]]] = None
    ) -> np.ndarray:
        """
        Extract features for a batch of diseases.
        
        Args:
            disease_ids: List of disease identifiers.
            ontology_data: Disease ontology data.
            kg_data: Knowledge graph data.
            
        Returns:
            Batch of disease features.
        """
        features = [
            self.extract_features(disease_id, ontology_data, kg_data)
            for disease_id in disease_ids
        ]
        return np.array(features, dtype=np.float32)
    
    def extract_features_from_mapping(
        self,
        ontology_data: Dict[str, Dict[str, str]],
        kg_data: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from disease ontology data.
        
        Args:
            ontology_data: Disease ontology data.
            kg_data: Knowledge graph data.
            
        Returns:
            Dictionary mapping disease IDs to features.
        """
        features = {}
        for disease_id in ontology_data:
            features[disease_id] = self.extract_features(disease_id, ontology_data, kg_data)
        
        return features
