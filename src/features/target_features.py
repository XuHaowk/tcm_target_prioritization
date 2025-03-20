"""
Target feature extraction for TCM target prioritization system.
"""
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from collections import Counter
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import substitution_matrices
from src.config import FeatureConfig

class TargetFeatureExtractor:
    """Target feature extractor for TCM target prioritization."""
    
    def __init__(self, config: FeatureConfig):
        """
        Initialize target feature extractor.
        
        Args:
            config: Feature configuration.
        """
        self.config = config
        self.use_sequence_encoding = config.use_sequence_encoding
        self.use_blosum = config.use_blosum
        
        # Load BLOSUM matrix
        self.blosum62 = substitution_matrices.load("BLOSUM62")
        
        # Define amino acid alphabet
        self.amino_acids = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
        ]
        
        # Create amino acid to index mapping
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
    
    def extract_sequence_composition(self, sequence: str) -> np.ndarray:
        """
        Extract amino acid composition from protein sequence.
        
        Args:
            sequence: Protein sequence.
            
        Returns:
            Amino acid composition.
        """
        # Count amino acids
        aa_counter = Counter(sequence.upper())
        
        # Calculate composition
        total_count = float(len(sequence))
        composition = np.zeros(len(self.amino_acids), dtype=np.float32)
        
        for i, aa in enumerate(self.amino_acids):
            composition[i] = aa_counter.get(aa, 0) / total_count
        
        return composition
    
    def extract_physico_chemical_properties(self, sequence: str) -> np.ndarray:
        """
        Extract physico-chemical properties from protein sequence.
        
        Args:
            sequence: Protein sequence.
            
        Returns:
            Physico-chemical properties.
        """
        # Molecular weight
        molecular_weight = 0
        aa_weights = {
            'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15,
            'Q': 146.15, 'E': 147.13, 'G': 75.07, 'H': 155.16, 'I': 131.17,
            'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
            'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
        }
        
        for aa in sequence.upper():
            if aa in aa_weights:
                molecular_weight += aa_weights[aa]
        
        # Hydrophobicity (Kyte-Doolittle)
        hydrophobicity = 0
        hydrophobicity_values = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        for aa in sequence.upper():
            if aa in hydrophobicity_values:
                hydrophobicity += hydrophobicity_values[aa]
        
        hydrophobicity /= len(sequence)
        
        # Isoelectric point
        isoelectric_point = 0
        isoelectric_values = {
            'A': 6.0, 'R': 10.76, 'N': 5.41, 'D': 2.77, 'C': 5.07,
            'Q': 5.65, 'E': 3.22, 'G': 5.97, 'H': 7.59, 'I': 6.02,
            'L': 5.98, 'K': 9.74, 'M': 5.74, 'F': 5.48, 'P': 6.30,
            'S': 5.68, 'T': 5.60, 'W': 5.89, 'Y': 5.66, 'V': 5.96
        }
        
        for aa in sequence.upper():
            if aa in isoelectric_values:
                isoelectric_point += isoelectric_values[aa]
        
        isoelectric_point /= len(sequence)
        
        # Secondary structure propensities
        helix_propensity = 0
        sheet_propensity = 0
        turn_propensity = 0
        
        helix_values = {
            'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
            'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
            'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
            'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06
        }
        
        sheet_values = {
            'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
            'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
            'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
            'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70
        }
        
        turn_values = {
            'A': 0.66, 'R': 0.95, 'N': 1.56, 'D': 1.46, 'C': 1.19,
            'Q': 0.98, 'E': 0.74, 'G': 1.56, 'H': 0.95, 'I': 0.47,
            'L': 0.59, 'K': 1.01, 'M': 0.60, 'F': 0.60, 'P': 1.52,
            'S': 1.43, 'T': 0.96, 'W': 0.96, 'Y': 1.14, 'V': 0.50
        }
        
        for aa in sequence.upper():
            if aa in helix_values:
                helix_propensity += helix_values[aa]
            if aa in sheet_values:
                sheet_propensity += sheet_values[aa]
            if aa in turn_values:
                turn_propensity += turn_values[aa]
        
        helix_propensity /= len(sequence)
        sheet_propensity /= len(sequence)
        turn_propensity /= len(sequence)
        
        # Combine properties
        properties = np.array([
            molecular_weight,
            hydrophobicity,
            isoelectric_point,
            helix_propensity,
            sheet_propensity,
            turn_propensity
        ], dtype=np.float32)
        
        return properties
    
    def calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """
        Calculate sequence similarity using BLOSUM62.
        
        Args:
            seq1: First protein sequence.
            seq2: Second protein sequence.
            
        Returns:
            Sequence similarity score.
        """
        # Use a simple global alignment score
        score = 0
        min_len = min(len(seq1), len(seq2))
        
        for i in range(min_len):
            aa1 = seq1[i].upper()
            aa2 = seq2[i].upper()
            
            if aa1 in self.blosum62 and aa2 in self.blosum62:
                score += self.blosum62[(aa1, aa2)]
        
        # Normalize by sequence length
        return score / min_len
    
    def extract_blosum_encoding(self, sequence: str) -> np.ndarray:
        """
        Extract BLOSUM encoding from protein sequence.
        
        Args:
            sequence: Protein sequence.
            
        Returns:
            BLOSUM encoding.
        """
        # Use the first 20 amino acids as reference
        ref_sequences = ["".join([aa] * 10) for aa in self.amino_acids]
        
        # Calculate similarity to each reference sequence
        similarities = np.zeros(len(ref_sequences), dtype=np.float32)
        for i, ref_seq in enumerate(ref_sequences):
            similarities[i] = self.calculate_sequence_similarity(sequence, ref_seq)
        
        return similarities
    
    def extract_sequence_features(self, sequence: str) -> np.ndarray:
        """
        Extract features from protein sequence.
        
        Args:
            sequence: Protein sequence.
            
        Returns:
            Sequence features.
        """
        # Extract sequence composition
        composition = self.extract_sequence_composition(sequence)
        
        # Extract physico-chemical properties
        properties = self.extract_physico_chemical_properties(sequence)
        
        # Extract BLOSUM encoding if configured
        if self.use_blosum:
            blosum_encoding = self.extract_blosum_encoding(sequence)
            features = np.concatenate([composition, properties, blosum_encoding])
        else:
            features = np.concatenate([composition, properties])
        
        return features
    
    def extract_features(self, sequence: str) -> np.ndarray:
        """
        Extract target features from protein sequence.
        
        Args:
            sequence: Protein sequence.
            
        Returns:
            Target features.
        """
        # Extract sequence features if configured
        if self.use_sequence_encoding:
            features = self.extract_sequence_features(sequence)
        else:
            # Extract simplified features
            composition = self.extract_sequence_composition(sequence)
            properties = self.extract_physico_chemical_properties(sequence)
            features = np.concatenate([composition, properties])
        
        return features
    
    def extract_batch_features(self, sequences: List[str]) -> np.ndarray:
        """
        Extract features for a batch of targets.
        
        Args:
            sequences: List of protein sequences.
            
        Returns:
            Batch of target features.
        """
        features = [self.extract_features(seq) for seq in sequences]
        return np.array(features, dtype=np.float32)
    
    def extract_features_from_mapping(self, sequence_mapping: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Extract features from target ID to sequence mapping.
        
        Args:
            sequence_mapping: Dictionary mapping target IDs to protein sequences.
            
        Returns:
            Dictionary mapping target IDs to features.
        """
        features = {}
        for target_id, sequence in sequence_mapping.items():
            features[target_id] = self.extract_features(sequence)
        
        return features
