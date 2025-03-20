"""
Compound feature extraction for TCM target prioritization system.
"""
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors

from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors
# Access functions like Descriptors.NumHBD instead of Lipinski.NumHBD


from src.config import FeatureConfig

class CompoundFeatureExtractor:
    """Compound feature extractor for TCM target prioritization."""
    
    def __init__(self, config: FeatureConfig):
        """
        Initialize compound feature extractor.
        
        Args:
            config: Feature configuration.
        """
        self.config = config
        self.morgan_radius = config.morgan_radius
        self.morgan_nbits = config.morgan_nbits
        self.use_morgan_fingerprint = config.use_morgan_fingerprint
    
    def extract_morgan_fingerprint(self, smiles: str) -> np.ndarray:
        """
        Extract Morgan fingerprint from SMILES.
    
        Args:
            smiles: SMILES string.
        
        Returns:
            Morgan fingerprint.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(self.morgan_nbits, dtype=np.float32)
        
            # Replace this line:
            # fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.morgan_radius, nBits=self.morgan_nbits)
        
            # With this:
            from rdkit.Chem import rdMolDescriptors
            fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, 
                radius=self.morgan_radius, 
                nBits=self.morgan_nbits
            )
        
            return np.array(fingerprint, dtype=np.float32)
        except Exception as e:
            print(f"Error extracting Morgan fingerprint: {e}")
            return np.zeros(self.morgan_nbits, dtype=np.float32)
    
    def extract_descriptors(self, smiles: str) -> np.ndarray:
        """
        Extract molecular descriptors from SMILES.
    
        Args:
            smiles: SMILES string.
        
        Returns:
            Molecular descriptors.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(12, dtype=np.float32)
        
            # Calculate descriptors
            descriptors = [
                Descriptors.MolWt(mol),                      # Molecular weight
                Descriptors.MolLogP(mol),                    # LogP
                Descriptors.NumHDonors(mol),                 # Number of H-bond donors
                Descriptors.NumHAcceptors(mol),              # Number of H-bond acceptors
                Descriptors.NumRotatableBonds(mol),          # Number of rotatable bonds
                Descriptors.TPSA(mol),                       # Topological polar surface area
                Descriptors.NumAromaticRings(mol),           # Number of aromatic rings
                Descriptors.NumHeteroatoms(mol),             # Number of heteroatoms
                Descriptors.NumAliphaticRings(mol),          # Number of aliphatic rings
                Descriptors.NumHDonors(mol),                 # Number of H-bond donors (use instead of Lipinski.NumHBD)
                Descriptors.NumHAcceptors(mol),              # Number of H-bond acceptors (use instead of Lipinski.NumHBA)
                mol.GetRingInfo().NumRings()                 # Number of rings using RingInfo
            ]
        
            return np.array(descriptors, dtype=np.float32)
        except Exception as e:
            print(f"Error extracting descriptors: {e}")
            return np.zeros(12, dtype=np.float32)
    
    def extract_features(self, smiles: str) -> np.ndarray:
        """
        Extract compound features from SMILES.
        
        Args:
            smiles: SMILES string.
            
        Returns:
            Compound features.
        """
        # Extract fingerprint if configured
        if self.use_morgan_fingerprint:
            fingerprint = self.extract_morgan_fingerprint(smiles)
        else:
            fingerprint = np.array([], dtype=np.float32)
        
        # Extract descriptors
        descriptors = self.extract_descriptors(smiles)
        
        # Combine features
        features = np.concatenate([fingerprint, descriptors])
        
        return features
    
    def extract_batch_features(self, smiles_list: List[str]) -> np.ndarray:
        """
        Extract features for a batch of compounds.
        
        Args:
            smiles_list: List of SMILES strings.
            
        Returns:
            Batch of compound features.
        """
        features = [self.extract_features(smiles) for smiles in smiles_list]
        return np.array(features, dtype=np.float32)
    
    def extract_features_from_mapping(self, smiles_mapping: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Extract features from compound ID to SMILES mapping.
        
        Args:
            smiles_mapping: Dictionary mapping compound IDs to SMILES strings.
            
        Returns:
            Dictionary mapping compound IDs to features.
        """
        features = {}
        for compound_id, smiles in smiles_mapping.items():
            features[compound_id] = self.extract_features(smiles)
        
        return features

class TCMCompoundFeatureExtractor(CompoundFeatureExtractor):
    """TCM compound feature extractor with additional TCM-specific features."""
    
    def __init__(self, config: FeatureConfig):
        """
        Initialize TCM compound feature extractor.
        
        Args:
            config: Feature configuration.
        """
        super().__init__(config)
        self.tcm_metadata = {}
    
    def load_tcm_metadata(self, tcm_metadata: Dict[str, Dict[str, str]]) -> None:
        """
        Load TCM metadata.
        
        Args:
            tcm_metadata: TCM metadata.
        """
        self.tcm_metadata = tcm_metadata
    
    def extract_tcm_descriptors(self, compound_id: str) -> np.ndarray:
        """
        Extract TCM-specific descriptors.
        
        Args:
            compound_id: Compound identifier.
            
        Returns:
            TCM-specific descriptors.
        """
        if compound_id not in self.tcm_metadata:
            return np.zeros(3, dtype=np.float32)
        
        metadata = self.tcm_metadata[compound_id]
        
        # Extract TCM-specific descriptors
        descriptors = [
            float(metadata.get("origin_type", 0)),  # Origin type (0: Unknown, 1: Plant, 2: Animal, 3: Mineral)
            float(metadata.get("traditional_use", 0)),  # Traditional use category
            float(metadata.get("preparation", 0))  # Preparation method
        ]
        
        return np.array(descriptors, dtype=np.float32)
    
    def extract_features(self, smiles: str, compound_id: Optional[str] = None) -> np.ndarray:
        """
        Extract TCM compound features.
        
        Args:
            smiles: SMILES string.
            compound_id: Compound identifier.
            
        Returns:
            TCM compound features.
        """
        # Extract base features
        base_features = super().extract_features(smiles)
        
        # Extract TCM-specific descriptors if compound ID is provided
        if compound_id is not None:
            tcm_descriptors = self.extract_tcm_descriptors(compound_id)
            features = np.concatenate([base_features, tcm_descriptors])
        else:
            features = base_features
        
        return features
    
    def extract_batch_features(
        self,
        smiles_list: List[str],
        compound_ids: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract features for a batch of TCM compounds.
        
        Args:
            smiles_list: List of SMILES strings.
            compound_ids: List of compound identifiers.
            
        Returns:
            Batch of TCM compound features.
        """
        if compound_ids is None:
            features = [self.extract_features(smiles) for smiles in smiles_list]
        else:
            features = [
                self.extract_features(smiles, compound_id)
                for smiles, compound_id in zip(smiles_list, compound_ids)
            ]
        
        return np.array(features, dtype=np.float32)
    
    def extract_features_from_mapping(
        self,
        smiles_mapping: Dict[str, str]
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from compound ID to SMILES mapping.
        
        Args:
            smiles_mapping: Dictionary mapping compound IDs to SMILES strings.
            
        Returns:
            Dictionary mapping compound IDs to features.
        """
        features = {}
        for compound_id, smiles in smiles_mapping.items():
            features[compound_id] = self.extract_features(smiles, compound_id)
        
        return features





