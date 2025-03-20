import os
import json
import pandas as pd
import numpy as np

# Create directory structure
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Sample data
# 1. Compound structures
compounds = {
    "C001": "CC(=O)Oc1ccccc1C(=O)O",
    "C002": "CCN(CC)C(=O)c1ccc(N)cc1",
    "C003": "CCc1cc(=O)oc2cc(OC)c(OC)c(OC)c12",
    "baicalin": "O=C(O)[C@H]1O[C@@H](Oc2cc3oc(-c4ccccc4)cc(=O)c3c(O)c2O)[C@H](O)[C@@H](O)[C@H]1O",
    "tetrandrine": "COc1cc2c(cc1OC)[C@H]1Cc3ccc(OC)c(OC)c3C[C@H]1N(C)CC2"
}

# 2. Target sequences (simplified for example)
targets = {
    "T001": "MTTQAPTFTQPLQSVVVLEGSTATFEAHISGFPVPEVSWFRDGQVISTSTLPG",
    "T002": "MESLVPGFNEKTHVQLSLPVLQVRDVLVRGFGDSVEEVECMVQDLLESNH",
    "T003": "MACPTGLGVWLALALALALALALTPGTGASFSFQVETQCLKGTFEGDLCNQ"
}

# 3. TCM-compound associations
tcm_compounds = [
    {"tcm_id": "SCU", "compound_id": "C001", "confidence": 0.9},
    {"tcm_id": "SCU", "compound_id": "C002", "confidence": 0.8},
    {"tcm_id": "HQB", "compound_id": "baicalin", "confidence": 1.0},
    {"tcm_id": "FZM", "compound_id": "tetrandrine", "confidence": 1.0},
    {"tcm_id": "HQB", "compound_id": "C003", "confidence": 0.75}
]

# 4. Drug-target interactions
drug_target = [
    {"compound_id": "C001", "target_id": "T001", "relation": "binds", "confidence": 0.9, "source": "Experiment"},
    {"compound_id": "C002", "target_id": "T002", "relation": "inhibits", "confidence": 0.8, "source": "Literature"},
    {"compound_id": "baicalin", "target_id": "T001", "relation": "activates", "confidence": 0.95, "source": "TCMSP"},
    {"compound_id": "tetrandrine", "target_id": "T003", "relation": "inhibits", "confidence": 0.9, "source": "PubChem"},
    {"compound_id": "baicalin", "target_id": "T002", "relation": "binds", "confidence": 0.7, "source": "DrugBank"}
]

# 5. Disease-target associations
disease_target = [
    {"disease_id": "D001", "target_id": "T001", "relation": "associated_with", "confidence": 0.9, "source": "OMIM"},
    {"disease_id": "D002", "target_id": "T002", "relation": "causal", "confidence": 0.7, "source": "DisGeNET"},
    {"disease_id": "D003", "target_id": "T003", "relation": "therapeutic", "confidence": 0.8, "source": "TTD"},
    {"disease_id": "D001", "target_id": "T002", "relation": "associated_with", "confidence": 0.6, "source": "DisGeNET"},
    {"disease_id": "D002", "target_id": "T003", "relation": "associated_with", "confidence": 0.5, "source": "OMIM"}
]

# 6. Disease ontology
disease_ontology = {
    "D001": {
        "name": "Hypertension",
        "categories": ["cardiovascular"],
        "broader": [
            {
                "id": "D100",
                "name": "Cardiovascular Diseases",
                "categories": ["cardiovascular"]
            }
        ],
        "narrower": [
            {
                "id": "D101",
                "name": "Pulmonary Hypertension",
                "categories": ["cardiovascular", "respiratory"]
            }
        ],
        "prevalence": 0.3,
        "chronicity": 0.9,
        "severity": 0.7,
        "treatability": 0.8,
        "age_of_onset": 50
    },
    "D002": {
        "name": "Rheumatoid Arthritis",
        "categories": ["inflammatory", "autoimmune", "musculoskeletal"],
        "broader": [
            {
                "id": "D200",
                "name": "Autoimmune Diseases",
                "categories": ["autoimmune"]
            }
        ],
        "narrower": [],
        "prevalence": 0.01,
        "chronicity": 0.9,
        "severity": 0.7,
        "treatability": 0.6,
        "age_of_onset": 40
    },
    "D003": {
        "name": "Asthma",
        "categories": ["respiratory", "inflammatory"],
        "broader": [
            {
                "id": "D300",
                "name": "Respiratory Diseases",
                "categories": ["respiratory"]
            }
        ],
        "narrower": [
            {
                "id": "D301",
                "name": "Allergic Asthma",
                "categories": ["respiratory", "inflammatory", "allergic"]
            }
        ],
        "prevalence": 0.08,
        "chronicity": 0.8,
        "severity": 0.6,
        "treatability": 0.7,
        "age_of_onset": 10
    }
}

# 7. Training and validation data
training_data = [
    {"compound_id": "C001", "target_id": "T001", "label": 1},
    {"compound_id": "C001", "target_id": "T002", "label": 0},
    {"compound_id": "C002", "target_id": "T001", "label": 0},
    {"compound_id": "C002", "target_id": "T002", "label": 1},
    {"compound_id": "baicalin", "target_id": "T001", "label": 1},
    {"compound_id": "baicalin", "target_id": "T003", "label": 0},
    {"compound_id": "tetrandrine", "target_id": "T002", "label": 0},
    {"compound_id": "tetrandrine", "target_id": "T003", "label": 1}
]

# Entity metadata
tcm_metadata = {
    "HQB": {
        "name": "Huangqin",
        "latin_name": "Scutellaria baicalensis",
        "origin_type": 1,
        "traditional_use": 2,
        "preparation": 1
    },
    "FZM": {
        "name": "Fangji",
        "latin_name": "Stephania tetrandra",
        "origin_type": 1,
        "traditional_use": 3,
        "preparation": 2
    },
    "SCU": {
        "name": "Chuanxiong",
        "latin_name": "Ligusticum chuanxiong",
        "origin_type": 1,
        "traditional_use": 1,
        "preparation": 3
    }
}

# Save files
# 1. Compound structures
pd.DataFrame({"compound_id": list(compounds.keys()), "smiles": list(compounds.values())}).to_csv("data/raw/compound_structures.csv", index=False)

# 2. Target sequences
pd.DataFrame({"target_id": list(targets.keys()), "sequence": list(targets.values())}).to_csv("data/raw/target_sequences.csv", index=False)

# 3. TCM-compound associations
pd.DataFrame(tcm_compounds).to_csv("data/raw/tcm_compounds.csv", index=False)

# 4. Drug-target interactions
pd.DataFrame(drug_target).to_csv("data/raw/bindingdb_drug_target.csv", index=False)

# 5. Disease-target associations
pd.DataFrame(disease_target).to_csv("data/raw/omim_disease_target.csv", index=False)

# 6. Disease ontology
with open("data/raw/disease_ontology.json", "w") as f:
    json.dump(disease_ontology, f, indent=2)

# 7. Training data
pd.DataFrame(training_data).to_csv("data/raw/training_data.csv", index=False)
# Use the same data for validation and testing for this example
pd.DataFrame(training_data).to_csv("data/raw/validation_data.csv", index=False)
pd.DataFrame(training_data).to_csv("data/raw/test_data.csv", index=False)

# Entity metadata
with open("data/raw/tcm_metadata.json", "w") as f:
    json.dump(tcm_metadata, f, indent=2)

print("Sample data files created successfully!")
