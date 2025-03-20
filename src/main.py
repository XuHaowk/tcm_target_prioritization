"""
Main module for TCM target prioritization system.
"""
import os
import argparse
import json
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Set
from src.config import Config
from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.features.feature_utils import FeatureBuilder
from src.models.attention_rgcn import AttentionRGCN
from src.models.model_utils import ModelTrainer
from src.ranking.target_ranker import TargetRanker
from src.utils.evaluation import Evaluator
from src.utils.visualization import Visualizer

def load_config(config_path: str) -> Config:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Configuration object.
    """
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    return Config.from_dict(config_dict)

def train(config_path: str) -> None:
    """
    Train TCM target prioritization model.
    
    Args:
        config_path: Path to the configuration file.
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed
    torch.manual_seed(config.random_seed)
    
    # Create data loader
    data_loader = DataLoader(config.data)
    
    # Load knowledge graph
    kg = data_loader.load_or_create_knowledge_graph()
    
    # Load compound structures
    smiles_mapping = data_loader.load_compound_structures()
    
    # Load target sequences
    sequence_mapping = data_loader.load_target_sequences()
    
    # Load disease ontology
    ontology_data = data_loader.load_disease_ontology()
    
    # Load entity metadata
    tcm_metadata = data_loader.load_entity_metadata("tcm")
    
    # Create feature builder
    feature_builder = FeatureBuilder(config.features)
    
    # Build node features
    node_features = feature_builder.build_node_features(
        kg=kg,
        smiles_mapping=smiles_mapping,
        sequence_mapping=sequence_mapping,
        ontology_data=ontology_data,
        tcm_metadata=tcm_metadata
    )
    
    # Create data processor
    data_processor = DataProcessor(config.data)
    
    # Load training data
    train_pairs, train_labels = data_loader.load_training_data()
    
    # Create graph dataset
    dataset = data_processor.create_graph_dataset(
        kg=kg,
        pairs=train_pairs,
        labels=train_labels,
        node_features=node_features
    )
    
    # Split dataset
    train_indices, val_indices, test_indices = data_processor.create_train_val_test_split(
        pairs=train_pairs,
        labels=train_labels
    )
    
    train_dataset, val_dataset, test_dataset = data_processor.split_dataset(
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices
    )
    
    # Create model
    feature_dims = {}
    for entity_type in kg.entity_types:
        if entity_type in node_features:
            feature_dims[entity_type] = node_features[entity_type].size(1)
    
    model = AttentionRGCN(
        entity_types=kg.entity_types,
        relation_types=kg.relation_types,
        feature_dims=feature_dims,
        config=config.model
    )
    
    # Create model trainer
    trainer = ModelTrainer(
        model=model,
        config=config.model,
        device=config.device,
        log_dir=config.log_dir,
        checkpoint_dir=config.checkpoint_dir
    )
    
    # Train model
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=config.model.num_epochs,
        batch_size=config.model.batch_size,
        early_stopping_patience=config.model.early_stopping_patience
    )
    
    # Evaluate model
    evaluator = Evaluator(
        kg=kg,
        model=model,
        ranker=None,  # Will be created later
        device=config.device
    )
    
    test_metrics = evaluator.evaluate_model(
        test_data=[(train_pairs[i][0], train_pairs[i][1], train_labels[i]) for i in test_indices]
    )
    
    print("Test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Create target ranker
    ranker = TargetRanker(
        kg=kg,
        model=model,
        config=config.ranking,
        device=config.device
    )
    
    # Optimize ranking weights
    if config.ranking.optimize_weights:
        best_weights = ranker.optimize_weights(
            validation_data=[(train_pairs[i][0], train_pairs[i][1], train_labels[i]) for i in val_indices],
            cv_folds=config.ranking.weight_cv_folds,
            grid_step=config.ranking.weight_grid_search_step
        )
        
        print("Optimized weights:")
        for weight_name, weight_value in best_weights.items():
            print(f"  {weight_name}: {weight_value:.4f}")
    
    # Save feature builder
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    feature_builder.save(os.path.join(config.checkpoint_dir, "feature_builder.pkl"))
    
    # Save knowledge graph
    kg.save(os.path.join(config.checkpoint_dir, "knowledge_graph.json"))
    
    print("Training completed.")

def evaluate(model_path: str, data_path: str, config_path: str, output_dir: str = None) -> None:
    """
    Evaluate TCM target prioritization model.
    
    Args:
        model_path: Path to the model file.
        data_path: Path to the test data file.
        config_path: Path to the configuration file.
        output_dir: Output directory for visualizations.
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed
    torch.manual_seed(config.random_seed)
    
    # Create output directory
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load knowledge graph
    print("Loading knowledge graph...")
    knowledge_graph_path = os.path.join(os.path.dirname(model_path), "knowledge_graph.json")
    kg = KnowledgeGraph.load(knowledge_graph_path, config.data)
    
    # Load feature builder
    print("Loading feature builder...")
    feature_builder_path = os.path.join(os.path.dirname(model_path), "feature_builder.pkl")
    feature_builder = FeatureBuilder.load(feature_builder_path)
    
    # Load model
    print("Loading model...")
    model = AttentionRGCN.load(model_path, device=config.device)
    
    # Create target ranker
    ranker = TargetRanker(
        kg=kg,
        model=model,
        config=config.ranking,
        device=config.device
    )
    
    # Load test data
    print("Loading test data...")
    test_data = pd.read_csv(data_path)
    test_pairs = list(zip(test_data["compound_id"], test_data["target_id"]))
    test_labels = test_data["label"].values
    
    # Create evaluator
    evaluator = Evaluator(
        kg=kg,
        model=model,
        ranker=ranker,
        device=config.device
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_metrics = evaluator.evaluate_model(
        test_data=[(c, t, l) for c, t, l in zip(test_data["compound_id"], test_data["target_id"], test_data["label"])]
    )
    
    print("Test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Evaluate ranking
    print("Evaluating ranking...")
    ranking_metrics = evaluator.evaluate_ranking(
        test_data=[(c, t, l) for c, t, l in zip(test_data["compound_id"], test_data["target_id"], test_data["label"])]
    )
    
    print("Ranking metrics:")
    for metric, value in ranking_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Create visualizations
    if output_dir is not None:
        print("Creating visualizations...")
        visualizer = Visualizer(kg=kg, model=model, ranker=ranker)
        
        # Plot precision-recall curve
        evaluator.plot_precision_recall_curve(
            test_data=[(c, t, l) for c, t, l in zip(test_data["compound_id"], test_data["target_id"], test_data["label"])],
            output_file=os.path.join(output_dir, "precision_recall_curve.png")
        )
        
        # Plot target rank distribution
        evaluator.plot_target_rank_distribution(
            test_data=[(c, t, l) for c, t, l in zip(test_data["compound_id"], test_data["target_id"], test_data["label"])],
            output_file=os.path.join(output_dir, "target_rank_distribution.png")
        )
        
        # Plot feature importance
        evaluator.analyze_feature_importance(
            test_data=[(c, t, l) for c, t, l in zip(test_data["compound_id"], test_data["target_id"], test_data["label"])],
            output_file=os.path.join(output_dir, "feature_importance.png")
        )
        
        # Plot knowledge graph
        visualizer.plot_knowledge_graph(
            output_file=os.path.join(output_dir, "knowledge_graph.png"),
            max_nodes=100
        )
        
        # Plot embedding space
        visualizer.plot_embedding_space(
            entity_type="target",
            output_file=os.path.join(output_dir, "target_embedding_space.png"),
            num_entities=100
        )
    
    print("Evaluation completed.")

def predict(compound_id: str, model_path: str, config_path: str, disease_id: str = None, top_k: int = 20) -> None:
    """
    Predict targets for a compound.
    
    Args:
        compound_id: Compound identifier.
        model_path: Path to the model file.
        config_path: Path to the configuration file.
        disease_id: Disease identifier. If None, disease priority is averaged.
        top_k: Number of top targets to return.
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed
    torch.manual_seed(config.random_seed)
    
    # Load knowledge graph
    print("Loading knowledge graph...")
    knowledge_graph_path = os.path.join(os.path.dirname(model_path), "knowledge_graph.json")
    kg = KnowledgeGraph.load(knowledge_graph_path, config.data)
    
    # Load model
    print("Loading model...")
    model = AttentionRGCN.load(model_path, device=config.device)
    
    # Create target ranker
    ranker = TargetRanker(
        kg=kg,
        model=model,
        config=config.ranking,
        device=config.device
    )
    
    # Create visualizer
    visualizer = Visualizer(kg=kg, model=model, ranker=ranker)
    
    # Check if compound exists
    if compound_id not in kg.entity_to_idx["compound"]:
        print(f"Compound not found: {compound_id}")
        return
    
    # Rank targets
    print(f"Ranking targets for compound {compound_id}...")
    ranked_targets = ranker.rank_targets(
        compound_id=compound_id,
        disease_id=disease_id,
        top_k=top_k
    )
    
    # Print ranked targets
    print(f"Top {len(ranked_targets)} targets for compound {compound_id}:")
    print(ranked_targets.to_string(index=False))
    
    # Plot ranked targets
    visualizer.plot_ranked_targets(
        compound_id=compound_id,
        disease_id=disease_id,
        top_k=top_k
    )
    
    # Plot feature contributions
    if len(ranked_targets) > 0:
        top_target_ids = ranked_targets["target_id"].tolist()
        visualizer.plot_feature_contributions(
            compound_id=compound_id,
            target_ids=top_target_ids,
            disease_id=disease_id
        )

def main():
    """Main function."""
    # Create argument parser
    parser = argparse.ArgumentParser(description="TCM Target Prioritization System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--config", required=True, help="Path to configuration file")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    evaluate_parser.add_argument("--model", required=True, help="Path to model file")
    evaluate_parser.add_argument("--data", required=True, help="Path to test data file")
    evaluate_parser.add_argument("--config", required=True, help="Path to configuration file")
    evaluate_parser.add_argument("--output-dir", help="Output directory for visualizations")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict targets for a compound")
    predict_parser.add_argument("--compound", required=True, help="Compound identifier")
    predict_parser.add_argument("--model", required=True, help="Path to model file")
    predict_parser.add_argument("--config", required=True, help="Path to configuration file")
    predict_parser.add_argument("--disease", help="Disease identifier")
    predict_parser.add_argument("--top-k", type=int, default=20, help="Number of top targets to return")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "train":
        train(args.config)
    elif args.command == "evaluate":
        evaluate(args.model, args.data, args.config, args.output_dir)
    elif args.command == "predict":
        predict(args.compound, args.model, args.config, args.disease, args.top_k)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
