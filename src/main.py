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
from src.data.knowledge_graph import KnowledgeGraph

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
    
    # Create target ranker with node features
    ranker = TargetRanker(
        kg=kg,
        model=model,
        config=config.ranking,
        device=config.device,
        node_features=node_features  # Pass node features!
    )
    
    # Create evaluator with node features
    evaluator = Evaluator(
        kg=kg,
        model=model,
        ranker=ranker,
        device=config.device,
        node_features=node_features  # Pass node features!
    )
    
    # Evaluate model with node features
    test_metrics = evaluator.evaluate_model(
        test_data=[(train_pairs[i][0], train_pairs[i][1], train_labels[i]) for i in test_indices],
        node_features=node_features  # Pass node features!
    )
    
    print("Test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Optimize ranking weights
    if config.ranking.optimize_weights:
        try:
            best_weights = ranker.optimize_weights(
                validation_data=[(train_pairs[i][0], train_pairs[i][1], train_labels[i]) for i in val_indices],
                cv_folds=config.ranking.weight_cv_folds,
                grid_step=config.ranking.weight_grid_search_step
            )
            
            print("Optimized weights:")
            for weight_name, weight_value in best_weights.items():
                print(f"  {weight_name}: {weight_value:.4f}")
        except Exception as e:
            print(f"Warning: Could not optimize weights: {e}")
    
    # Save feature builder
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    feature_builder.save(os.path.join(config.checkpoint_dir, "feature_builder.pkl"))
    
    # Save knowledge graph
    kg.save(os.path.join(config.checkpoint_dir, "knowledge_graph.json"))
    
    # Save final model
    model.save(os.path.join(config.checkpoint_dir, "final_model.pth"))
    # Save complete model state
    model_output_dir = os.path.join(config.checkpoint_dir, "complete_model")
    save_complete_model_state(model, feature_builder, kg, config, model_output_dir)
    print("Training completed.")

def evaluate(model_path: str, data_path: str, config_path: str, output_dir: str = None) -> None:
    """Evaluate TCM target prioritization model."""
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
    
    # Create data loader for accessing data files
    data_loader = DataLoader(config.data)
    
    # Load compound structures, target sequences, and disease ontology
    print("Loading feature data...")
    smiles_mapping = data_loader.load_compound_structures()
    sequence_mapping = data_loader.load_target_sequences()
    ontology_data = data_loader.load_disease_ontology()
    tcm_metadata = data_loader.load_entity_metadata("tcm")
    
    # Create feature builder and build node features
    print("Building node features...")
    feature_builder_path = os.path.join(os.path.dirname(model_path), "feature_builder.pkl")
    
    if os.path.exists(feature_builder_path):
        # Load the saved feature builder if available
        feature_builder = FeatureBuilder.load(feature_builder_path)
    else:
        # Create a new feature builder if not available
        feature_builder = FeatureBuilder(config.features)
    
    # Build node features
    node_features = feature_builder.build_node_features(
        kg=kg,
        smiles_mapping=smiles_mapping,
        sequence_mapping=sequence_mapping,
        ontology_data=ontology_data,
        tcm_metadata=tcm_metadata
    )
    
    # Load model
    print("Loading model...")
    model = AttentionRGCN.load(model_path, device=config.device)
    
    # Create target ranker with node features
    ranker = TargetRanker(
        kg=kg,
        model=model,
        config=config.ranking,
        device=config.device,
        node_features=node_features  # Pass node features to ranker
    )
    
    # Create evaluator
    evaluator = Evaluator(
        kg=kg,
        model=model,
        ranker=ranker,
        device=config.device,
        node_features=node_features  # Pass node features to evaluator
    )
    
    # Load test data
    print("Loading test data...")
    test_data = pd.read_csv(data_path)
    
    # Evaluate model
    print("Evaluating model...")
    test_metrics = evaluator.evaluate_model(
        test_data=[(c, t, l) for c, t, l in zip(test_data["compound_id"], test_data["target_id"], test_data["label"])],
        node_features=node_features  # Pass node features to the evaluation method
    )
    
    print("Test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Evaluate ranking
    print("Evaluating ranking...")
    try:
        ranking_metrics = evaluator.evaluate_ranking(
            test_data=[(c, t, l) for c, t, l in zip(test_data["compound_id"], test_data["target_id"], test_data["label"])]
        )
        
        print("Ranking metrics:")
        for metric, value in ranking_metrics.items():
            print(f"  {metric}: {value:.4f}")
    except Exception as e:
        print(f"Warning: Could not evaluate ranking: {e}")
    
    # Create visualizations
    if output_dir is not None:
        print("Creating visualizations...")
        try:
            visualizer = Visualizer(kg=kg, model=model, ranker=ranker)
            
            # Plot precision-recall curve
            try:
                evaluator.plot_precision_recall_curve(
                    test_data=[(c, t, l) for c, t, l in zip(test_data["compound_id"], test_data["target_id"], test_data["label"])],
                    output_file=os.path.join(output_dir, "precision_recall_curve.png")
                )
            except Exception as e:
                print(f"Warning: Could not create precision-recall curve: {e}")
            
            # Plot target rank distribution
            try:
                evaluator.plot_target_rank_distribution(
                    test_data=[(c, t, l) for c, t, l in zip(test_data["compound_id"], test_data["target_id"], test_data["label"])],
                    output_file=os.path.join(output_dir, "target_rank_distribution.png")
                )
            except Exception as e:
                print(f"Warning: Could not create target rank distribution: {e}")
            
            # Plot feature importance
            try:
                evaluator.analyze_feature_importance(
                    test_data=[(c, t, l) for c, t, l in zip(test_data["compound_id"], test_data["target_id"], test_data["label"])],
                    output_file=os.path.join(output_dir, "feature_importance.png")
                )
            except Exception as e:
                print(f"Warning: Could not analyze feature importance: {e}")
            
            # Plot knowledge graph
            try:
                visualizer.plot_knowledge_graph(
                    output_file=os.path.join(output_dir, "knowledge_graph.png"),
                    max_nodes=100
                )
            except Exception as e:
                print(f"Warning: Could not visualize knowledge graph: {e}")
            
            # Plot embedding space
            try:
                visualizer.plot_embedding_space(
                    entity_type="target",
                    output_file=os.path.join(output_dir, "target_embedding_space.png"),
                    num_entities=100
                )
            except Exception as e:
                print(f"Warning: Could not visualize embedding space: {e}")
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
    
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
    try:
        ranked_targets = ranker.rank_targets(
            compound_id=compound_id,
            disease_id=disease_id,
            top_k=top_k
        )
        
        # Print ranked targets
        print(f"Top {len(ranked_targets)} targets for compound {compound_id}:")
        print(ranked_targets.to_string(index=False))
        
        # Plot ranked targets
        try:
            visualizer.plot_ranked_targets(
                compound_id=compound_id,
                disease_id=disease_id,
                top_k=top_k
            )
        except Exception as e:
            print(f"Warning: Could not visualize ranked targets: {e}")
        
        # Plot feature contributions
        if len(ranked_targets) > 0:
            try:
                top_target_ids = ranked_targets["target_id"].tolist()
                visualizer.plot_feature_contributions(
                    compound_id=compound_id,
                    target_ids=top_target_ids,
                    disease_id=disease_id
                )
            except Exception as e:
                print(f"Warning: Could not visualize feature contributions: {e}")
    except Exception as e:
        print(f"Error ranking targets: {e}")

def save_complete_model_state(model, feature_builder, kg, config, output_dir):
    """
    Save all components needed for future prediction.
    
    Args:
        model: Trained AttentionRGCN model.
        feature_builder: Feature builder with extracted features.
        kg: Knowledge graph.
        config: Configuration object.
        output_dir: Output directory for saved components.
    """
    import os
    import torch
    import pickle
    import json
    from rdkit import rdBase
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save model
    model.save(os.path.join(output_dir, "final_model.pth"))
    print(f"Model saved to {os.path.join(output_dir, 'final_model.pth')}")
    
    # 2. Save feature builder
    with open(os.path.join(output_dir, "feature_builder.pkl"), "wb") as f:
        pickle.dump(feature_builder, f)
    print(f"Feature builder saved to {os.path.join(output_dir, 'feature_builder.pkl')}")
    
    # 3. Save knowledge graph
    kg.save(os.path.join(output_dir, "knowledge_graph.json"))
    print(f"Knowledge graph saved to {os.path.join(output_dir, 'knowledge_graph.json')}")
    
    # 4. Save configuration
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Configuration saved to {os.path.join(output_dir, 'config.json')}")
    
    # 5. Save version information for reproducibility
    version_info = {
        "rdkit_version": rdBase.rdkitVersion,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__ if 'np' in globals() else "unknown",
        "pandas_version": pd.__version__ if 'pd' in globals() else "unknown"
    }
    
    with open(os.path.join(output_dir, "version_info.json"), "w") as f:
        json.dump(version_info, f, indent=2)
    print(f"Version information saved to {os.path.join(output_dir, 'version_info.json')}")
    
    print(f"Complete model state saved to {output_dir}")

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


