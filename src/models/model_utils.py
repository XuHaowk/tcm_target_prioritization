"""
Model utilities for TCM target prioritization system.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional, Union, Callable
from tqdm import tqdm
import numpy as np
from src.config import ModelConfig
from src.models.attention_rgcn import AttentionRGCN
from src.models.layers import CombinedLoss
from src.data.knowledge_graph import KnowledgeGraph
from src.data.data_processor import GraphDataset

class ModelTrainer:
    """Model trainer for TCM target prioritization."""
    
    def __init__(
        self,
        model: AttentionRGCN,
        config: ModelConfig,
        device: str = "cpu",
        log_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize model trainer.
        
        Args:
            model: AttentionRGCN model.
            config: Model configuration.
            device: Device to train on.
            log_dir: Directory for tensorboard logs.
            checkpoint_dir: Directory for checkpoints.
        """
        self.model = model
        self.config = config
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        # Move model to device
        self.model.to(self.device)
        
        # Create loss function
        self.loss_fn = CombinedLoss(
            focal_gamma=config.focal_loss_gamma,
            focal_alpha=config.focal_loss_alpha
        )
        
        # Create optimizer
        if config.optimizer == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
        
        # Create learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Create tensorboard writer
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
        
        # Create checkpoint directory
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(
        self,
        dataset: GraphDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        max_hops: int = 2
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataset: Training dataset.
            batch_size: Batch size.
            shuffle: Whether to shuffle the dataset.
            max_hops: Maximum number of hops for subgraphs.
            
        Returns:
            Dictionary of training metrics.
        """
        # Set model to training mode
        self.model.train()
        
        # Get training batches
        batches = dataset.to_train_batch(batch_size=batch_size, shuffle=shuffle, max_hops=max_hops)
        
        # Initialize metrics
        total_loss = 0.0
        total_focal_loss = 0.0
        total_rank_loss = 0.0
        total_samples = 0
        
        # Train on batches
        for batch in tqdm(batches, desc="Training", leave=False):
            # Process batch
            batch_loss, batch_metrics = self._process_batch(batch)
            
            # Update metrics
            batch_size = len(batch)
            total_loss += batch_loss.item() * batch_size
            total_focal_loss += batch_metrics["focal_loss"].item() * batch_size
            if "rank_loss" in batch_metrics:
                total_rank_loss += batch_metrics["rank_loss"].item() * batch_size
            total_samples += batch_size
        
        # Calculate average metrics
        avg_loss = total_loss / total_samples
        avg_focal_loss = total_focal_loss / total_samples
        avg_rank_loss = total_rank_loss / total_samples if total_rank_loss > 0 else 0.0
        
        # Return metrics
        return {
            "loss": avg_loss,
            "focal_loss": avg_focal_loss,
            "rank_loss": avg_rank_loss
        }
    
    def evaluate(
        self,
        dataset: GraphDataset,
        batch_size: int = 32,
        max_hops: int = 2
    ) -> Dict[str, float]:
        """
        Evaluate model.
        
        Args:
            dataset: Evaluation dataset.
            batch_size: Batch size.
            max_hops: Maximum number of hops for subgraphs.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Get evaluation batches
        batches = dataset.to_train_batch(batch_size=batch_size, shuffle=False, max_hops=max_hops)
        
        # Initialize metrics
        total_loss = 0.0
        total_focal_loss = 0.0
        total_rank_loss = 0.0
        total_samples = 0
        
        # Initialize confusion matrix
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        # Evaluate on batches
        with torch.no_grad():
            for batch in tqdm(batches, desc="Evaluating", leave=False):
                # Process batch
                batch_loss, batch_metrics, batch_outputs, batch_targets = self._process_batch(
                    batch, return_outputs=True
                )
                
                # Update metrics
                batch_size = len(batch)
                total_loss += batch_loss.item() * batch_size
                total_focal_loss += batch_metrics["focal_loss"].item() * batch_size
                if "rank_loss" in batch_metrics:
                    total_rank_loss += batch_metrics["rank_loss"].item() * batch_size
                total_samples += batch_size
                
                # Update confusion matrix
                predictions = (batch_outputs > 0.5).float()
                true_positives += ((predictions == 1) & (batch_targets == 1)).sum().item()
                false_positives += ((predictions == 1) & (batch_targets == 0)).sum().item()
                true_negatives += ((predictions == 0) & (batch_targets == 0)).sum().item()
                false_negatives += ((predictions == 0) & (batch_targets == 1)).sum().item()
        
        # Calculate average metrics
        avg_loss = total_loss / total_samples
        avg_focal_loss = total_focal_loss / total_samples
        avg_rank_loss = total_rank_loss / total_samples if total_rank_loss > 0 else 0.0
        
        # Calculate classification metrics
        accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Return metrics
        return {
            "loss": avg_loss,
            "focal_loss": avg_focal_loss,
            "rank_loss": avg_rank_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def _process_batch(
        self,
        batch: List[Tuple[KnowledgeGraph, Dict[str, torch.Tensor], float]],
        update_model: bool = True,
        return_outputs: bool = False
    ) -> Union[
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]
    ]:
        """
        Process batch.
        
        Args:
            batch: Batch of (subgraph, node_features, label) tuples.
            update_model: Whether to update model parameters.
            return_outputs: Whether to return outputs.
            
        Returns:
            Tuple of (loss, loss_components, outputs, targets) if return_outputs=True,
            otherwise tuple of (loss, loss_components).
        """
        # Reset optimizer
        if update_model:
            self.optimizer.zero_grad()
        
        # Extract batch data
        subgraphs = [item[0] for item in batch]
        node_features = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        
        # Convert labels to tensor
        targets = torch.tensor(labels, dtype=torch.float32, device=self.device)
        
        # Initialize outputs
        outputs = torch.zeros(len(batch), device=self.device)
        
        # Process each sample in batch
        for i, (subgraph, features) in enumerate(zip(subgraphs, node_features)):
            # Move features to device
            features_device = {k: v.to(self.device) for k, v in features.items()}
            
            # Convert subgraph to PyTorch Geometric format
            pyg_data = subgraph.to_torch_geometric(features_device)
            
            # Extract edge indices and attributes
            edge_index_dict = {}
            edge_attr_dict = {}
            
            for edge_type, edge_info in pyg_data.edge_items():
                edge_index_dict[edge_type] = edge_info.edge_index.to(self.device)
                edge_attr_dict[edge_type] = edge_info.edge_attr.to(self.device)
            
            # Forward pass
            node_embeddings = self.model(features_device, edge_index_dict, edge_attr_dict)
            
            # Find compound and target nodes in the subgraph
            compound_pair = batch[i][0].valid_pairs[0]
            compound_id, target_id = compound_pair
            
            # Convert to indices
            compound_idx = subgraph.entity_to_idx["compound"][compound_id]
            target_idx = subgraph.entity_to_idx["target"][target_id]
            
            # Get embeddings
            compound_embed = node_embeddings["compound"][compound_idx].unsqueeze(0)
            target_embed = node_embeddings["target"][target_idx].unsqueeze(0)
            
            # Calculate similarity score
            score = self.model.predict_link(compound_embed, target_embed).squeeze()
            outputs[i] = score
        
        # Calculate loss
        pos_mask = targets == 1
        neg_mask = targets == 0
        
        loss, loss_components = self.loss_fn(outputs, targets, pos_mask, neg_mask)
        
        # Update model if required
        if update_model:
            loss.backward()
            self.optimizer.step()
        
        # Return results
        if return_outputs:
            return loss, loss_components, outputs, targets
        else:
            return loss, loss_components
    
    def train(
        self,
        train_dataset: GraphDataset,
        val_dataset: GraphDataset,
        num_epochs: int = None,
        batch_size: int = None,
        max_hops: int = 2,
        early_stopping_patience: int = None,
        eval_every: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train model.
        
        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            num_epochs: Number of epochs. If None, use config value.
            batch_size: Batch size. If None, use config value.
            max_hops: Maximum number of hops for subgraphs.
            early_stopping_patience: Early stopping patience. If None, use config value.
            eval_every: Evaluate every N epochs.
            
        Returns:
            Dictionary of training history.
        """
        # Use config values if not provided
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        if batch_size is None:
            batch_size = self.config.batch_size
        if early_stopping_patience is None:
            early_stopping_patience = self.config.early_stopping_patience
        
        # Initialize training history
        history = {
            "train_loss": [],
            "train_focal_loss": [],
            "train_rank_loss": [],
            "val_loss": [],
            "val_focal_loss": [],
            "val_rank_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": []
        }
        
        # Initialize early stopping
        best_val_f1 = 0.0
        patience_counter = 0
        
        # Train for specified number of epochs
        for epoch in range(num_epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                max_hops=max_hops
            )
            
            # Update training history
            history["train_loss"].append(train_metrics["loss"])
            history["train_focal_loss"].append(train_metrics["focal_loss"])
            history["train_rank_loss"].append(train_metrics["rank_loss"])
            
            # Evaluate if required
            if (epoch + 1) % eval_every == 0:
                val_metrics = self.evaluate(
                    dataset=val_dataset,
                    batch_size=batch_size,
                    max_hops=max_hops
                )
                
                # Update validation history
                history["val_loss"].append(val_metrics["loss"])
                history["val_focal_loss"].append(val_metrics["focal_loss"])
                history["val_rank_loss"].append(val_metrics["rank_loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                history["val_precision"].append(val_metrics["precision"])
                history["val_recall"].append(val_metrics["recall"])
                history["val_f1"].append(val_metrics["f1"])
                
                # Update learning rate scheduler
                self.scheduler.step(val_metrics["f1"])
                
                # Check for early stopping
                if val_metrics["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["f1"]
                    patience_counter = 0
                    
                    # Save best model
                    if self.checkpoint_dir is not None:
                        self.save_checkpoint(os.path.join(self.checkpoint_dir, "best_model.pth"))
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
                
                # Log to tensorboard
                if self.writer is not None:
                    self._log_metrics(train_metrics, val_metrics, epoch + 1)
                
                # Print metrics
                print(f"Epoch {epoch + 1}/{num_epochs}, "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val F1: {val_metrics['f1']:.4f}")
            else:
                # Print training metrics only
                print(f"Epoch {epoch + 1}/{num_epochs}, "
                      f"Train Loss: {train_metrics['loss']:.4f}")
        
        # Save final model
        if self.checkpoint_dir is not None:
            self.save_checkpoint(os.path.join(self.checkpoint_dir, "final_model.pth"))
        
        return history
    
    def _log_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int
    ) -> None:
        """
        Log metrics to tensorboard.
        
        Args:
            train_metrics: Training metrics.
            val_metrics: Validation metrics.
            epoch: Current epoch.
        """
        # Log training metrics
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"train/{key}", value, epoch)
        
        # Log validation metrics
        for key, value in val_metrics.items():
            self.writer.add_scalar(f"val/{key}", value, epoch)
        
        # Log learning rate
        self.writer.add_scalar(
            "train/learning_rate",
            self.optimizer.param_groups[0]["lr"],
            epoch
        )
    
    def save_checkpoint(self, file_path: str) -> None:
        """
        Save checkpoint.
        
        Args:
            file_path: Path to the output file.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.config
            },
            file_path
        )
    
    def load_checkpoint(self, file_path: str) -> None:
        """
        Load checkpoint.
        
        Args:
            file_path: Path to the input file.
        """
        checkpoint = torch.load(file_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
