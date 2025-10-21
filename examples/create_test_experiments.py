#!/usr/bin/env python3
"""
Script to create test experiments for the Opik dataset using Comet ML.
This script creates 6 different experiments with realistic metrics, parameters, and training data.
"""

import comet_ml
import numpy as np
import random
import time
from datetime import datetime, timedelta

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Project configuration
PROJECT_NAME = "comet-mcp-tests"

def create_experiment(experiment_name, model_type, hyperparams, performance_profile):
    """Create a single experiment with realistic data"""
    
    # Initialize experiment
    experiment = comet_ml.Experiment(
        project_name=PROJECT_NAME,
        auto_output_logging="simple"
    )
    
    # Set experiment name
    experiment.set_name(experiment_name)
    
    # Log hyperparameters
    experiment.log_parameters(hyperparams)
    
    # Log system info
    experiment.log_other("model_type", model_type)
    experiment.log_other("dataset", "CIFAR-10")
    experiment.log_other("optimizer", "Adam")
    
    # Generate training data
    num_epochs = hyperparams.get("epochs", 50)
    steps_per_epoch = 100
    
    # Generate realistic training curves based on performance profile
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            global_step = epoch * steps_per_epoch + step
            
            # Generate training metrics with realistic patterns
            if performance_profile["type"] == "good_convergence":
                # Smooth convergence with good final performance
                train_loss = performance_profile["final_train_loss"] * (1 + 0.5 * np.exp(-epoch/10))
                train_acc = performance_profile["final_train_acc"] * (1 - 0.3 * np.exp(-epoch/8))
                val_loss = performance_profile["final_val_loss"] * (1 + 0.3 * np.exp(-epoch/12))
                val_acc = performance_profile["final_val_acc"] * (1 - 0.2 * np.exp(-epoch/10))
                
            elif performance_profile["type"] == "overfitting":
                # Good training performance but validation plateaus/degrades
                train_loss = performance_profile["final_train_loss"] * (1 + 0.8 * np.exp(-epoch/6))
                train_acc = performance_profile["final_train_acc"] * (1 - 0.1 * np.exp(-epoch/5))
                val_loss = performance_profile["final_val_loss"] * (1 + 0.2 * np.exp(-epoch/8) + 0.1 * max(0, epoch-20))
                val_acc = performance_profile["final_val_acc"] * (1 - 0.1 * np.exp(-epoch/8) - 0.05 * max(0, epoch-25))
                
            elif performance_profile["type"] == "slow_convergence":
                # Slower convergence, takes longer to reach good performance
                train_loss = performance_profile["final_train_loss"] * (1 + 1.2 * np.exp(-epoch/20))
                train_acc = performance_profile["final_train_acc"] * (1 - 0.4 * np.exp(-epoch/15))
                val_loss = performance_profile["final_val_loss"] * (1 + 0.8 * np.exp(-epoch/18))
                val_acc = performance_profile["final_val_acc"] * (1 - 0.3 * np.exp(-epoch/12))
                
            elif performance_profile["type"] == "unstable":
                # More volatile training with some instability
                noise_factor = 0.05 * np.sin(epoch * 0.5) + 0.02 * np.random.normal()
                train_loss = performance_profile["final_train_loss"] * (1 + 0.6 * np.exp(-epoch/10) + noise_factor)
                train_acc = performance_profile["final_train_acc"] * (1 - 0.25 * np.exp(-epoch/8) - abs(noise_factor))
                val_loss = performance_profile["final_val_loss"] * (1 + 0.4 * np.exp(-epoch/12) + noise_factor)
                val_acc = performance_profile["final_val_acc"] * (1 - 0.2 * np.exp(-epoch/10) - abs(noise_factor))
                
            elif performance_profile["type"] == "poor_performance":
                # Lower overall performance
                train_loss = performance_profile["final_train_loss"] * (1 + 0.4 * np.exp(-epoch/8))
                train_acc = performance_profile["final_train_acc"] * (1 - 0.2 * np.exp(-epoch/6))
                val_loss = performance_profile["final_val_loss"] * (1 + 0.3 * np.exp(-epoch/10))
                val_acc = performance_profile["final_val_acc"] * (1 - 0.15 * np.exp(-epoch/8))
                
            else:  # excellent_performance
                # Very good performance with smooth convergence
                train_loss = performance_profile["final_train_loss"] * (1 + 0.3 * np.exp(-epoch/6))
                train_acc = performance_profile["final_train_acc"] * (1 - 0.15 * np.exp(-epoch/5))
                val_loss = performance_profile["final_val_loss"] * (1 + 0.2 * np.exp(-epoch/8))
                val_acc = performance_profile["final_val_acc"] * (1 - 0.1 * np.exp(-epoch/6))
            
            # Add some step-level noise
            step_noise = 0.01 * np.random.normal()
            train_loss += step_noise
            train_acc += abs(step_noise) * 0.1
            val_loss += step_noise
            val_acc -= abs(step_noise) * 0.1
            
            # Log metrics
            experiment.log_metric("train_loss", train_loss, step=global_step)
            experiment.log_metric("train_accuracy", train_acc, step=global_step)
            experiment.log_metric("val_loss", val_loss, step=global_step)
            experiment.log_metric("val_accuracy", val_acc, step=global_step)
            
            # Log epoch-level metrics
            if step == steps_per_epoch - 1:
                experiment.log_metric("epoch_train_loss", train_loss, epoch=epoch)
                experiment.log_metric("epoch_train_accuracy", train_acc, epoch=epoch)
                experiment.log_metric("epoch_val_loss", val_loss, epoch=epoch)
                experiment.log_metric("epoch_val_accuracy", val_acc, epoch=epoch)
    
    # Log final metrics
    experiment.log_metric("final_train_loss", train_loss)
    experiment.log_metric("final_train_accuracy", train_acc)
    experiment.log_metric("final_val_loss", val_loss)
    experiment.log_metric("final_val_accuracy", val_acc)
    
    # Log additional metrics
    experiment.log_metric("best_val_accuracy", max(val_accuracies) if val_accuracies else val_acc)
    experiment.log_metric("best_train_accuracy", max(train_accuracies) if train_accuracies else train_acc)
    
    # Log model size and complexity
    experiment.log_metric("model_parameters", hyperparams.get("model_parameters", 1000000))
    experiment.log_metric("training_time_minutes", hyperparams.get("training_time", 45))
    
    # Log some confusion matrix data (simplified)
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    for i, class_name in enumerate(classes):
        experiment.log_metric(f"class_{class_name}_accuracy", 
                            val_acc * (0.8 + 0.4 * np.random.random()))
    
    experiment.end()
    print(f"Created experiment: {experiment_name}")
    return experiment.get_key()

def main():
    """Create all test experiments"""
    
    print("Creating test experiments for Opik dataset...")
    
    # Define experiment configurations
    experiments = [
        {
            "name": "resnet50_baseline",
            "model_type": "ResNet-50",
            "hyperparams": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,
                "dropout": 0.0,
                "weight_decay": 0.0001,
                "model_parameters": 25000000,
                "training_time": 45
            },
            "performance": {
                "type": "good_convergence",
                "final_train_loss": 0.15,
                "final_train_acc": 0.95,
                "final_val_loss": 0.25,
                "final_val_acc": 0.88
            }
        },
        {
            "name": "resnet50_dropout",
            "model_type": "ResNet-50",
            "hyperparams": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,
                "dropout": 0.3,
                "weight_decay": 0.0001,
                "model_parameters": 25000000,
                "training_time": 45
            },
            "performance": {
                "type": "excellent_performance",
                "final_train_loss": 0.12,
                "final_train_acc": 0.96,
                "final_val_loss": 0.20,
                "final_val_acc": 0.92
            }
        },
        {
            "name": "efficientnet_b0",
            "model_type": "EfficientNet-B0",
            "hyperparams": {
                "learning_rate": 0.0008,
                "batch_size": 64,
                "epochs": 50,
                "dropout": 0.2,
                "weight_decay": 0.0001,
                "model_parameters": 5000000,
                "training_time": 30
            },
            "performance": {
                "type": "good_convergence",
                "final_train_loss": 0.18,
                "final_train_acc": 0.94,
                "final_val_loss": 0.28,
                "final_val_acc": 0.85
            }
        },
        {
            "name": "vit_base",
            "model_type": "Vision Transformer",
            "hyperparams": {
                "learning_rate": 0.0005,
                "batch_size": 16,
                "epochs": 50,
                "dropout": 0.1,
                "weight_decay": 0.01,
                "model_parameters": 86000000,
                "training_time": 120
            },
            "performance": {
                "type": "overfitting",
                "final_train_loss": 0.08,
                "final_train_acc": 0.98,
                "final_val_loss": 0.35,
                "final_val_acc": 0.82
            }
        },
        {
            "name": "resnet50_augmented",
            "model_type": "ResNet-50",
            "hyperparams": {
                "learning_rate": 0.0008,
                "batch_size": 32,
                "epochs": 50,
                "dropout": 0.2,
                "weight_decay": 0.0001,
                "data_augmentation": True,
                "model_parameters": 25000000,
                "training_time": 50
            },
            "performance": {
                "type": "excellent_performance",
                "final_train_loss": 0.10,
                "final_train_acc": 0.97,
                "final_val_loss": 0.18,
                "final_val_acc": 0.94
            }
        },
        {
            "name": "efficientnet_b3",
            "model_type": "EfficientNet-B3",
            "hyperparams": {
                "learning_rate": 0.0005,
                "batch_size": 32,
                "epochs": 50,
                "dropout": 0.3,
                "weight_decay": 0.0001,
                "model_parameters": 12000000,
                "training_time": 60
            },
            "performance": {
                "type": "slow_convergence",
                "final_train_loss": 0.20,
                "final_train_acc": 0.93,
                "final_val_loss": 0.30,
                "final_val_acc": 0.87
            }
        }
    ]
    
    # Create all experiments
    experiment_keys = []
    for exp_config in experiments:
        try:
            key = create_experiment(
                exp_config["name"],
                exp_config["model_type"],
                exp_config["hyperparams"],
                exp_config["performance"]
            )
            experiment_keys.append(key)
            time.sleep(1)  # Small delay between experiments
        except Exception as e:
            print(f"Error creating experiment {exp_config['name']}: {e}")
    
    print(f"\nCreated {len(experiment_keys)} experiments:")
    for i, key in enumerate(experiment_keys):
        print(f"  {i+1}. {experiments[i]['name']}: {key}")
    
    print(f"\nAll experiments created in project: {PROJECT_NAME}")
    print("You can now test the Opik dataset questions against these experiments!")

if __name__ == "__main__":
    main()
