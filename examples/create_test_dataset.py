#!/usr/bin/env python3
"""
Script to load and log the Opik test dataset for comet-mcp-tests project.
This script loads the dataset from the JSON file and submits it to Opik for evaluation.
"""

import json
import os
from datetime import datetime
import opik


def load_opik_dataset(dataset_path="opik_dataset.json"):
    """Load the Opik dataset from JSON file"""
    try:
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        print(f"Loaded dataset from {dataset_path}")
        return dataset
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        print(
            "Please ensure opik_dataset_complete.json exists in the examples directory"
        )
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in dataset file: {e}")
        return None


def print_dataset_info(dataset):
    """Print information about the dataset"""
    print(f"\n=== Opik Dataset Information ===")
    print(f"Total questions: {len(dataset)}")
    print(f"Project name: comet-mcp-tests")
    print(
        f"Experiments covered: 6 (resnet50_baseline, resnet50_dropout, efficientnet_b0, vit_base, resnet50_augmented, efficientnet_b3)"
    )

    # Count question types
    question_types = {
        "experiment_discovery": 0,
        "performance_comparison": 0,
        "training_analysis": 0,
        "configuration_analysis": 0,
        "results_analysis": 0,
    }

    for item in dataset:
        question = item["question"].lower()
        if "available" in question:
            question_types["experiment_discovery"] += 1
        elif any(
            word in question for word in ["compare", "better", "outperform", "superior"]
        ):
            question_types["performance_comparison"] += 1
        elif any(
            word in question
            for word in ["training", "converge", "curve", "progression"]
        ):
            question_types["training_analysis"] += 1
        elif any(
            word in question
            for word in ["config", "setting", "parameter", "hyperparameter"]
        ):
            question_types["configuration_analysis"] += 1
        else:
            question_types["results_analysis"] += 1

    print(f"\nQuestion type distribution:")
    for qtype, count in question_types.items():
        print(f"  {qtype.replace('_', ' ').title()}: {count}")


def log_dataset_to_opik(dataset, dataset_name="comet-mcp-tests-dataset"):
    """Log the dataset to Opik for evaluation"""
    try:
        print(f"\n=== Logging Dataset to Opik ===")
        print(f"Dataset name: {dataset_name}")
        print(f"Total questions: {len(dataset)}")

        # Initialize Opik client
        client = opik.Opik(project_name="comet-mcp-tests")

        # Create or get existing dataset in Opik
        try:
            opik_dataset = client.create_dataset(
                name=dataset_name,
                description="ML experiment analysis dataset for comet-mcp-tests project with 48 questions covering experiment discovery, performance comparison, training analysis, configuration analysis, and results analysis.",
            )
            print(f"Created new dataset in Opik with ID: {opik_dataset.id}")
        except Exception as e:
            if "already exists" in str(e):
                print(
                    f"Dataset '{dataset_name}' already exists, using existing dataset..."
                )
                # Get existing dataset
                try:
                    opik_dataset = client.get_dataset(dataset_name)
                    print(f"Using existing dataset with ID: {opik_dataset.id}")
                except Exception as get_error:
                    print(f"Could not retrieve existing dataset: {get_error}")
                    # Try to create with a unique name
                    import time

                    unique_name = f"{dataset_name}-{int(time.time())}"
                    print(f"Creating new dataset with unique name: {unique_name}")
                    opik_dataset = client.create_dataset(
                        name=unique_name,
                        description="ML experiment analysis dataset for comet-mcp-tests project with 48 questions covering experiment discovery, performance comparison, training analysis, configuration analysis, and results analysis.",
                    )
                    print(f"Created new dataset with ID: {opik_dataset.id}")
            else:
                raise e

        # Prepare data for insertion
        dataset_items = []
        for i, item in enumerate(dataset, 1):
            print(f"Preparing question {i}/{len(dataset)}: {item['question'][:50]}...")

            dataset_items.append(
                {"question": item["question"], "answer": item["answer"]}
            )

        # Insert questions one by one to debug
        print(f"Inserting {len(dataset_items)} questions into Opik dataset...")
        for i, item in enumerate(dataset_items, 1):
            print(f"Inserting question {i}/{len(dataset_items)}...")
            try:
                opik_dataset.insert([item])
            except Exception as insert_error:
                print(f"Error inserting question {i}: {insert_error}")
                print(f"Item structure: {item}")
                raise insert_error

        print(
            f"\n✅ Successfully logged {len(dataset)} questions to Opik dataset: {dataset_name}"
        )
        print(f"Dataset ID: {opik_dataset.id}")
        return opik_dataset.id

    except Exception as e:
        print(f"❌ Error logging dataset to Opik: {e}")
        print("Make sure you have:")
        print("1. Installed the opik package: pip install opik")
        print("2. Set up Opik authentication (API key or credentials)")
        print("3. Valid Opik account and permissions")
        return None


def main():
    """Main function to load and log the dataset to Opik"""
    print("Loading Opik dataset for comet-mcp-tests...")

    # Load the dataset from JSON file
    dataset = load_opik_dataset()

    if dataset is None:
        print("Failed to load dataset. Exiting.")
        return

    # Print dataset information
    print_dataset_info(dataset)

    # Log dataset to Opik
    dataset_id = log_dataset_to_opik(dataset)

    if dataset_id:
        print(f"\n=== Dataset Logging Complete ===")
        print(f"Dataset file: opik_dataset.json")
        print(f"Opik dataset ID: {dataset_id}")
        print(f"Total questions logged: {len(dataset)}")
        print(f"All questions reference: comet-mcp-tests project")
        print(f"\nYou can now use this dataset in Opik for evaluation!")
    else:
        print(f"\n=== Dataset Loading Complete (Opik logging failed) ===")
        print(f"Dataset file: opik_dataset.json")
        print(f"Total questions: {len(dataset)}")
        print(f"All questions reference: comet-mcp-tests project")
        print(f"Note: Dataset was not logged to Opik due to errors above.")


if __name__ == "__main__":
    main()
