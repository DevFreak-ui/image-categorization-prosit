#!/usr/bin/env python3
"""
Model Evaluation Script
Evaluates trained models with comprehensive metrics including Top-1 accuracy, F1 score, Precision, and performance over multiple runs.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import sys
import json
from datetime import datetime
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import model architectures
sys.path.append('model_architecture')
from DNN import get_dnn_model
from CNN import get_cnn_model

class ModelEvaluator:
    def __init__(self, data_root='./dataset', device=None):
        self.data_root = data_root
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Data transforms (same as training)
        MEAN = (0.4914, 0.4822, 0.4465)
        STD = (0.2470, 0.2435, 0.2616)
        
        self.eval_tfms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        
        self._load_test_data()
    
    def _load_test_data(self):
        """Load test dataset - always uses 10% of test set for faster evaluation"""
        print(f"Loading test data from {self.data_root}...")
        test_ds = torchvision.datasets.ImageFolder(
            os.path.join(self.data_root, "test"), 
            transform=self.eval_tfms
        )
        
        # Always use 10% of test set for faster evaluation
        from torch.utils.data import Subset
        import random
        
        # Set random seed for reproducibility
        random.seed(42)
        torch.manual_seed(42)
        
        total_samples = len(test_ds)
        subset_size = int(total_samples * 0.1)  # Always 10%
        subset_indices = random.sample(range(total_samples), subset_size)
        test_ds = Subset(test_ds, subset_indices)
        
        print(f"Using 10% of test set: {subset_size}/{total_samples} samples")
        
        # Adjust num_workers based on system capabilities
        num_workers = min(4, os.cpu_count()) if os.cpu_count() else 2
        
        self.test_loader = DataLoader(
            test_ds, 
            batch_size=64, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True
        )
        
        print(f"Test dataset loaded: {len(test_ds)} samples, {len(self.test_loader)} batches")
    
    def load_model(self, model_path, model_type='dnn', model_variant='vanilla'):
        """Load a trained model"""
        print(f"Loading {model_type.upper()} model from {model_path}...")
        
        # Create model architecture
        if model_type.lower() == 'dnn':
            model = get_dnn_model(model_variant, num_classes=10)
        elif model_type.lower() == 'cnn':
            model = get_cnn_model(model_variant, num_classes=10)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load state dict
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
    
    def evaluate_single_run(self, model, return_predictions=False):
        """Evaluate model for a single run"""
        print("Evaluating model...")
        
        all_predictions = []
        all_labels = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader, 1):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if return_predictions:
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                # Progress tracking
                if batch_idx % 50 == 0 or batch_idx == len(self.test_loader):
                    current_accuracy = 100 * correct / total
                    print(f"  Batch {batch_idx}/{len(self.test_loader)} | Current Accuracy: {current_accuracy:.2f}%")
        
        accuracy = 100 * correct / total
        
        if return_predictions:
            return accuracy, np.array(all_predictions), np.array(all_labels)
        else:
            return accuracy
    
    def evaluate_multiple_runs(self, model, num_runs=3):
        """Evaluate model over multiple runs for statistical analysis"""
        print(f"Evaluating model over {num_runs} runs...")
        
        accuracies = []
        all_f1_scores = []
        all_precisions = []
        all_recalls = []
        
        for run in range(num_runs):
            print(f"\n--- Run {run + 1}/{num_runs} ---")
            accuracy, predictions, labels = self.evaluate_single_run(model, return_predictions=True)
            
            # Calculate overall metrics (macro and micro averages)
            f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
            f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
            f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
            
            precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
            precision_micro = precision_score(labels, predictions, average='micro', zero_division=0)
            precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
            
            recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
            recall_micro = recall_score(labels, predictions, average='micro', zero_division=0)
            recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
            
            accuracies.append(accuracy)
            all_f1_scores.append(f1_weighted)  # Use weighted as primary
            all_precisions.append(precision_weighted)  # Use weighted as primary
            all_recalls.append(recall_weighted)  # Use weighted as primary
            
            print(f"Run {run + 1} - Accuracy: {accuracy:.2f}%, F1: {f1_weighted:.4f}, Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}")
        
        # Calculate statistics
        results = {
            'accuracy': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'values': accuracies
            },
            'f1_score': {
                'mean': np.mean(all_f1_scores),
                'std': np.std(all_f1_scores),
                'values': all_f1_scores
            },
            'precision': {
                'mean': np.mean(all_precisions),
                'std': np.std(all_precisions),
                'values': all_precisions
            },
            'recall': {
                'mean': np.mean(all_recalls),
                'std': np.std(all_recalls),
                'values': all_recalls
            }
        }
        
        return results
    
    def visualize_predictions(self, model, num_samples=16, save_path=None):
        """Visualize model predictions with actual vs predicted labels"""
        print(f"Generating prediction visualization for {num_samples} samples...")
        
        # Get predictions and collect sample data
        model.eval()
        sample_images = []
        sample_labels = []
        sample_predictions = []
        
        with torch.no_grad():
            samples_collected = 0
            for images, labels in self.test_loader:
                if samples_collected >= num_samples:
                    break
                    
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                # Collect samples
                batch_size = min(images.size(0), num_samples - samples_collected)
                for i in range(batch_size):
                    # Denormalize image for display
                    img = images[i].cpu()
                    # Denormalize using the same mean and std used in transforms
                    MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                    STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
                    img = img * STD + MEAN
                    img = torch.clamp(img, 0, 1)
                    
                    sample_images.append(img.permute(1, 2, 0).numpy())
                    sample_labels.append(labels[i].cpu().item())
                    sample_predictions.append(predicted[i].cpu().item())
                    samples_collected += 1
        
        # Create visualization
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('Model Predictions: Actual vs Predicted', fontsize=16, fontweight='bold')
        
        for i in range(min(num_samples, len(sample_images))):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            
            # Display image
            ax.imshow(sample_images[i])
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Get class names
            actual_class = self.class_names[sample_labels[i]]
            predicted_class = self.class_names[sample_predictions[i]]
            
            # Color code the prediction
            if sample_labels[i] == sample_predictions[i]:
                color = 'green'
                status = '✓'
            else:
                color = 'red'
                status = '✗'
            
            # Set title with color coding
            ax.set_xlabel(f'Actual: {actual_class}\nPredicted: {predicted_class} {status}', 
                         fontsize=9, color=color, fontweight='bold')
        
        # Hide empty subplots
        for i in range(len(sample_images), 16):
            row = i // 4
            col = i % 4
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Prediction visualization saved to: {save_path}")
        
        plt.show()
        
        return sample_images, sample_labels, sample_predictions

    def generate_detailed_report(self, model, save_path=None):
        """Generate detailed evaluation report with confusion matrix and classification report"""
        print("Generating detailed evaluation report...")
        
        accuracy, predictions, labels = self.evaluate_single_run(model, return_predictions=True)
        
        # Classification report
        report = classification_report(labels, predictions, target_names=self.class_names, output_dict=True, zero_division=0)
        
        # Calculate overall metrics for detailed report
        f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
        precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion matrix heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        axes[1].bar(self.class_names, per_class_acc)
        axes[1].set_title('Per-Class Accuracy')
        axes[1].set_ylabel('Accuracy')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Detailed report saved to: {save_path}")
        
        plt.show()
        
        return {
            'accuracy': accuracy,
            'f1_score': f1_weighted,
            'precision': precision_weighted,
            'recall': recall_weighted,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'per_class_accuracy': dict(zip(self.class_names, per_class_acc))
        }
    
    def save_results(self, results, model_name, save_dir='evaluation_results'):
        """Save evaluation results to JSON file"""
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_evaluation_{timestamp}.json"
        filepath = os.path.join(save_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in value.items()}
            else:
                json_results[key] = value.tolist() if isinstance(value, np.ndarray) else value
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return filepath

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--model_type', type=str, choices=['dnn', 'cnn'], default='dnn', help='Model type')
    parser.add_argument('--model_variant', type=str, default='vanilla', help='Model variant')
    parser.add_argument('--data_root', type=str, default='./dataset', help='Path to CINIC-10 dataset')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of evaluation runs')
    parser.add_argument('--detailed', action='store_true', help='Generate detailed report with confusion matrix')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to visualize in prediction grid (default: 16)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(data_root=args.data_root)
    
    # Load model
    model = evaluator.load_model(args.model_path, args.model_type, args.model_variant)
    
    # Run evaluation
    if args.detailed:
        # Single detailed evaluation
        results = evaluator.generate_detailed_report(model, f"{args.model_type}_{args.model_variant}_detailed_report.png")
        evaluator.save_results(results, f"{args.model_type}_{args.model_variant}_detailed")
    else:
        # Multiple runs evaluation
        results = evaluator.evaluate_multiple_runs(model, args.num_runs)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY - {args.model_type.upper()} ({args.model_variant})")
        print(f"{'='*60}")
        print(f"Top-1 Accuracy: {results['accuracy']['mean']:.2f}% ± {results['accuracy']['std']:.2f}%")
        print(f"F1 Score:       {results['f1_score']['mean']:.4f} ± {results['f1_score']['std']:.4f}")
        print(f"Precision:      {results['precision']['mean']:.4f} ± {results['precision']['std']:.4f}")
        print(f"Recall:         {results['recall']['mean']:.4f} ± {results['recall']['std']:.4f}")
        print(f"{'='*60}")
        
        # Save results
        evaluator.save_results(results, f"{args.model_type}_{args.model_variant}")
    
    # Always generate prediction visualization as part of main evaluation
    print(f"\n{'='*60}")
    print("GENERATING PREDICTION VISUALIZATION")
    print(f"{'='*60}")
    evaluator.visualize_predictions(model, args.num_samples, 
                                  f"{args.model_type}_{args.model_variant}_predictions.png")

if __name__ == "__main__":
    main()
