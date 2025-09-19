
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Import model architectures
sys.path.append('model_architecture')
from DNN import get_dnn_model
from CNN import get_cnn_model

# =============================================================================
# QUICK CONFIGURATION - Change these values for easy testing
# =============================================================================
QUICK_TEST_MODE = True  # Set to True for quick testing with 10 samples
QUICK_TEST_SAMPLES = 4  # Number of samples for quick test
QUICK_MODEL_TYPE = 'dnn'  # 'dnn' or 'cnn'
QUICK_MODEL_VARIANT = 'vanilla'  # 'vanilla', 'deep', 'simple', 'advanced'
# =============================================================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train image classification models')
    parser.add_argument('--model_type', type=str, choices=['dnn', 'cnn'], default='dnn', 
                       help='Model architecture type (dnn or cnn)')
    parser.add_argument('--model_variant', type=str, default='vanilla', 
                       help='Model variant (vanilla, deep, simple, advanced)')
    parser.add_argument('--test_mode', action='store_true', 
                       help='Run in test mode with 10 samples')
    parser.add_argument('--test_samples', type=int, default=10, 
                       help='Number of samples for test mode')
    parser.add_argument('--data_root', type=str, default='./dataset', 
                       help='Path to CINIC-10 dataset')
    parser.add_argument('--auto_evaluate', action='store_true', 
                       help='Automatically run evaluation after training')
    parser.add_argument('--num_eval_runs', type=int, default=3, 
                       help='Number of evaluation runs for statistical analysis')
    
    args = parser.parse_args()
    
    # Configuration - Use quick config if no command line args provided, otherwise use args
    if len(sys.argv) == 1:  # No command line arguments provided
        print("🚀 Using QUICK CONFIGURATION (no command line args provided)")
        TEST_MODE = QUICK_TEST_MODE
        TEST_SAMPLES = QUICK_TEST_SAMPLES
        MODEL_TYPE = QUICK_MODEL_TYPE
        MODEL_VARIANT = QUICK_MODEL_VARIANT
        DATA_ROOT = './dataset'  # Default dataset path
        AUTO_EVALUATE = False
        NUM_EVAL_RUNS = 3
    else:  # Command line arguments provided
        print("📋 Using COMMAND LINE ARGUMENTS")
        TEST_MODE = args.test_mode
        TEST_SAMPLES = args.test_samples
        MODEL_TYPE = args.model_type
        MODEL_VARIANT = args.model_variant
        DATA_ROOT = args.data_root
        AUTO_EVALUATE = args.auto_evaluate
        NUM_EVAL_RUNS = args.num_eval_runs
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    if TEST_MODE:
        print(f"\nQUICK TEST MODE: Using only {TEST_SAMPLES} samples per dataset")
    else:
        print("\nFULL TRAINING MODE: Using complete dataset")
    
    # Data Loading & Preprocessing
    print("\n=== Data Loading & Preprocessing ===")
    
    # Check if data directory exists
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data directory not found at {DATA_ROOT}")
        sys.exit(1)
    
    # Checking if data has successfully loaded
    required_subdirs = ["train", "valid", "test"]
    for sub in required_subdirs:
        path = os.path.join(DATA_ROOT, sub)
        if os.path.isdir(path):
            print(f"Found subfolder: {path}")
        else:
            raise FileNotFoundError(f"Expected subfolder not found: {path}")
    
    # Data transforms
    MEAN = (0.4914, 0.4822, 0.4465)
    STD  = (0.2470, 0.2435, 0.2616)
    
    train_tfms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    eval_tfms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    # Load dataset
    print("Loading datasets...")
    train_ds = torchvision.datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_tfms)
    valid_ds = torchvision.datasets.ImageFolder(os.path.join(DATA_ROOT, "valid"), transform=eval_tfms)
    test_ds  = torchvision.datasets.ImageFolder(os.path.join(DATA_ROOT, "test"),  transform=eval_tfms)
    
    # Create subset for quick testing
    if TEST_MODE:
        from torch.utils.data import Subset
        import random
        
        # Set random seed for reproducibility
        random.seed(42)
        torch.manual_seed(42)
        
        # Create indices for subset
        train_indices = random.sample(range(len(train_ds)), min(TEST_SAMPLES, len(train_ds)))
        valid_indices = random.sample(range(len(valid_ds)), min(TEST_SAMPLES, len(valid_ds)))
        test_indices = random.sample(range(len(test_ds)), min(TEST_SAMPLES, len(test_ds)))
        
        # Create subsets
        train_ds = Subset(train_ds, train_indices)
        valid_ds = Subset(valid_ds, valid_indices)
        test_ds = Subset(test_ds, test_indices)
        
        print(f"Created subsets: Train={len(train_ds)}, Valid={len(valid_ds)}, Test={len(test_ds)}")
    
    # Adjust num_workers based on system capabilities
    num_workers = min(4, os.cpu_count()) if os.cpu_count() else 2
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Quick sanity checks
    if not TEST_MODE:  # Only show classes for full dataset
        print("Classes:", train_ds.classes)
    print("Train/Valid/Test sizes:", len(train_ds), len(valid_ds), len(test_ds))
    
    
    # Model Architecture
    print(f"\n=== Model Architecture: {MODEL_TYPE.upper()} ({MODEL_VARIANT}) ===")
    
    # Create model based on arguments
    if MODEL_TYPE == 'dnn':
        model = get_dnn_model(MODEL_VARIANT, num_classes=10)
    elif MODEL_TYPE == 'cnn':
        model = get_cnn_model(MODEL_VARIANT, num_classes=10)
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")
    
    # Initialize model, loss, optimizer
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    print(f"Model initialized on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    print("\n=== Training ===")
    
    # Adjust training parameters for quick test
    if TEST_MODE:
        num_epochs = 3  # Fewer epochs for quick testing
        log_every = 1   # Log every batch for small dataset
        print(f"Quick test mode: {num_epochs} epochs, logging every {log_every} batch")
    else:
        num_epochs = 30
        log_every = 20
    
    batch_losses = []         # per-batch loss (training)
    batch_steps  = []         # global batch index for plotting
    epoch_losses = []         # average train loss per epoch
    epoch_accs   = []         # average train accuracy per epoch
    epoch_val_accs = []       # validation accuracy per epoch
    
    global_step = 0
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_examples = 0
        running_correct = 0
    
        for b, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(images)      
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # logging
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_examples += batch_size
    
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
    
            # record per-batch loss
            global_step += 1
            batch_losses.append(loss.item())
            batch_steps.append(global_step)
    
            if b % log_every == 0:
                print(f"Epoch {epoch} | Batch {b}/{len(train_loader)} | batch_loss={loss.item():.4f}")
    
        epoch_loss = running_loss / running_examples
        epoch_acc  = running_correct / running_examples
        epoch_losses.append(epoch_loss)
        epoch_accs.append(epoch_acc)
        
        # Validation evaluation (limit to first 10 batches for efficiency)
        print(f"  Running validation evaluation...")
        model.eval()
        val_correct, val_total = 0, 0
        max_val_batches = min(10, len(valid_loader))  # Limit validation batches for efficiency
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(valid_loader, 1):
                if batch_idx > max_val_batches:
                    break
                    
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Progress tracking for validation
                if batch_idx % 5 == 0 or batch_idx == max_val_batches:
                    current_val_acc = val_correct / val_total
                    print(f"    Validation batch {batch_idx}/{max_val_batches} | Current accuracy: {current_val_acc:.2%}")
        
        val_acc = val_correct / val_total
        epoch_val_accs.append(val_acc)
        model.train()  # Set back to training mode
    
        print(f"Epoch {epoch} done | avg_loss={epoch_loss:.4f} | train_acc={epoch_acc:.2%} | val_acc={val_acc:.2%}")
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    
    # Save model checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    
    if TEST_MODE:
        checkpoint_path = f"checkpoints/{MODEL_TYPE}_{MODEL_VARIANT}_quick_test.pth"
    else:
        checkpoint_path = f"checkpoints/{MODEL_TYPE}_{MODEL_VARIANT}_trained.pth"
    
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nModel saved to: {checkpoint_path}")
    
    # Evaluation
    print("\n=== Evaluation ===")
    correct, total = 0, 0
    model.eval()
    
    print(f"Evaluating on {len(test_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader, 1):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Progress tracking
            if batch_idx % 50 == 0 or batch_idx == len(test_loader):
                current_accuracy = 100 * correct / total
                print(f"  Batch {batch_idx}/{len(test_loader)} | Current Accuracy: {current_accuracy:.2f}% | Processed: {total} samples")
    
    test_accuracy = 100 * correct / total
    print(f"\n Final Test Accuracy: {test_accuracy:.2f}% ({correct}/{total} correct)")
    
    # Plotting - All charts in one figure
    print("\n=== Generating Plots ===")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress Overview', fontsize=16, fontweight='bold')
    
    # 1) Per-batch training loss (global batches)
    axes[0, 0].plot(batch_steps, batch_losses, color='blue', alpha=0.7)
    axes[0, 0].set_xlabel("Global Batch")
    axes[0, 0].set_ylabel("Train Loss (per batch)")
    axes[0, 0].set_title("Per-batch Training Loss")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2) Per-epoch average training loss
    axes[0, 1].plot(range(1, len(epoch_losses)+1), epoch_losses, marker="o", color='red', linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Avg Train Loss (epoch)")
    axes[0, 1].set_title("Per-epoch Average Training Loss")
    axes[0, 1].set_xticks(range(1, len(epoch_losses)+1))
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3) Training and Validation accuracy per epoch
    axes[1, 0].plot(range(1, len(epoch_accs)+1), epoch_accs, marker="o", color='green', linewidth=2, label='Training')
    axes[1, 0].plot(range(1, len(epoch_val_accs)+1), epoch_val_accs, marker="s", color='orange', linewidth=2, label='Validation')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Training vs Validation Accuracy")
    axes[1, 0].set_xticks(range(1, len(epoch_accs)+1))
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)  # Set y-axis from 0 to 1 for accuracy
    axes[1, 0].legend()
    
    # 4) Summary statistics
    axes[1, 1].axis('off')  # Turn off axis for text summary
    summary_text = f"""
    Training Summary:
    
    • Final Test Accuracy: {test_accuracy:.2f}%
    • Total Epochs: {len(epoch_losses)}
    • Total Batches: {len(batch_losses)}
    • Final Training Loss: {epoch_losses[-1]:.4f}
    • Final Training Accuracy: {epoch_accs[-1]:.2%}
    • Final Validation Accuracy: {epoch_val_accs[-1]:.2%}
    • Dataset Size: {len(train_ds)} train, {len(valid_ds)} valid, {len(test_ds)} test
    • Model Parameters: {sum(p.numel() for p in model.parameters()):,}
    • Device: {device}
    """
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("training_overview.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    
    print("\n=== Training Complete ===")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    if TEST_MODE:
        print("Quick test completed! Use --test_mode=False to run on full dataset.")
    print("Plots saved as PNG files in the current directory")
    print("Main overview: training_overview.png")
    
    # Automatic evaluation if requested
    if AUTO_EVALUATE and not TEST_MODE:
        print("\n=== Running Automatic Evaluation ===")
        try:
            sys.path.append('scripts')
            from evaluate import ModelEvaluator
            evaluator = ModelEvaluator(data_root=DATA_ROOT, device=device)
            # Single evaluation run (multiple runs don't make sense for a single trained model)
            accuracy, predictions, labels = evaluator.evaluate_single_run(model, return_predictions=True)
            
            # Calculate metrics
            from sklearn.metrics import f1_score, precision_score, recall_score
            f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
            precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
            recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
            
            print(f"\n{'='*50}")
            print(f"AUTOMATIC EVALUATION SUMMARY - {MODEL_TYPE.upper()} ({MODEL_VARIANT})")
            print(f"{'='*50}")
            print(f"Top-1 Accuracy: {accuracy:.2f}%")
            print(f"F1 Score: {f1_weighted:.4f}")
            print(f"Precision: {precision_weighted:.4f}")
            print(f"Recall: {recall_weighted:.4f}")
            print(f"{'='*50}")
            
            # Create results dict for saving
            results = {
                'accuracy': {'mean': accuracy, 'std': 0.0},
                'f1_score': {'mean': f1_weighted, 'std': 0.0},
                'precision': {'mean': precision_weighted, 'std': 0.0},
                'recall': {'mean': recall_weighted, 'std': 0.0}
            }
            
            # Save evaluation results
            evaluator.save_results(results, f"{MODEL_TYPE}_{MODEL_VARIANT}_auto_eval")
            
            # Generate prediction visualization
            print(f"\n{'='*50}")
            print("GENERATING PREDICTION VISUALIZATION")
            print(f"{'='*50}")
            evaluator.visualize_predictions(model, 16, f"{MODEL_TYPE}_{MODEL_VARIANT}_auto_predictions.png")
            
        except ImportError:
            print("Warning: Could not import evaluation module. Run evaluation manually.")
        except Exception as e:
            print(f"Warning: Automatic evaluation failed: {e}")
    
    print(f"\nTo run detailed evaluation manually:")
    print(f"python scripts/evaluate.py --model_path {checkpoint_path} --model_type {MODEL_TYPE} --model_variant {MODEL_VARIANT} --detailed")

if __name__ == "__main__":
    main()
