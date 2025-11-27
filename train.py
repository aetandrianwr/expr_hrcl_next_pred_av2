import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import random
import os
import json
from tqdm import tqdm

from model_v3 import SimpleHierarchicalTransformer as HierarchicalTransformer
from dataset import get_dataloaders
from metrics import calculate_correct_total_prediction, get_performance_dict


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        logits = model(batch)
        loss = criterion(logits, batch['Y'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': total_loss / num_batches})
    
    return total_loss / num_batches


def evaluate(model, data_loader, criterion, device, split_name='Val'):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Metrics accumulation
    metrics_dict = {
        "correct@1": 0,
        "correct@3": 0,
        "correct@5": 0,
        "correct@10": 0,
        "rr": 0,
        "ndcg": 0,
        "f1": 0,
        "total": 0,
    }
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f'[{split_name}]')
        for batch in pbar:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            logits = model(batch)
            loss = criterion(logits, batch['Y'])
            
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate metrics
            metrics_array, _, _ = calculate_correct_total_prediction(logits, batch['Y'])
            metrics_dict["correct@1"] += metrics_array[0]
            metrics_dict["correct@3"] += metrics_array[1]
            metrics_dict["correct@5"] += metrics_array[2]
            metrics_dict["correct@10"] += metrics_array[3]
            metrics_dict["f1"] += metrics_array[4]
            metrics_dict["rr"] += metrics_array[5]
            metrics_dict["ndcg"] += metrics_array[6]
            metrics_dict["total"] += metrics_array[7]
            
            # Update progress bar
            current_acc = metrics_dict["correct@1"] / metrics_dict["total"] * 100
            pbar.set_postfix({'loss': total_loss / num_batches, 'acc@1': f'{current_acc:.2f}%'})
    
    avg_loss = total_loss / num_batches
    perf = get_performance_dict(metrics_dict)
    
    return avg_loss, perf


def main():
    # Set seed
    set_seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data directory
    data_dir = '/content/expr_hrcl_next_pred_av2/data/geolife'
    
    # Hyperparameters  
    config = {
        'd_model': 96,
        'nhead': 4,
        'num_layers': 4,
        'dropout': 0.2,
        'num_locations': 1200,
        'num_users': 50,
        'num_weekdays': 8,
        'num_s2_l11': 320,
        'num_s2_l12': 680,
        'num_s2_l13': 930,
        'num_s2_l14': 1260,
    }
    
    batch_size = 128
    learning_rate = 0.002
    num_epochs = 80
    patience = 15
    max_len = 60
    label_smoothing = 0.05
    
    # Create model
    model = HierarchicalTransformer(config).to(device)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    if num_params >= 1_000_000:
        print(f"WARNING: Model has {num_params:,} parameters (>= 1M). Need to reduce!")
        return
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir, batch_size=batch_size, max_len=max_len
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7, verbose=True, min_lr=1e-6)
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    results = {
        'train_loss': [],
        'val_loss': [],
        'val_acc@1': [],
        'best_val_acc@1': 0,
        'best_epoch': 0,
        'test_metrics': None,
    }
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_perf = evaluate(model, val_loader, criterion, device, 'Val')
        
        # Update scheduler
        scheduler.step(val_perf['acc@1'])
        
        # Log
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)
        results['val_acc@1'].append(val_perf['acc@1'])
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Acc@1: {val_perf['acc@1']:.2f}%")
        print(f"  Val Acc@5: {val_perf['acc@5']:.2f}%")
        print(f"  Val MRR: {val_perf['mrr']:.2f}%")
        
        # Early stopping
        if val_perf['acc@1'] > best_val_acc:
            best_val_acc = val_perf['acc@1']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': config,
            }, 'best_model.pt')
            print(f"  ✓ New best model saved! (Acc@1: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
    
    # Load best model and evaluate on test set
    print("\n" + "="*80)
    print("Loading best model and evaluating on test set...")
    print("="*80 + "\n")
    
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_perf = evaluate(model, test_loader, criterion, device, 'Test')
    
    print("\n" + "="*80)
    print("FINAL TEST RESULTS")
    print("="*80)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc@1: {test_perf['acc@1']:.2f}%")
    print(f"Test Acc@5: {test_perf['acc@5']:.2f}%")
    print(f"Test Acc@10: {test_perf['acc@10']:.2f}%")
    print(f"Test MRR: {test_perf['mrr']:.2f}%")
    print(f"Test NDCG: {test_perf['ndcg']:.2f}%")
    print(f"Test F1: {test_perf['f1']:.4f}")
    print("="*80 + "\n")
    
    # Save results
    results['best_val_acc@1'] = best_val_acc
    results['best_epoch'] = best_epoch
    results['test_metrics'] = test_perf
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to results.json")
    print(f"Best model saved to best_model.pt")
    
    # Check if we met the goal
    if test_perf['acc@1'] >= 50.0:
        print(f"\n🎉 SUCCESS! Achieved {test_perf['acc@1']:.2f}% Acc@1 (>= 50%)")
    elif test_perf['acc@1'] >= 45.0:
        print(f"\n⚠️  Close! Achieved {test_perf['acc@1']:.2f}% Acc@1 (>= 45% but < 50%)")
    else:
        print(f"\n❌ Need improvement. Achieved {test_perf['acc@1']:.2f}% Acc@1 (< 45%)")


if __name__ == '__main__':
    main()
