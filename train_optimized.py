"""
Optimized training with best practices for next-location prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import random
import json
from tqdm import tqdm

from model_optimized import OptimizedHierarchicalPredictor
from dataset import get_dataloaders
from metrics import calculate_correct_total_prediction, get_performance_dict


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        optimizer.zero_grad(set_to_none=True)  # More efficient
        
        logits = model(batch)
        loss = criterion(logits, batch['Y'])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # OneCycleLR steps every batch
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{total_loss / num_batches:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
    
    return total_loss / num_batches


def evaluate(model, data_loader, criterion, device, split_name='Val'):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    metrics_dict = {
        "correct@1": 0, "correct@3": 0, "correct@5": 0, "correct@10": 0,
        "rr": 0, "ndcg": 0, "f1": 0, "total": 0,
    }
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f'[{split_name}]')
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            logits = model(batch)
            loss = criterion(logits, batch['Y'])
            
            total_loss += loss.item()
            num_batches += 1
            
            metrics_array, _, _ = calculate_correct_total_prediction(logits, batch['Y'])
            metrics_dict["correct@1"] += metrics_array[0]
            metrics_dict["correct@3"] += metrics_array[1]
            metrics_dict["correct@5"] += metrics_array[2]
            metrics_dict["correct@10"] += metrics_array[3]
            metrics_dict["f1"] += metrics_array[4]
            metrics_dict["rr"] += metrics_array[5]
            metrics_dict["ndcg"] += metrics_array[6]
            metrics_dict["total"] += metrics_array[7]
            
            current_acc = metrics_dict["correct@1"] / metrics_dict["total"] * 100
            pbar.set_postfix({'loss': f'{total_loss / num_batches:.4f}', 'acc@1': f'{current_acc:.2f}%'})
    
    avg_loss = total_loss / num_batches
    perf = get_performance_dict(metrics_dict)
    
    return avg_loss, perf


def main():
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = '/content/expr_hrcl_next_pred_av2/data/geolife'
    
    # Optimized config based on data analysis
    config = {
        'd_model': 112,        # Maximum within budget
        'nhead': 8,            # More heads for multi-resolution attention
        'num_layers': 2,       # 2 strong layers (979,800 params)
        'dropout': 0.15,       # Lower dropout - we have enough data
        'num_locations': 1200,
        'num_users': 50,
        'num_weekdays': 8,
        'num_s2_l11': 320,
        'num_s2_l12': 680,
        'num_s2_l13': 930,
        'num_s2_l14': 1260,
    }
    
    # Training hyperparameters - best practices
    batch_size = 256       # Larger batch for stable gradients
    max_lr = 0.003         # Higher LR with OneCycleLR
    num_epochs = 50        # Fewer epochs with better LR schedule
    warmup_epochs = 3
    max_len = 60
    
    model = OptimizedHierarchicalPredictor(config).to(device)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    if num_params >= 1_000_000:
        print(f"ERROR: Model has {num_params:,} parameters (>= 1M)")
        return
    
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir, batch_size=batch_size, max_len=max_len
    )
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # AdamW with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    
    # OneCycleLR - proven to work better than ReduceLROnPlateau
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=warmup_epochs / num_epochs,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    patience = 10
    
    results = {
        'train_loss': [], 'val_loss': [], 'val_acc@1': [],
        'best_val_acc@1': 0, 'best_epoch': 0, 'test_metrics': None,
    }
    
    print("\n" + "="*80)
    print("Starting optimized training...")
    print("="*80 + "\n")
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        val_loss, val_perf = evaluate(model, val_loader, criterion, device, 'Val')
        
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)
        results['val_acc@1'].append(val_perf['acc@1'])
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc@1: {val_perf['acc@1']:.2f}%, Acc@5: {val_perf['acc@5']:.2f}%, MRR: {val_perf['mrr']:.2f}%")
        
        if val_perf['acc@1'] > best_val_acc:
            best_val_acc = val_perf['acc@1']
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': config,
            }, 'best_model.pt')
            print(f"  ✓ NEW BEST! Val Acc@1: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Test evaluation
    print("\n" + "="*80)
    print("TESTING BEST MODEL")
    print("="*80 + "\n")
    
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_perf = evaluate(model, test_loader, criterion, device, 'Test')
    
    print("\n" + "="*80)
    print("FINAL TEST RESULTS")
    print("="*80)
    print(f"Test Acc@1:  {test_perf['acc@1']:.2f}%")
    print(f"Test Acc@5:  {test_perf['acc@5']:.2f}%")
    print(f"Test Acc@10: {test_perf['acc@10']:.2f}%")
    print(f"Test MRR:    {test_perf['mrr']:.2f}%")
    print(f"Test NDCG:   {test_perf['ndcg']:.2f}%")
    print("="*80 + "\n")
    
    results['best_val_acc@1'] = best_val_acc
    results['best_epoch'] = best_epoch
    results['test_metrics'] = test_perf
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    if test_perf['acc@1'] >= 50.0:
        print(f"✅ SUCCESS! Achieved {test_perf['acc@1']:.2f}% (>= 50%)")
    elif test_perf['acc@1'] >= 45.0:
        print(f"⚠️  Close: {test_perf['acc@1']:.2f}% (>= 45%)")
    else:
        print(f"❌ Failed: {test_perf['acc@1']:.2f}% (< 45%)")
    
    print(f"\nModel size: {num_params:,} parameters")


if __name__ == '__main__':
    main()
