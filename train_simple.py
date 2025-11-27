"""
Simple effective training - no tricks, just solid fundamentals
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import random
import json
from tqdm import tqdm

from model_simple import SimpleEffectivePredictor
from dataset import get_dataloaders
from metrics import calculate_correct_total_prediction, get_performance_dict


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc='Train'):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch['Y'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, name='Val'):
    model.eval()
    total_loss = 0
    metrics_dict = {
        "correct@1": 0, "correct@3": 0, "correct@5": 0, "correct@10": 0,
        "rr": 0, "ndcg": 0, "f1": 0, "total": 0,
    }
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=name):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            logits = model(batch)
            loss = criterion(logits, batch['Y'])
            total_loss += loss.item()
            
            metrics_array, _, _ = calculate_correct_total_prediction(logits, batch['Y'])
            metrics_dict["correct@1"] += metrics_array[0]
            metrics_dict["correct@3"] += metrics_array[1]
            metrics_dict["correct@5"] += metrics_array[2]
            metrics_dict["correct@10"] += metrics_array[3]
            metrics_dict["f1"] += metrics_array[4]
            metrics_dict["rr"] += metrics_array[5]
            metrics_dict["ndcg"] += metrics_array[6]
            metrics_dict["total"] += metrics_array[7]
    
    return total_loss / len(loader), get_performance_dict(metrics_dict)


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = {
        'd_model': 64,
        'num_locations': 1200,
        'num_users': 50,
        'num_weekdays': 8,
        'num_s2_l11': 320,
        'num_s2_l12': 680,
        'num_s2_l13': 930,
        'num_s2_l14': 1260,
    }
    
    model = SimpleEffectivePredictor(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        '/content/expr_hrcl_next_pred_av2/data/geolife', batch_size=256, max_len=60
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(1, 51):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_perf = evaluate(model, val_loader, criterion, device, 'Val')
        scheduler.step()
        
        print(f"\nEpoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        print(f"  Val Acc@1={val_perf['acc@1']:.2f}%, Acc@5={val_perf['acc@5']:.2f}%, MRR={val_perf['mrr']:.2f}%")
        
        if val_perf['acc@1'] > best_val_acc:
            best_val_acc = val_perf['acc@1']
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  ✓ NEW BEST!")
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print("Early stopping")
                break
    
    # Test
    model.load_state_dict(torch.load('best_model.pt'))
    test_loss, test_perf = evaluate(model, test_loader, criterion, device, 'Test')
    
    print("\n" + "="*80)
    print("FINAL TEST RESULTS")
    print("="*80)
    print(f"Acc@1: {test_perf['acc@1']:.2f}%")
    print(f"Acc@5: {test_perf['acc@5']:.2f}%")
    print(f"MRR:   {test_perf['mrr']:.2f}%")
    print("="*80)
    
    with open('results.json', 'w') as f:
        json.dump({'test': test_perf, 'params': num_params}, f, indent=2)
    
    if test_perf['acc@1'] >= 50:
        print(f"\n✅ SUCCESS: {test_perf['acc@1']:.2f}%")
    else:
        print(f"\n❌ FAILED: {test_perf['acc@1']:.2f}% (need 50%)")


if __name__ == '__main__':
    main()
