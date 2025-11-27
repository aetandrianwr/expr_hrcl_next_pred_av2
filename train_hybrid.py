import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import random
import json
from tqdm import tqdm

from model_hybrid import HybridPatternPredictor
from dataset import get_dataloaders
from metrics import calculate_correct_total_prediction, get_performance_dict

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc='Train', leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch['Y'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    metrics = {"correct@1": 0, "correct@3": 0, "correct@5": 0, "correct@10": 0,
               "rr": 0, "ndcg": 0, "f1": 0, "total": 0}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval', leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(batch)
            loss = criterion(logits, batch['Y'])
            total_loss += loss.item()
            
            m, _, _ = calculate_correct_total_prediction(logits, batch['Y'])
            for i, k in enumerate(["correct@1", "correct@3", "correct@5", "correct@10", "f1", "rr", "ndcg", "total"]):
                metrics[k] += m[i]
    
    return total_loss / len(loader), get_performance_dict(metrics)

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HybridPatternPredictor({
        'd_model': 64,
        'num_locations': 1200,
        'num_users': 50,
        'num_weekdays': 8,
        'num_s2_l13': 930,
        'num_s2_l14': 1260,
    }).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        '/content/expr_hrcl_next_pred_av2/data/geolife', batch_size=256, max_len=60
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_val_acc = 0
    patience = 0
    
    for epoch in range(1, 81):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_perf = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Acc@1: {val_perf['acc@1']:.2f}% | MRR: {val_perf['mrr']:.2f}%")
        
        if val_perf['acc@1'] > best_val_acc:
            best_val_acc = val_perf['acc@1']
            patience = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  🎯 NEW BEST: {best_val_acc:.2f}%")
        else:
            patience += 1
            if patience >= 20:
                break
    
    model.load_state_dict(torch.load('best_model.pt'))
    _, test_perf = evaluate(model, test_loader, criterion, device)
    
    print(f"\n{'='*80}")
    print(f"TEST: Acc@1={test_perf['acc@1']:.2f}%, Acc@5={test_perf['acc@5']:.2f}%, MRR={test_perf['mrr']:.2f}%")
    print(f"{'='*80}")
    
    with open('results.json', 'w') as f:
        json.dump({'test': test_perf, 'params': num_params}, f, indent=2)
    
    if test_perf['acc@1'] >= 50:
        print(f"🎉 SUCCESS: {test_perf['acc@1']:.2f}%")
    else:
        print(f"Current: {test_perf['acc@1']:.2f}% (need 50%)")

if __name__ == '__main__':
    main()
