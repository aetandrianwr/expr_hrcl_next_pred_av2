import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import random
import json
from tqdm import tqdm
from model_final import FinalModel
from dataset import get_dataloaders
from metrics import calculate_correct_total_prediction, get_performance_dict

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc='Train', leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch['Y'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    m = {"correct@1": 0, "correct@3": 0, "correct@5": 0, "correct@10": 0,
         "rr": 0, "ndcg": 0, "f1": 0, "total": 0}
    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval', leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(batch)
            loss = criterion(logits, batch['Y'])
            total_loss += loss.item()
            arr, _, _ = calculate_correct_total_prediction(logits, batch['Y'])
            for i, k in enumerate(["correct@1", "correct@3", "correct@5", "correct@10", "f1", "rr", "ndcg", "total"]):
                m[k] += arr[i]
    return total_loss / len(loader), get_performance_dict(m)

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FinalModel({
        'd_model': 96,
        'num_locations': 1200,
        'num_users': 50,
        'num_s2_l11': 320,
        'num_s2_l12': 680,
        'num_s2_l13': 930,
        'num_s2_l14': 1260,
    }).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        '/content/expr_hrcl_next_pred_av2/data/geolife', batch_size=256, max_len=60
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    
    steps = len(train_loader) * 80
    scheduler = OneCycleLR(optimizer, max_lr=0.003, total_steps=steps, pct_start=0.1)
    
    best_val = 0
    patience = 0
    
    for epoch in range(1, 81):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_perf = evaluate(model, val_loader, criterion, device)
        
        print(f"Ep {epoch:2d} | TL:{train_loss:.3f} VL:{val_loss:.3f} | Acc@1:{val_perf['acc@1']:.2f}% MRR:{val_perf['mrr']:.2f}%")
        
        if val_perf['acc@1'] > best_val:
            best_val = val_perf['acc@1']
            patience = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  ✓ {best_val:.2f}%")
        else:
            patience += 1
            if patience >= 25:
                break
    
    model.load_state_dict(torch.load('best_model.pt'))
    _, test_perf = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTEST: Acc@1={test_perf['acc@1']:.2f}%")
    
    with open('results.json', 'w') as f:
        json.dump({'test': test_perf, 'params': params}, f)
    
    if test_perf['acc@1'] >= 50:
        print("🎉 SUCCESS!")

if __name__ == '__main__':
    main()
