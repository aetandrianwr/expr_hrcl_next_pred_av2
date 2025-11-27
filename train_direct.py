import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import random
import json
from tqdm import tqdm

from model_direct import DirectTransitionModel
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
    m_dict = {"correct@1": 0, "correct@3": 0, "correct@5": 0, "correct@10": 0,
              "rr": 0, "ndcg": 0, "f1": 0, "total": 0}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval', leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(batch)
            loss = criterion(logits, batch['Y'])
            total_loss += loss.item()
            
            m, _, _ = calculate_correct_total_prediction(logits, batch['Y'])
            for i, k in enumerate(["correct@1", "correct@3", "correct@5", "correct@10", "f1", "rr", "ndcg", "total"]):
                m_dict[k] += m[i]
    
    return total_loss / len(loader), get_performance_dict(m_dict)

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DirectTransitionModel({
        'd_model': 96,
        'num_locations': 1200,
        'num_users': 50,
        'num_s2_l14': 1260,
    }).to(device)
    
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        '/content/expr_hrcl_next_pred_av2/data/geolife', batch_size=512, max_len=60
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    
    steps = len(train_loader) * 100
    scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=steps, pct_start=0.05, 
                           div_factor=25, final_div_factor=10000)
    
    best = 0
    patience = 0
    
    for epoch in range(1, 101):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_perf = evaluate(model, val_loader, criterion, device)
        
        print(f"Ep {epoch:3d} | TL:{train_loss:.3f} VL:{val_loss:.3f} | "
              f"Acc@1:{val_perf['acc@1']:.2f}% MRR:{val_perf['mrr']:.2f}%")
        
        if val_perf['acc@1'] > best:
            best = val_perf['acc@1']
            patience = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  ✓ {best:.2f}%")
        else:
            patience += 1
            if patience >= 25:
                break
    
    model.load_state_dict(torch.load('best_model.pt'))
    _, test_perf = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTEST: {test_perf['acc@1']:.2f}%")
    
    with open('results.json', 'w') as f:
        json.dump(test_perf, f)
    
    if test_perf['acc@1'] >= 50:
        print("🎉 SUCCESS")
    else:
        print(f"Need {50 - test_perf['acc@1']:.1f}% more")

if __name__ == '__main__':
    main()
