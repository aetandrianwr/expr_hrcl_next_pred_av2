"""
Training script for next-location prediction
Target: ≥50% Acc@1 on test set
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import random
import json
from tqdm import tqdm

from model import UserAwareTransformer
from dataset import get_dataloaders
from config import CONFIG
from metrics import calculate_correct_total_prediction, get_performance_dict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}', leave=False)
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch['Y'])
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.3f}'})
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, name='Val'):
    model.eval()
    total_loss = 0
    m_dict = {
        "correct@1": 0, "correct@3": 0, "correct@5": 0, "correct@10": 0,
        "rr": 0, "ndcg": 0, "f1": 0, "total": 0,
    }
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=name, leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            logits = model(batch)
            loss = criterion(logits, batch['Y'])
            total_loss += loss.item()
            
            m, _, _ = calculate_correct_total_prediction(logits, batch['Y'])
            for i, k in enumerate(["correct@1", "correct@3", "correct@5", "correct@10", 
                                   "f1", "rr", "ndcg", "total"]):
                m_dict[k] += m[i]
    
    return total_loss / len(loader), get_performance_dict(m_dict)


def main():
    set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("USER-AWARE HIERARCHICAL TRANSFORMER TRAINING")
    print("="*80)
    print(f"Target: ≥50% Acc@1 on test set")
    print(f"Parameter budget: <500K")
    print("="*80)
    
    # Model
    model = UserAwareTransformer(CONFIG).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {params:,} parameters ({'✓ PASS' if params < 500000 else '✗ FAIL'})")
    
    if params >= 500000:
        print(f"ERROR: Exceeds budget by {params - 500000:,}")
        return
    
    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        CONFIG['data_path'],
        batch_size=CONFIG['batch_size'],
        max_len=CONFIG['max_seq_len']
    )
    
    print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, "
          f"{len(test_loader.dataset)} test")
    
    # Loss & optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay'],
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # Scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * CONFIG['epochs']
    scheduler = OneCycleLR(
        optimizer,
        max_lr=CONFIG['lr'],
        total_steps=total_steps,
        pct_start=CONFIG['warmup_epochs'] / CONFIG['epochs'],
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    patience = 0
    history = []
    
    print(f"\nStarting training...")
    print(f"Epochs: {CONFIG['epochs']}, Patience: {CONFIG['patience']}\n")
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        val_loss, val_perf = evaluate(model, val_loader, criterion, device, 'Val')
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc@1': val_perf['acc@1'],
            'val_mrr': val_perf['mrr']
        })
        
        print(f"Epoch {epoch:3d} | TL:{train_loss:.4f} VL:{val_loss:.4f} | "
              f"Acc@1:{val_perf['acc@1']:6.2f}% Acc@5:{val_perf['acc@5']:6.2f}% MRR:{val_perf['mrr']:6.2f}%", end='')
        
        if val_perf['acc@1'] > best_val_acc:
            best_val_acc = val_perf['acc@1']
            best_epoch = epoch
            patience = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': best_val_acc,
                'config': CONFIG,
            }, 'best_model.pt')
            
            print(f"  ✓ BEST")
        else:
            patience += 1
            print()
            
            if patience >= CONFIG['patience']:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Test evaluation
    print("\n" + "="*80)
    print("FINAL TEST EVALUATION")
    print("="*80)
    
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_perf = evaluate(model, test_loader, criterion, device, 'Test')
    
    print(f"\nBest validation: Epoch {best_epoch}, Acc@1 = {best_val_acc:.2f}%")
    print(f"\nTest Results:")
    print(f"  Acc@1:  {test_perf['acc@1']:.2f}%")
    print(f"  Acc@5:  {test_perf['acc@5']:.2f}%")
    print(f"  Acc@10: {test_perf['acc@10']:.2f}%")
    print(f"  MRR:    {test_perf['mrr']:.2f}%")
    print(f"  NDCG:   {test_perf['ndcg']:.2f}%")
    print("="*80)
    
    # Save results
    results = {
        'model': 'UserAwareTransformer',
        'params': params,
        'best_val_acc@1': best_val_acc,
        'best_epoch': best_epoch,
        'test': test_perf,
        'history': history
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Check target
    if test_perf['acc@1'] >= 50.0:
        print(f"\n🎉 SUCCESS! Test Acc@1 = {test_perf['acc@1']:.2f}% ≥ 50%")
    else:
        print(f"\n⚠️  Test Acc@1 = {test_perf['acc@1']:.2f}% < 50% (gap: {50 - test_perf['acc@1']:.2f}%)")
        print(f"Need to improve model/training...")


if __name__ == '__main__':
    main()
