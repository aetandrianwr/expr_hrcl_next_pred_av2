"""
AGGRESSIVE TRAINING - Every proven technique to reach 50%
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import random
import json
from tqdm import tqdm

from model_transition import TransitionAwarePredictor
from dataset import get_dataloaders
from metrics import calculate_correct_total_prediction, get_performance_dict


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MixUpLoss(nn.Module):
    """Mixup augmentation for better generalization"""
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    def forward(self, logits, targets, training=True):
        if not training or self.alpha <= 0:
            return self.criterion(logits, targets)
        
        # Mixup
        batch_size = targets.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        index = torch.randperm(batch_size, device=targets.device)
        
        mixed_logits = lam * logits + (1 - lam) * logits[index]
        loss = lam * self.criterion(mixed_logits, targets) + \
               (1 - lam) * self.criterion(mixed_logits, targets[index])
        
        return loss


def train_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch['Y'], training=True)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{total_loss / (pbar.n + 1):.4f}'})
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, name='Val'):
    model.eval()
    total_loss = 0
    metrics_dict = {
        "correct@1": 0, "correct@3": 0, "correct@5": 0, "correct@10": 0,
        "rr": 0, "ndcg": 0, "f1": 0, "total": 0,
    }
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=name, leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            logits = model(batch)
            loss = criterion(logits, batch['Y'], training=False)
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
    
    model = TransitionAwarePredictor(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*80)
    print(f"TransitionAwarePredictor - {num_params:,} parameters")
    print("="*80)
    
    # Larger batch for better gradient estimates
    train_loader, val_loader, test_loader = get_dataloaders(
        '/content/expr_hrcl_next_pred_av2/data/geolife', batch_size=512, max_len=60
    )
    
    # MixUp + Label Smoothing
    criterion = MixUpLoss(alpha=0.2)
    
    # AdamW with proper weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.005,  # Higher LR
        betas=(0.9, 0.999),
        weight_decay=0.02,  # Stronger regularization
        eps=1e-8
    )
    
    # OneCycleLR for fast convergence
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * 100
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.005,
        total_steps=total_steps,
        pct_start=0.05,  # Quick warmup
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )
    
    best_val_acc = 0
    best_epoch = 0
    patience = 0
    max_patience = 20
    
    history = []
    
    for epoch in range(1, 101):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        val_loss, val_perf = evaluate(model, val_loader, criterion, device, 'Val')
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc@1': val_perf['acc@1']
        })
        
        print(f"\nEpoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Acc@1: {val_perf['acc@1']:.2f}% | Acc@5: {val_perf['acc@5']:.2f}% | "
              f"MRR: {val_perf['mrr']:.2f}%")
        
        if val_perf['acc@1'] > best_val_acc:
            best_val_acc = val_perf['acc@1']
            best_epoch = epoch
            patience = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': best_val_acc,
                'config': config,
            }, 'best_model.pt')
            print(f"  🎯 NEW BEST: {best_val_acc:.2f}%")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"\n⏹  Early stop at epoch {epoch}")
                break
    
    # Final test
    print("\n" + "="*80)
    print("FINAL TEST")
    print("="*80)
    
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_perf = evaluate(model, test_loader, criterion, device, 'Test')
    
    print(f"\nTest Acc@1:  {test_perf['acc@1']:.2f}%")
    print(f"Test Acc@5:  {test_perf['acc@5']:.2f}%")
    print(f"Test Acc@10: {test_perf['acc@10']:.2f}%")
    print(f"Test MRR:    {test_perf['mrr']:.2f}%")
    print(f"Test NDCG:   {test_perf['ndcg']:.2f}%")
    print("="*80)
    
    results = {
        'test': test_perf,
        'best_val': best_val_acc,
        'best_epoch': best_epoch,
        'params': num_params,
        'history': history
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    if test_perf['acc@1'] >= 50.0:
        print(f"\n🎉🎉🎉 SUCCESS: {test_perf['acc@1']:.2f}% >= 50%")
    elif test_perf['acc@1'] >= 45.0:
        print(f"\n⚠️  CLOSE: {test_perf['acc@1']:.2f}% (need 50%)")
    else:
        print(f"\n❌ Current: {test_perf['acc@1']:.2f}% (need 50%)")
        print("Continuing experiments...")


if __name__ == '__main__':
    main()
