"""
Train Geography-Aware Hierarchical Transformer
Based on proven research techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import random
import json
from tqdm import tqdm

from model_geosan import GeographyAwareHierarchicalTransformer
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
    
    for batch in tqdm(loader, desc='Train', leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch['Y'])
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, name='Val'):
    model.eval()
    total_loss = 0
    m_dict = {"correct@1": 0, "correct@3": 0, "correct@5": 0, "correct@10": 0,
              "rr": 0, "ndcg": 0, "f1": 0, "total": 0}
    
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
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("Geography-Aware Hierarchical Transformer")
    print("Based on: GeoSAN (AAAI 2020) + LSTPM (KDD 2020)")
    print("="*80)
    
    config = {
        'd_model': 96,
        'num_locations': 1200,
        'num_users': 50,
        'num_s2_l11': 320,
        'num_s2_l12': 680,
        'num_s2_l13': 930,
        'num_s2_l14': 1260,
    }
    
    model = GeographyAwareHierarchicalTransformer(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters: {num_params:,}")
    print(f"Within budget: {num_params < 1_000_000}\n")
    
    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(
        '/content/expr_hrcl_next_pred_av2/data/geolife',
        batch_size=128,  # Moderate batch size
        max_len=60
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.002,
        betas=(0.9, 0.98),  # Standard for Transformers
        weight_decay=0.01,
        eps=1e-9
    )
    
    # Cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    best_val_acc = 0
    best_epoch = 0
    patience = 0
    max_patience = 25
    
    history = []
    
    print("Starting training...\n")
    
    for epoch in range(1, 101):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_perf = evaluate(model, val_loader, criterion, device, 'Val')
        scheduler.step()
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc@1': val_perf['acc@1'],
            'val_acc@5': val_perf['acc@5'],
            'val_mrr': val_perf['mrr']
        })
        
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"           | Acc@1: {val_perf['acc@1']:6.2f}% | Acc@5: {val_perf['acc@5']:6.2f}% | "
              f"MRR: {val_perf['mrr']:6.2f}%")
        
        if val_perf['acc@1'] > best_val_acc:
            best_val_acc = val_perf['acc@1']
            best_epoch = epoch
            patience = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': config,
            }, 'best_model.pt')
            
            print(f"           | ✓ NEW BEST: {best_val_acc:.2f}%")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        print()
    
    # Final test evaluation
    print("="*80)
    print("FINAL TEST EVALUATION")
    print("="*80)
    
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_perf = evaluate(model, test_loader, criterion, device, 'Test')
    
    print(f"\nBest validation epoch: {best_epoch}")
    print(f"Best validation Acc@1: {best_val_acc:.2f}%\n")
    
    print("Test Results:")
    print(f"  Acc@1:  {test_perf['acc@1']:.2f}%")
    print(f"  Acc@5:  {test_perf['acc@5']:.2f}%")
    print(f"  Acc@10: {test_perf['acc@10']:.2f}%")
    print(f"  MRR:    {test_perf['mrr']:.2f}%")
    print(f"  NDCG:   {test_perf['ndcg']:.2f}%")
    print("="*80)
    
    results = {
        'model': 'GeographyAwareHierarchicalTransformer',
        'params': num_params,
        'best_val_acc@1': best_val_acc,
        'best_epoch': best_epoch,
        'test': test_perf,
        'history': history
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    if test_perf['acc@1'] >= 50.0:
        print(f"\n🎉🎉🎉 SUCCESS! Achieved {test_perf['acc@1']:.2f}% >= 50%")
    elif test_perf['acc@1'] >= 45.0:
        print(f"\n⚠️  Very close: {test_perf['acc@1']:.2f}% (need 50%)")
    else:
        print(f"\n📊 Current: {test_perf['acc@1']:.2f}% (target: 50%)")
        print(f"   Gap: {50.0 - test_perf['acc@1']:.2f}%")


if __name__ == '__main__':
    main()
