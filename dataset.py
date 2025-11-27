import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle


class GeoLifeDataset(Dataset):
    def __init__(self, data_path, max_len=60):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Get sequence length
        seq_len = len(sample['X'])
        
        # Truncate if necessary
        if seq_len > self.max_len:
            seq_len = self.max_len
        
        # Create item dict
        item = {
            'X': sample['X'][:seq_len],
            'user_X': sample['user_X'][:seq_len],
            'weekday_X': sample['weekday_X'][:seq_len],
            's2_level11_X': sample['s2_level11_X'][:seq_len],
            's2_level13_X': sample['s2_level13_X'][:seq_len],
            's2_level14_X': sample['s2_level14_X'][:seq_len],
            's2_level15_X': sample['s2_level15_X'][:seq_len],
            'Y': sample['Y'],
            'seq_len': seq_len,
        }
        
        return item


def collate_fn(batch):
    """Collate function to pad sequences to same length in batch."""
    # Find max length in batch
    max_len = max(item['seq_len'] for item in batch)
    
    batch_size = len(batch)
    
    # Initialize tensors
    X = torch.zeros(batch_size, max_len, dtype=torch.long)
    user_X = torch.zeros(batch_size, max_len, dtype=torch.long)
    weekday_X = torch.zeros(batch_size, max_len, dtype=torch.long)
    s2_level11_X = torch.zeros(batch_size, max_len, dtype=torch.long)
    s2_level13_X = torch.zeros(batch_size, max_len, dtype=torch.long)
    s2_level14_X = torch.zeros(batch_size, max_len, dtype=torch.long)
    s2_level15_X = torch.zeros(batch_size, max_len, dtype=torch.long)
    Y = torch.zeros(batch_size, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        X[i, :seq_len] = torch.tensor(item['X'], dtype=torch.long)
        user_X[i, :seq_len] = torch.tensor(item['user_X'], dtype=torch.long)
        weekday_X[i, :seq_len] = torch.tensor(item['weekday_X'], dtype=torch.long)
        s2_level11_X[i, :seq_len] = torch.tensor(item['s2_level11_X'], dtype=torch.long)
        s2_level13_X[i, :seq_len] = torch.tensor(item['s2_level13_X'], dtype=torch.long)
        s2_level14_X[i, :seq_len] = torch.tensor(item['s2_level14_X'], dtype=torch.long)
        s2_level15_X[i, :seq_len] = torch.tensor(item['s2_level15_X'], dtype=torch.long)
        Y[i] = item['Y']
        mask[i, :seq_len] = True
    
    return {
        'X': X,
        'user_X': user_X,
        'weekday_X': weekday_X,
        's2_level11_X': s2_level11_X,
        's2_level13_X': s2_level13_X,
        's2_level14_X': s2_level14_X,
        's2_level15_X': s2_level15_X,
        'Y': Y,
        'mask': mask,
    }


def get_dataloaders(data_dir, batch_size=64, max_len=60):
    """Create train, validation, and test dataloaders."""
    train_dataset = GeoLifeDataset(f'{data_dir}/geolife_transformer_7_train.pk', max_len=max_len)
    val_dataset = GeoLifeDataset(f'{data_dir}/geolife_transformer_7_validation.pk', max_len=max_len)
    test_dataset = GeoLifeDataset(f'{data_dir}/geolife_transformer_7_test.pk', max_len=max_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
