import json

with open('dataset_config.json', 'r') as f:
    dataset_cfg = json.load(f)

CONFIG = {
    'data_path': '/content/expr_hrcl_next_pred_av2/data/geolife',
    'batch_size': 256,
    'max_seq_len': 60,
    
    'n_locations': dataset_cfg['n_locations'],
    'n_users': dataset_cfg['n_users'],
    'n_s2_l11': dataset_cfg['n_s2_l11'],
    'n_s2_l13': dataset_cfg['n_s2_l13'],
    'n_s2_l14': dataset_cfg['n_s2_l14'],
    'n_s2_l15': dataset_cfg['n_s2_l15'],
    'n_weekdays': 8,
    
    'd_model': 48,  # 466K params
    'n_heads': 6,   # 48 % 6 = 0
    'n_layers': 3,
    'dropout': 0.15,
    'max_pe': 100,
    
    'epochs': 80,
    'lr': 0.003,
    'weight_decay': 0.01,
    'warmup_epochs': 5,
    'patience': 20,
    'grad_clip': 1.0,
    'seed': 42,
}
