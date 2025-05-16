import argparse
import torch
from torch import optim
from pathlib import Path
from tqdm import tqdm
from data_loader import loaders
from models.autoencoder import LinearAutoencoder, TopologicalAutoencoder, DistancesTopologicalAutoencoder

from train_model import train_model
from evaluate_metrics import evaluate_metrics

import json

def main():
    parser = argparse.ArgumentParser(description='Process a JSON configuration file.')
    parser.add_argument('config_path', help='Path to the JSON configuration file')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
            cfg = json.load(f)

    print(f"Running {cfg['experiment_name']}")
    print(cfg)
    model = LinearAutoencoder(
        input_dim=cfg['arch_config']['input_dim'],
        layers_dims=cfg['arch_config']['layers_dims'],
        activation=cfg['arch_config']['activation']
    )
    assert cfg['distance_metric'] in ['euclidean', 'correlation'], 'Invalid distance metric!'
    if cfg['distance_metric'] == 'euclidean':
        print('euclidean metric is used')
        topo_model = TopologicalAutoencoder(
                        model,
                        complex_max_dim=cfg['model_config']['complex_max_dim'],
                        min_persistence=cfg['model_config']['min_persistence'],
                        top_loss_dims=cfg['model_config']['top_loss_dims'],
                        dims_weights=cfg['model_config']['dims_weights'],
                        lam=cfg['model_config']['lam']
        )
    if cfg['distance_metric'] == 'correlation':
        print('Correlatoin metric is used')
        topo_model = DistancesTopologicalAutoencoder(
                        model,
                        complex_max_dim=cfg['model_config']['complex_max_dim'],
                        #min_persistence=cfg['model_config']['min_persistence'],
                        top_loss_dims=cfg['model_config']['top_loss_dims'],
                        #dims_weights=cfg['model_config']['dims_weights'],
                        lam=cfg['model_config']['lam']
        )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Available device:', device)
    
    topo_model = topo_model.to(device)

    optimizer = optim.Adam(topo_model.parameters(), lr=cfg['training_config']['learning_rate'])
    train_loader, val_loader, test_loader = loaders(
        data_path=cfg['training_config']['data_path'],
        train_size=cfg['training_config']['train_size'],
        val_size=cfg['training_config']['val_size'],
        batch_size=cfg['training_config']['batch_size'],
        data_seed=cfg['training_config']['data_seed']
    )
    n_epochs = cfg['training_config']['epochs']
    
    topo_model = train_model(topo_model, optimizer, train_loader, val_loader, test_loader, device, n_epochs=n_epochs)
    metrics_values = evaluate_metrics(topo_model, test_loader, device)

    report_file = 'test_reports/' + cfg['experiment_name'] + '.txt'
    with open(report_file, 'w') as f:
        f.write("\n".join([metric_name + ' = ' + str(metrics_values[metric_name]) for metric_name in metrics_values.keys()]))
    print('Metrics saved to', report_file)

if __name__ == "__main__":
    main()