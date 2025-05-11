import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import optim
from pathlib import Path
from tqdm import tqdm
from data_loader import loaders
from models.autoencoder import LinearAutoencoder, TopologicalAutoencoder

import json

from train_model import train_model
from evaluate_metrics import evaluate_metrics

@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: DictConfig) -> None:

    model = LinearAutoencoder(
                            #input_dim=mice_data.dimension,
                            # layers_dims=[500, 350, 100],
                            # activation='relu'
                            **cfg.arch_configs
    )
    topo_model = TopologicalAutoencoder(
                                    model, **cfg.model_configs)
                                    # complex_max_dim=1,
                                    # min_persistence=1,
                                    # top_loss_dims=[0, 1], 
                                    # dims_weights=[1, 0.3],
                                    # lam=0.3)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Available device:', device)
    
    topo_model = topo_model.to(device)
    optimizer = optim.Adam(topo_model.parameters(), lr=cfg.training_configs.learning_rate)

    train_loader, val_loader, test_loader = loaders(**cfg.training_configs.data_loader)

    
    n_epochs = cfg.training_configs.epochs
    topo_model = train_model(topo_model, optimizer, train_loader, val_loader, test_loader, device, n_epochs=n_epochs)
    metrics_values = evaluate_metrics(topo_model, test_loader, device)

    report_file = 'test_reports/' + cfg.experiment_name + '.txt'
    with open(report_file, 'w') as f:
        f.write("\n".join([metric_name + ' = ' + str(metrics_values[metric_name]) for metric_name in metrics_values.keys()]))
    print('Metrics saved to', report_file)
    # Training loop would go here
    # print(f"Starting experiment: {cfg.experiment_name}")
    # print(f"Model architecture: {cfg.model_name}")
    # print(f"Training hyperparameters: {cfg.training}")
    # print(f"Data augmentation: {cfg.data.augmentation}")

if __name__ == "__main__":
    main()