from metrics import knn_neighborhood_preservation,\
                    spearman_distance_correlation, \
                    procrustes_error, \
                    compute_trustworthiness, \
                    mean_relative_rank_error, \
                    estimate_kl_divergence_kde, \
                    calculate_top_loss

metrics_list = [knn_neighborhood_preservation,\
                spearman_distance_correlation, \
                procrustes_error, \
                compute_trustworthiness, \
                mean_relative_rank_error, \
                estimate_kl_divergence_kde, \
                calculate_top_loss
]
metrics_names = ['knn_neighborhood_preservation',\
                'spearman_distance_correlation', \
                'procrustes_error', \
                'trustworthiness', \
                'mean_relative_rank_error', \
                'kl-divergence', \
                'top_loss'
]

def evaluate_metrics(model, test_loader, device):
    
    model.eval()
    values = {}
    for metric, name in zip(metrics_list, metrics_names):
        print(f'Evaluating {name}')
        metric_value = 0
        for activation_map, coords in test_loader:
            latent_map = model.model.encode(activation_map.to(device)).cpu().detach().numpy()
            if name != 'top_loss':
                metric_value += metric(coords, latent_map)
            else:
                metric_value += metric(coords, latent_map, max_dim=0, loss_dims=[0])
    
        values[name] = round(metric_value / len(test_loader), 3)
    return values