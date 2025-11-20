import torch
from scipy.stats import pearsonr, spearmanr
import torch.distributed as dist
import numpy as np

def get_metrics(predicted : torch.Tensor, expected : torch.Tensor):
    predicted = predicted.flatten()
    expected = expected.flatten()

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    MAE = torch.nn.L1Loss()(predicted, expected)

    # collect distributed results
    if dist.is_initialized():
        dist.all_reduce(MAE, op=dist.ReduceOp.SUM)
        MAE = (MAE / world_size).item()
    else:
        MAE = MAE.item()

    predicted_np = predicted.detach().cpu().numpy()
    expected_np = expected.detach().cpu().numpy()

    if dist.is_initialized():
        if rank == 0:
            gathered_preds = [None for _ in range(world_size)]
            gathered_labels = [None for _ in range(world_size)]
        else:
            gathered_preds = None
            gathered_labels = None

        # gather everything onto rank 0
        dist.gather_object(predicted_np, gathered_preds, dst=0)
        dist.gather_object(expected_np, gathered_labels, dst=0)

        if rank == 0:
            # get stats
            all_preds = np.concatenate(gathered_preds, axis=0)
            all_labels = np.concatenate(gathered_labels, axis=0)

            spearman = spearmanr(all_preds, all_labels).statistic
            pearson = pearsonr(all_preds, all_labels).statistic
        else:
            spearman = pearson = None
    else:
        spearman = spearmanr(predicted_np, expected_np).statistic
        pearson = pearsonr(predicted_np, expected_np).statistic

    return MAE, spearman, pearson