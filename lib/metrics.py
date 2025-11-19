import torch
from scipy.stats import pearsonr, spearmanr

def get_metrics(predicted : torch.Tensor, expected : torch.Tensor):
    predicted = predicted.flatten()
    expected = expected.flatten()

    MAE = torch.nn.L1Loss()(predicted, expected)

    predicted_np = predicted.detach().cpu().numpy()
    expected_np = expected.detach().cpu().numpy()

    spearman = spearmanr(predicted_np, expected_np).statistic
    pearson = pearsonr(predicted_np, expected_np).statistic

    return MAE, spearman, pearson