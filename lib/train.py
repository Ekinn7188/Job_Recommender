import torch
import torch.nn as nn
from .metrics import get_metrics
from tqdm import tqdm

def train_one_epoch(dataloader : torch.utils.data.DataLoader, 
                    model : type[nn.Module],
                    criterion : type[nn.Module], 
                    opt : type[torch.optim.Optimizer],
                    device : torch.device,
                    debug: bool = True):
    """
    Train the model for one epoch.

    Parameters
    ----------
    dataloader : torch.utils.data.Dataloader
        The loaded and pre-configured dataset to iterate over.
    model : nn.Module
        The model to train with.
    criterion : nn.Module
        The loss function.
    opt : torch.optim.Optimizer
        The optimizer to update the model's weights.
    device : torch.device
        The device to put the tensors on.

    Returns
    -------
    TBD
    """

    model.train()

    predicted_y = []
    true_y = []

    for batch_idx, (resumes, resumes_attention_masks, descriptions, descriptions_attention_masks, labels) in enumerate(tqdm(dataloader)):
        opt.zero_grad()

        # debug: capture some param stats before forward on the first batch
        if debug and batch_idx == 0:
            params_before = [p.data.mean().item() for p in list(model.parameters())[:3]]
            requires_grad_before = [p.requires_grad for p in list(model.parameters())[:3]]

        predicted = model(resumes.to(device), resumes_attention_masks.to(device), descriptions.to(device), descriptions_attention_masks.to(device))

        l = criterion(predicted, labels.flatten().to(device))

        l.backward()

        if debug and batch_idx == 0:
            # compute a simple aggregate of gradient norms for a quick sanity check
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    try:
                        total_grad_norm += float(p.grad.detach().norm().item())
                    except Exception:
                        # guard in case of sparse or unusual grads
                        pass

            # print a few diagnostics
            try:
                sample_preds = predicted[:5].cpu().detach().numpy()
            except Exception:
                sample_preds = None
            try:
                sample_labels = labels[:5].cpu().detach().numpy()
            except Exception:
                sample_labels = None

            print(f"[debug] batch0 loss={l.item():.6f} grad_norm={total_grad_norm:.6f} sample_preds={sample_preds} sample_labels={sample_labels}")

        opt.step()

        if debug and batch_idx == 0:
            params_after = [p.data.mean().item() for p in list(model.parameters())[:3]]
            requires_grad = [p.requires_grad for p in list(model.parameters())[:3]]
            print(f"[debug] params_before={params_before} params_after={params_after}")
            print(f"[debug] params_before={requires_grad_before} params_after={requires_grad}")

        # ensure we don't keep CUDA tensors in memory
        l.cpu().detach()

        predicted_y.append(predicted.cpu().detach())
        true_y.append(labels.cpu().detach())
    
    predicted_y = torch.cat(predicted_y)
    true_y = torch.cat(true_y)

    avg_MAE, spearman_coeff, pearson_coeff = get_metrics(predicted_y, true_y)

    return avg_MAE, spearman_coeff, pearson_coeff

def validate(dataloader : torch.utils.data.DataLoader, 
             model : type[nn.Module],
             device : torch.device):
    """
    Validate the model's mid-epoch progress.

    Parameters
    ----------
    dataloader : torch.utils.data.Dataloader
        The loaded and pre-configured dataset to iterate over.
    model : nn.Module
        The model to validate with.
    criterion : nn.Module
        The loss function.
    device : torch.device
        The device to put the tensors on.

    Returns
    -------
    TBD
    """

    model.eval()

    predicted_y = []
    true_y = []

    with torch.no_grad():
        for (resumes, resumes_attention_masks, descriptions, descriptions_attention_masks, labels) in tqdm(dataloader):
            predicted = model(
                resumes.to(device),
                resumes_attention_masks.to(device),
                descriptions.to(device),
                descriptions_attention_masks.to(device),
            )

            predicted_y.append(predicted.cpu().detach())
            true_y.append(labels.cpu().detach())
    
    predicted_y = torch.cat(predicted_y)
    true_y = torch.cat(true_y)

    avg_MAE, spearman_coeff, pearson_coeff = get_metrics(predicted_y, true_y)

    return avg_MAE, spearman_coeff, pearson_coeff

def test(dataloader : torch.utils.data.DataLoader, 
         model : type[nn.Module],
         device : torch.device):
    """
    Test the model's full progress.

    Parameters
    ----------
    dataloader : torch.utils.data.Dataloader
        The loaded and pre-configured dataset to iterate over.
    model : nn.Module
        The model to test with.
    device : torch.device
        The device to put the tensors on.

    Returns
    -------
    TBD
    """
    model.eval()

    predicted_y = []
    true_y = []


    with torch.no_grad():
        for (resumes, resumes_attention_masks, descriptions, descriptions_attention_masks, labels) in tqdm(dataloader):
            predicted = model(
                resumes.to(device),
                resumes_attention_masks.to(device),
                descriptions.to(device),
                descriptions_attention_masks.to(device),
            )

            predicted_y.append(predicted.cpu().detach())
            true_y.append(labels.cpu().detach())
    
    predicted_y = torch.cat(predicted_y)
    true_y = torch.cat(true_y)

    avg_MAE, spearman_coeff, pearson_coeff = get_metrics(predicted_y, true_y)

    return avg_MAE, spearman_coeff, pearson_coeff
