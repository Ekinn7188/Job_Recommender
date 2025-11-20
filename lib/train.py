import torch
import torch.nn as nn
from .metrics import get_metrics
from tqdm import tqdm

def train_one_epoch(dataloader : torch.utils.data.DataLoader, 
                    model : type[nn.Module],
                    criterion : type[nn.Module], 
                    opt : type[torch.optim.Optimizer],
                    device : torch.device):
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

    for (resumes, resumes_attention_masks, descriptions, descriptions_attention_masks, labels) in tqdm(dataloader):
        opt.zero_grad()

        predicted = model(resumes.to(device), resumes_attention_masks.to(device), descriptions.to(device), descriptions_attention_masks.to(device))

        l = criterion(predicted, labels.flatten().to(device))

        l.backward()
        opt.step()
        l.cpu().detach()

        predicted_y.append(predicted.cpu().detach())
        true_y.append(labels.cpu().detach())
    
    predicted_y = torch.cat(predicted_y)
    true_y = torch.cat(true_y)

    CE, accuracy, precision, recall = get_metrics(predicted_y, true_y)

    return CE, accuracy, precision, recall

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

    CE, accuracy, precision, recall = get_metrics(predicted_y, true_y)

    return CE, accuracy, precision, recall

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

    CE, accuracy, precision, recall = get_metrics(predicted_y, true_y)

    return CE, accuracy, precision, recall
