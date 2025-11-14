import torch
import torch.nn as nn

def train_one_epoch(dataloader : torch.utils.data.DataLoader, 
                    model : type[nn.Module],
                    criterion : type[nn.Module], 
                    opt : type[torch.optim.Optimizer]):
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

    Returns
    -------
    TBD
    """

    model.train()

    total_loss = 0

    predicted_y = []
    true_y = []

    for (x, y) in dataloader:
        opt.zero_grad()

        predicted = model(x)

        l = criterion(x, y)

        l.backward()
        opt.step()

        total_loss += l.cpu().detach() * x.shape[0]

        predicted_y.append(predicted.cpu().detach())
        true_y.append(y.cpu().detach())
    
    predicted_y = torch.cat(predicted_y)
    true_y = torch.cat(true_y)

    avg_loss = total_loss / len(dataloader.dataset)

    # TODO more metrics depending on what we do

    return avg_loss

def validate(dataloader : torch.utils.data.DataLoader, 
             model : type[nn.Module],
             criterion : type[nn.Module]):
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

    Returns
    -------
    TBD
    """

    model.eval()

    total_loss = 0

    predicted_y = []
    true_y = []

    for (x, y) in dataloader:
        with torch.no_grad():
            predicted = model(x)

        l = criterion(x, y)

        total_loss += l.cpu().detach() * x.shape[0]

        predicted_y.append(predicted.cpu().detach())
        true_y.append(y.cpu().detach())
    
    predicted_y = torch.cat(predicted_y)
    true_y = torch.cat(true_y)

    avg_loss = total_loss / len(dataloader.dataset)

    # TODO more metrics depending on what we do

    return avg_loss 

def test(dataloader : torch.utils.data.DataLoader, 
         model : type[nn.Module]):
    """
    Test the model's full progress.

    Parameters
    ----------
    dataloader : torch.utils.data.Dataloader
        The loaded and pre-configured dataset to iterate over.
    model : nn.Module
        The model to test with.

    Returns
    -------
    TBD
    """
    model.eval()

    total_loss = 0

    predicted_y = []
    true_y = []

    for (x, y) in dataloader:
        with torch.no_grad():
            predicted = model(x)

        predicted_y.append(predicted.cpu().detach())
        true_y.append(y.cpu().detach())
    
    predicted_y = torch.cat(predicted_y)
    true_y = torch.cat(true_y)

    # TODO calculate some metrics depending on what we do

    return # some metrics
