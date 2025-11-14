import torch
import torch.nn as nn

def train_one_epoch(dataloader : torch.utils.data.DataLoader, 
                    model : type[nn.Module],
                    criterion : type[nn.Module], 
                    opt : type[torch.optim.Optimizer]):
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
