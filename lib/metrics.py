import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, precision_score, recall_score

def get_metrics(predicted : torch.Tensor, expected : torch.Tensor):
    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1

    CE = torch.nn.CrossEntropyLoss()(predicted, expected.flatten())
    
    # logits to probabilities
    predicted = predicted.softmax(dim=1)

    expected = expected.flatten()
    predicted = predicted.argmax(dim=1).flatten()
    
    # multi-class confusion matrix
    accuracy = accuracy_score(expected, predicted)
    precision = precision_score(expected, predicted, average="macro", zero_division=0)
    recall = recall_score(expected, predicted, average="macro", zero_division=0)

    # for DDP
    CE = ddp_reduce_tensor(CE, world_size)
    accuracy = ddp_reduce_tensor(torch.tensor(accuracy), world_size)
    precision = ddp_reduce_tensor(torch.tensor(precision), world_size)
    recall = ddp_reduce_tensor(torch.tensor(recall), world_size)

    return CE, accuracy, precision, recall

def ddp_reduce_tensor(metric, world_size):
    if dist.is_initialized():
        reduced = metric.clone()
        if torch.cuda.is_available():
            reduced = reduced.cuda() # tensor operations must be done on cuda

        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        metric = (reduced / world_size).item()
    else:
        metric = metric.item()

    return metric