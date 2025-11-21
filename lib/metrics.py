import torch
import torch.distributed as dist
from sklearn.metrics import multilabel_confusion_matrix

def get_metrics(predicted : torch.Tensor, expected : torch.Tensor):
    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1

    CE = torch.nn.CrossEntropyLoss()(predicted, expected.flatten())

    # multi-class confusion matrix
    matrix = multilabel_confusion_matrix(y_true=expected.flatten(), y_pred=predicted.argmax(dim=1).flatten())
    tn, fn, tp, fp = matrix[:,0,0].sum(), matrix[:,1,0].sum(), matrix[:,1,1].sum(), matrix[:,0,1].sum()

    accuracy = (tn + tp) / (tn + fn + tp + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

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