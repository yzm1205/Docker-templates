
import torch
def get_accuracy(preds,labels,batch_size):
    correct = (torch.max(preds,1)[1].view(labels.size()).data == labels.data).sum()
    accuracy = 100.0 * correct/batch_size
    return accuracy.item()