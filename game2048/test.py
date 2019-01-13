import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable


def test(epoch):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        with torch.no_grad():
            data = Variable(data).cuda()
            target =Variable(target).cuda()
        data = data.unsqueeze(dim=1)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))
