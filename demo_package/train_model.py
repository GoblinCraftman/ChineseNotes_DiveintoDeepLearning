
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import optimizer


def fit(epoch,model,data_loader,phase='Training',volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'volidation':
        model.val()
        volatile=True

    net=Net()
    if torch.cuda.is_available():
        net = net.cuda()
    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
    learnrate = 0.01
    optimizer = torch.optim.SGD(net.parameters(), lr=learnrate)
    running_loss=0
    running_currecr=0

    for batch_idx,(data,target) in enumerate(data_loader):
        data,target=data.cuda(),target.cuda()
        data,target=Variable(data,volatile),Variable(target)

        if phase == 'training':
            optimizer.zero_gard()
        output = model(data)
        loss = F.nll_loss(output,target)