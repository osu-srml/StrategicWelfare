import torch
import numpy as np
from torch.autograd import Variable



# setting 1: the agents have certain effort budget and will move towards the direction of gradient at point x
def Grad_effort(model,h, x, delta, ctv=None):
    x.requires_grad = True
    Yhat = model(x)
    loss = torch.nn.BCELoss(reduction='sum')
    cost = loss(Yhat.squeeze(),torch.ones(Yhat.squeeze().shape))
    model.zero_grad()
    coef, = torch.autograd.grad(cost, x, create_graph=True)
    efforts = (-delta)*coef
    Yhat = model(x+efforts)
    Yhat_total = h(x+efforts*ctv)
    return Yhat, Yhat_total



