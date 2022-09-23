import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.colors import LightSource
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#import ipdb
import random



def fgsm(model, X, y, epsilon=0.1):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

def norms_l2(Z):
    return norms(Z)

def norms_l2_squeezed(Z):
    return norms(Z).squeeze(1).squeeze(1).squeeze(1)

def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None]

def norms_l1_squeezed(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None].squeeze(1).squeeze(1).squeeze(1)

def norms_l0(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float()

def norms_l0_squeezed(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float().squeeze(1).squeeze(1).squeeze(1)

def norms_linf(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0]


def pgd_linf(model, X, y, epsilon=0.3, alpha=0.01, num_iter = 50, randomize = 0, restarts = 0, device = "cuda:0"):
    """ Construct FGSM adversarial examples on the examples X"""
    # ipdb.set_trace()
   
    max_delta = torch.zeros_like(X)
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = (delta.data * 2.0 - 1.0) * epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)    
    for t in range(num_iter):
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        #Finding the correct examples so as to attack only them
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta.data + alpha*correct*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        delta.grad.zero_()
    max_delta = delta.detach()
    
    for i in range (restarts):
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = (delta.data * 2.0 - 1.0) * epsilon

        for t in range(num_iter):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            #Finding the correct examples so as to attack only them            
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss.backward()
            delta.data = (delta.data + alpha*correct*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
            delta.grad.zero_()

        output = model(X+delta)
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks        
        max_delta[incorrect] = delta.detach()[incorrect]
    return max_delta



def pgd_l2(model, X, y, epsilon=2.0, alpha=0.1, num_iter = 100, restarts = 0, device = "cuda:0", randomize = 0):
    # ipdb.set_trace()
    max_delta = torch.zeros_like(X)
    if random:
        delta = torch.rand_like(X, requires_grad=True) 
        delta.data *= (2.0*delta.data - 1.0)
        delta.data = delta.data*epsilon/norms_l2(delta.detach()) 
    else:
        delta = torch.zeros_like(X, requires_grad=True) 
    for t in range(num_iter):
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        #Finding the correct examples so as to attack only them
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data +=  correct*alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data *=  epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.data =   torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]     
        delta.grad.zero_()  

    max_delta = delta.detach()

    #restarts

    for k in range (restarts):
        delta = torch.rand_like(X, requires_grad=True) 
        delta.data *= (2.0*delta.data - 1.0)*epsilon 
        delta.data /= norms(delta.detach()).clamp(min=epsilon)
        for t in range(num_iter):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            #Finding the correct examples so as to attack only them
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss.backward()
            delta.data +=  correct*alpha*delta.grad.detach() / norms(delta.grad.detach())
            delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
            delta.grad.zero_()  

        output = model(X+delta)
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect] 

    return max_delta    

def pgd_l0(model, X,y, epsilon = 10, alpha = 0.5, num_iter = 100, device = "cuda:1"):
    delta = torch.zeros_like(X, requires_grad = True)
    batch_size = X.shape[0]
    for t in range (epsilon):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        temp = delta.grad.view(batch_size, 1, -1)
        neg = (delta.data != 0)
        X_curr = X + delta
        neg1 = (delta.grad < 0)*(X_curr < 0.1)
        neg2 = (delta.grad > 0)*(X_curr > 0.9)
        neg += neg1 + neg2
        u = neg.view(batch_size,1,-1)
        temp[u] = 0
        my_delta = torch.zeros_like(X).view(batch_size, 1, -1)
        
        maxv =  temp.max(dim = 2)
        minv =  temp.min(dim = 2)
        val_max = maxv[0].view(batch_size)
        val_min = minv[0].view(batch_size)
        pos_max = maxv[1].view(batch_size)
        pos_min = minv[1].view(batch_size)
        select_max = (val_max.abs()>=val_min.abs()).float()
        select_min = (val_max.abs()<val_min.abs()).float()
        my_delta[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_max] = (1-X.view(batch_size, 1, -1)[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_max])*select_max
        my_delta[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_min] = -X.view(batch_size, 1, -1)[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_min]*select_min
        delta.data += my_delta.view(batch_size, 1, 28, 28)
        delta.grad.zero_()
    delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
    
    return delta.detach()


def pgd_l1_topk(model, X,y, epsilon = 10, alpha = 1.0, num_iter = 50, k_map = 0, gap = 0.05, device = "cuda:0", restarts = 0, randomize = 0):
    #Gap : Dont attack pixels closer than the gap value to 0 or 1
    # ipdb.set_trace()
    gap = gap
    max_delta = torch.zeros_like(X)
    if randomize:
        delta = torch.from_numpy(np.random.laplace(size=X.shape)).float().to(device)
        delta.data = delta.data*epsilon/norms_l1(delta.detach())
        delta.requires_grad = True
    else:
        delta = torch.zeros_like(X, requires_grad = True)
    alpha_l_1_default = alpha

    for t in range (num_iter):
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        #Finding the correct examples so as to attack only them
        loss = nn.CrossEntropyLoss()(model(X+delta), y)
        loss.backward()
        if k_map == 0:
            k = random.randint(5,20)
            alpha   = (alpha_l_1_default/k)
        elif k_map == 1:
            k = random.randint(10,40)
            alpha   = (alpha_l_1_default/k)
        else:
            k = 10
            alpha = alpha_l_1_default
        delta.data += alpha*correct*l1_dir_topk(delta.grad.detach(), delta.data, X, gap,k)
        if (norms_l1(delta) > epsilon).any():
            delta.data = proj_l1ball(delta.data, epsilon, device)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
        delta.grad.zero_() 

    max_delta = delta.detach()

    #Restarts    
    for k in range(restarts):
        delta = torch.rand_like(X,requires_grad = True)
        delta.data = (2*delta.data - 1.0)
        delta.data = delta.data*epsilon/norms_l1(delta.detach())
        for t in range (num_iter):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            #Finding the correct examples so as to attack only them
            loss = nn.CrossEntropyLoss()(model(X+delta), y)
            loss.backward()
            if k_map == 0:
                k = random.randint(5,20)
                alpha   = (alpha_l_1_default/k)
            elif k_map == 1:
                k = random.randint(10,40)
                alpha   = (alpha_l_1_default/k)
            else:
                k = 10
                alpha = alpha_l_1_default
            delta.data += alpha*correct*l1_dir_topk(delta.grad.detach(), delta.data, X, gap,k)
            if (norms_l1(delta) > epsilon).any():
                delta.data = proj_l1ball(delta.data, epsilon, device)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
            delta.grad.zero_() 
        output = model(X+delta)
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect]   

    return max_delta

def kthlargest(tensor, k, dim=-1):
    val, idx = tensor.topk(k, dim = dim)
    return val[:,:,-1], idx[:,:,-1]

def l1_dir_topk(grad, delta, X, gap, k = 10) :
    #Check which all directions can still be increased such that
    #they haven't been clipped already and have scope of increasing
    # ipdb.set_trace()
    X_curr = X + delta
    batch_size = X.shape[0]
    channels = X.shape[1]
    pix = X.shape[2]
    # print (batch_size)
    neg1 = (grad < 0)*(X_curr <= gap)
#     neg1 = (grad < 0)*(X_curr == 0)
    neg2 = (grad > 0)*(X_curr >= 1-gap)
#     neg2 = (grad > 0)*(X_curr == 1)
    neg3 = X_curr <= 0
    neg4 = X_curr >= 1
    neg = neg1 + neg2 + neg3 + neg4
    u = neg.view(batch_size,1,-1)
    grad_check = grad.view(batch_size,1,-1)
    grad_check[u] = 0

    kval = kthlargest(grad_check.abs().float(), k, dim = 2)[0].unsqueeze(1)
    k_hot = (grad_check.abs() >= kval).float() * grad_check.sign()
    return k_hot.view(batch_size, channels, pix, pix)


def proj_l1ball(x, epsilon=10, device = "cuda:1"):
#     print (epsilon)
    # print (device)
    assert epsilon > 0
    # compute the vector of absolute values
    u = x.abs()
    if (u.sum(dim = (1,2,3)) <= epsilon).all():
        # print (u.sum(dim = (1,2,3)))
         # check if x is already a solution
        return x

    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    y = proj_simplex(u, s=epsilon, device = device)
    # compute the solution to the original problem on v
    y *= x.sign()
    y *= epsilon/norms_l1(y)
    return y


def proj_simplex(v, s=1, device = "cuda:1"):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    batch_size = v.shape[0]

    # check if we are already on the simplex    
    '''
    #Not checking this as we are calling this from the previous function only
    if v.sum(dim = (1,2,3)) == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    '''
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = v.view(batch_size,1,-1)
    n = u.shape[2]
    u, indices = torch.sort(u, descending = True)
    cssv = u.cumsum(dim = 2)
    # get the number of > 0 components of the optimal solution
    vec = u * torch.arange(1, n+1).float().to(device)
    comp = (vec > (cssv - s)).float()

    u = comp.cumsum(dim = 2)
    w = (comp-1).cumsum(dim = 2)
    u = u + w
    rho = torch.argmax(u, dim = 2)
    rho = rho.view(batch_size)
    c = torch.FloatTensor([cssv[i,0,rho[i]] for i in range( cssv.shape[0]) ]).to(device)
    c = c-s
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = torch.div(c,(rho.float() + 1))
    theta = theta.view(batch_size,1,1,1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w


# wbn funcs
def pgd_linf_wbn(model, X, y, epsilon=0.3, alpha=0.01, num_iter = 50, randomize = 0, restarts = 0, device = "cuda:0"):
    """ Construct FGSM adversarial examples on the examples X"""
    # ipdb.set_trace()
   
    max_delta = torch.zeros_like(X)
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = (delta.data * 2.0 - 1.0) * epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)    
    for t in range(num_iter):
        output = model(X+delta)[0]
        incorrect = output.max(1)[1] != y 
        correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        #Finding the correct examples so as to attack only them
        loss = nn.CrossEntropyLoss()(model(X + delta)[0], y)
        loss.backward()
        delta.data = (delta.data + alpha*correct*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        delta.grad.zero_()
    max_delta = delta.detach()
    
    for i in range (restarts):
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = (delta.data * 2.0 - 1.0) * epsilon

        for t in range(num_iter):
            output = model(X+delta)[0]
            incorrect = output.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            #Finding the correct examples so as to attack only them            
            loss = nn.CrossEntropyLoss()(model(X + delta)[0], y)
            loss.backward()
            delta.data = (delta.data + alpha*correct*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
            delta.grad.zero_()

        output = model(X+delta)[0]
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks        
        max_delta[incorrect] = delta.detach()[incorrect]
    return max_delta



def pgd_l2_wbn(model, X, y, epsilon=2.0, alpha=0.1, num_iter = 100, restarts = 0, device = "cuda:0", randomize = 0):
    # ipdb.set_trace()
    max_delta = torch.zeros_like(X)
    if random:
        delta = torch.rand_like(X, requires_grad=True) 
        delta.data *= (2.0*delta.data - 1.0)
        delta.data = delta.data*epsilon/norms_l2(delta.detach()) 
    else:
        delta = torch.zeros_like(X, requires_grad=True) 
    for t in range(num_iter):
        output = model(X+delta)[0]
        incorrect = output.max(1)[1] != y 
        correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        #Finding the correct examples so as to attack only them
        loss = nn.CrossEntropyLoss()(model(X + delta)[0], y)
        loss.backward()
        delta.data +=  correct*alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data *=  epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.data =   torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]     
        delta.grad.zero_()  

    max_delta = delta.detach()

    #restarts

    for k in range (restarts):
        delta = torch.rand_like(X, requires_grad=True) 
        delta.data *= (2.0*delta.data - 1.0)*epsilon 
        delta.data /= norms(delta.detach()).clamp(min=epsilon)
        for t in range(num_iter):
            output = model(X+delta)[0]
            incorrect = output.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            #Finding the correct examples so as to attack only them
            loss = nn.CrossEntropyLoss()(model(X + delta)[0], y)
            loss.backward()
            delta.data +=  correct*alpha*delta.grad.detach() / norms(delta.grad.detach())
            delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
            delta.grad.zero_()  

        output = model(X+delta)[0]
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect] 

    return max_delta    


def pgd_l1_topk_wbn(model, X,y, epsilon = 10, alpha = 1.0, num_iter = 50, k_map = 0, gap = 0.05, device = "cuda:0", restarts = 0, randomize = 0):
    #Gap : Dont attack pixels closer than the gap value to 0 or 1
    # ipdb.set_trace()
    gap = gap
    max_delta = torch.zeros_like(X)
    if randomize:
        delta = torch.from_numpy(np.random.laplace(size=X.shape)).float().to(device)
        delta.data = delta.data*epsilon/norms_l1(delta.detach())
        delta.requires_grad = True
    else:
        delta = torch.zeros_like(X, requires_grad = True)
    alpha_l_1_default = alpha

    for t in range (num_iter):
        output = model(X+delta)[0]
        incorrect = output.max(1)[1] != y 
        correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        #Finding the correct examples so as to attack only them
        loss = nn.CrossEntropyLoss()(model(X+delta)[0], y)
        loss.backward()
        if k_map == 0:
            k = random.randint(5,20)
            alpha   = (alpha_l_1_default/k)
        elif k_map == 1:
            k = random.randint(10,40)
            alpha   = (alpha_l_1_default/k)
        else:
            k = 10
            alpha = alpha_l_1_default
        delta.data += alpha*correct*l1_dir_topk(delta.grad.detach(), delta.data, X, gap,k)
        if (norms_l1(delta) > epsilon).any():
            delta.data = proj_l1ball(delta.data, epsilon, device)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
        delta.grad.zero_() 

    max_delta = delta.detach()

    #Restarts    
    for k in range(restarts):
        delta = torch.rand_like(X,requires_grad = True)
        delta.data = (2*delta.data - 1.0)
        delta.data = delta.data*epsilon/norms_l1(delta.detach())
        for t in range (num_iter):
            output = model(X+delta)[0]
            incorrect = output.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            #Finding the correct examples so as to attack only them
            loss = nn.CrossEntropyLoss()(model(X+delta)[0], y)
            loss.backward()
            if k_map == 0:
                k = random.randint(5,20)
                alpha   = (alpha_l_1_default/k)
            elif k_map == 1:
                k = random.randint(10,40)
                alpha   = (alpha_l_1_default/k)
            else:
                k = 10
                alpha = alpha_l_1_default
            delta.data += alpha*correct*l1_dir_topk(delta.grad.detach(), delta.data, X, gap,k)
            if (norms_l1(delta) > epsilon).any():
                delta.data = proj_l1ball(delta.data, epsilon, device)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
            delta.grad.zero_() 
        output = model(X+delta)[0]
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect]   

    return max_delta



# norm funcs
def pgd_linf_norm(model, X, y, epsilon=0.3, alpha=0.01, num_iter = 50, randomize = 0, restarts = 0, device = "cuda:0", norm_type=3):
    """ Construct FGSM adversarial examples on the examples X"""
    # ipdb.set_trace()
   
    max_delta = torch.zeros_like(X)
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = (delta.data * 2.0 - 1.0) * epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)    
    for t in range(num_iter):
        output = model(X+delta, is_training=True, norm_type=norm_type)[0]
        incorrect = output.max(1)[1] != y 
        correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        #Finding the correct examples so as to attack only them
        loss = nn.CrossEntropyLoss()(model(X + delta, is_training=True, norm_type=norm_type)[0], y)
        loss.backward()
        delta.data = (delta.data + alpha*correct*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        delta.grad.zero_()
    max_delta = delta.detach()
    
    for i in range (restarts):
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = (delta.data * 2.0 - 1.0) * epsilon

        for t in range(num_iter):
            output = model(X+delta, is_training=True, norm_type=norm_type)[0]
            incorrect = output.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            #Finding the correct examples so as to attack only them            
            loss = nn.CrossEntropyLoss()(model(X + delta, is_training=True, norm_type=norm_type)[0], y)
            loss.backward()
            delta.data = (delta.data + alpha*correct*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
            delta.grad.zero_()

        output = model(X+delta, is_training=True, norm_type=norm_type)[0]
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks        
        max_delta[incorrect] = delta.detach()[incorrect]
    return max_delta



def pgd_l2_norm(model, X, y, epsilon=2.0, alpha=0.1, num_iter = 100, restarts = 0, device = "cuda:0", randomize = 0, norm_type=2):
    # ipdb.set_trace()
    max_delta = torch.zeros_like(X)
    if random:
        delta = torch.rand_like(X, requires_grad=True) 
        delta.data *= (2.0*delta.data - 1.0)
        delta.data = delta.data*epsilon/norms_l2(delta.detach()) 
    else:
        delta = torch.zeros_like(X, requires_grad=True) 
    for t in range(num_iter):
        output = model(X+delta, is_training=True, norm_type=norm_type)[0]
        incorrect = output.max(1)[1] != y 
        correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        #Finding the correct examples so as to attack only them
        loss = nn.CrossEntropyLoss()(model(X + delta, is_training=True, norm_type=norm_type)[0], y)
        loss.backward()
        delta.data +=  correct*alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data *=  epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.data =   torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]     
        delta.grad.zero_()  

    max_delta = delta.detach()

    #restarts

    for k in range (restarts):
        delta = torch.rand_like(X, requires_grad=True) 
        delta.data *= (2.0*delta.data - 1.0)*epsilon 
        delta.data /= norms(delta.detach()).clamp(min=epsilon)
        for t in range(num_iter):
            output = model(X+delta, is_training=True, norm_type=norm_type)[0]
            incorrect = output.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            #Finding the correct examples so as to attack only them
            loss = nn.CrossEntropyLoss()(model(X + delta, is_training=True, norm_type=norm_type)[0], y)
            loss.backward()
            delta.data +=  correct*alpha*delta.grad.detach() / norms(delta.grad.detach())
            delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
            delta.grad.zero_()  

        output = model(X+delta, is_training=True, norm_type=norm_type)[0]
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect] 

    return max_delta    


def pgd_l1_topk_norm(model, X,y, epsilon = 10, alpha = 1.0, num_iter = 50, k_map = 0, gap = 0.05, device = "cuda:0", restarts = 0, randomize = 0, norm_type=1):
    #Gap : Dont attack pixels closer than the gap value to 0 or 1
    # ipdb.set_trace()
    gap = gap
    max_delta = torch.zeros_like(X)
    if randomize:
        delta = torch.from_numpy(np.random.laplace(size=X.shape)).float().to(device)
        delta.data = delta.data*epsilon/norms_l1(delta.detach())
        delta.requires_grad = True
    else:
        delta = torch.zeros_like(X, requires_grad = True)
    alpha_l_1_default = alpha

    for t in range (num_iter):
        output = model(X+delta, is_training=True, norm_type=norm_type)[0]
        incorrect = output.max(1)[1] != y 
        correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        #Finding the correct examples so as to attack only them
        loss = nn.CrossEntropyLoss()(model(X+delta, is_training=True, norm_type=norm_type)[0], y)
        loss.backward()
        if k_map == 0:
            k = random.randint(5,20)
            alpha   = (alpha_l_1_default/k)
        elif k_map == 1:
            k = random.randint(10,40)
            alpha   = (alpha_l_1_default/k)
        else:
            k = 10
            alpha = alpha_l_1_default
        delta.data += alpha*correct*l1_dir_topk(delta.grad.detach(), delta.data, X, gap,k)
        if (norms_l1(delta) > epsilon).any():
            delta.data = proj_l1ball(delta.data, epsilon, device)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
        delta.grad.zero_() 

    max_delta = delta.detach()

    #Restarts    
    for k in range(restarts):
        delta = torch.rand_like(X,requires_grad = True)
        delta.data = (2*delta.data - 1.0)
        delta.data = delta.data*epsilon/norms_l1(delta.detach())
        for t in range (num_iter):
            output = model(X+delta, is_training=True, norm_type=norm_type)[0]
            incorrect = output.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            #Finding the correct examples so as to attack only them
            loss = nn.CrossEntropyLoss()(model(X+delta, is_training=True, norm_type=norm_type)[0], y)
            loss.backward()
            if k_map == 0:
                k = random.randint(5,20)
                alpha   = (alpha_l_1_default/k)
            elif k_map == 1:
                k = random.randint(10,40)
                alpha   = (alpha_l_1_default/k)
            else:
                k = 10
                alpha = alpha_l_1_default
            delta.data += alpha*correct*l1_dir_topk(delta.grad.detach(), delta.data, X, gap,k)
            if (norms_l1(delta) > epsilon).any():
                delta.data = proj_l1ball(delta.data, epsilon, device)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
            delta.grad.zero_() 
        output = model(X+delta, is_training=True, norm_type=norm_type)[0]
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect]   

    return max_delta
