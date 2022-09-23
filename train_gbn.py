from __future__ import print_function
import sys, os
sys.path.append('./model/')

import argparse
import random
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
import numpy as np
from lenet5_gate_conv import lenet5_conv1_fc
from torchvision.datasets import mnist
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from mnist_funcs import *
import time


# Training settings
parser = argparse.ArgumentParser(description='attack implementation')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.1, help='Learning Rate')
parser.add_argument('--model_path', default="./ckpt/", help='model path')


args = parser.parse_args()

train_dataset = mnist.MNIST(root='./data', train=True, download=True, transform=ToTensor())
test_dataset_all = mnist.MNIST(root='./data', train=False, download=True, transform=ToTensor())
test_dataset, val_dataset = random_split(test_dataset_all, [8000, 2000])
train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)

gate_params = []


def train_op(model):    
    optimizer1 = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.8)
    optimizer2 = torch.optim.SGD(gate_params, lr=args.lr, momentum=0.8)
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 'max', factor=0.1, patience=5, eps=1e-06, verbose=True)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'max', factor=0.1, patience=5, eps=1e-06, verbose=True)
    loss_func = nn.CrossEntropyLoss()
    curr_lr = args.lr

    end = time.time()
    testop_acc = 0
    best_testop_acc = 0
    for epoch in range(args.epoch):
        # adjust_learning_rate(args.lr, optimizer, epoch)
        scheduler1.step(testop_acc)
        scheduler2.step(testop_acc)
        for step, (train_x, train_label) in enumerate(train_loader):
            x, y = train_x.cuda(), train_label.cuda().long()
            clean_x = x.cuda()
            clean_y = y.cuda()           
            model.eval()
            # generate l1, l2, linf adv data
            adv_x_L1_l1 = x + pgd_l1_topk_norm(model, x, y, norm_type=1)
            adv_x_L2_l2 = x + pgd_l2_norm(model, x, y, norm_type=2)
            adv_x_Linf_linf = x + pgd_linf_norm(model, x, y, norm_type=3)
            model.train()
            
            adv_x_L1_l1, adv_x_L2_l2, adv_x_Linf_linf = adv_x_L1_l1.cuda(), adv_x_L2_l2.cuda(), adv_x_Linf_linf.cuda()

            # clean examples
            logits, clean_gt_loss = model(clean_x, True, 0)
            clean_loss = loss_func(logits, clean_y)

            # adversarial examples       
            logits_Linf, gt_loss_linf = model(adv_x_Linf_linf, True, 3)
            loss_linf = loss_func(logits_Linf, clean_y)

            logits_L1, gt_loss_l1 = model(adv_x_L1_l1, True, 1)
            loss_l1 = loss_func(logits_L1, clean_y)
            
            logits_L2, gt_loss_l2 = model(adv_x_L2_l2, True, 2)
            loss_l2 = loss_func(logits_L2, clean_y)
            
            loss = (clean_loss + loss_linf + loss_l1 + loss_l2) / 4
            gt_loss = (clean_gt_loss + gt_loss_linf + gt_loss_l1 + gt_loss_l2) / 4

            optimizer1.zero_grad()
            loss.backward(retain_graph=True)
            optimizer1.step()
            
            optimizer2.zero_grad()           
            gt_loss.backward()           
            optimizer2.step()

            # test_adv_white(model, 0.031, 'fgsm')
            # test acc for validation set
            if step % 160 == 0:
                print('epoch={}/{}, step={}/{}'.format(epoch, args.epoch, step, len(train_loader)))
                testop_acc = test_op(model, testing_loader=val_loader)
                if testop_acc > best_testop_acc:
                    best_testop_acc = testop_acc
                    print('save the best testop acc model!')
                    torch.save(model.state_dict(), args.model_path + 'gbn_best.pkl')
            if step % 40 == 0:
                with torch.no_grad():
                    test_output, _ = model(clean_x, True, 0)
                train_loss = loss_func(test_output, clean_y)
                pred_y = torch.max(test_output, 1)[1].cuda().data.cpu().squeeze().numpy()
                Accuracy = float((pred_y == clean_y.data.cpu().numpy()).astype(int).sum()) / float(clean_y.size(0))
                print('train loss: %.4f' % train_loss.data.cpu().numpy(), '| train accuracy: %.2f' % Accuracy)
                
        if os.path.exists(args.model_path) == False:
            os.makedirs(args.model_path)
        
        if epoch % 10 == 0:
            print('saving model...')
            test_op(model, testing_loader=val_loader)
            torch.save(model.state_dict(), args.model_path + 'gbn' + str(epoch) + '.pkl')
                
    print('saving model...')
    if os.path.exists(args.model_path) == False:
        os.makedirs(args.model_path)

    torch.save(model.state_dict(), args.model_path + 'final_gbn.pkl')
    print('Training spends {:.3f} time'.format(time.time() - end))


def test_op(model, f=None, testing_loader=None):
    # Test the model
    model.eval()
    correct = 0
    total = 0
    for step, (test_x, test_label) in enumerate(testing_loader):
        #print('start test')
        x, y = test_x, test_label
        x = x.cuda()
        y = y.cuda().long()
        with torch.no_grad():
            h = model(x, False, 0)
        _, predicted = torch.max(h.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    acc = 100 * correct / total
    if f != None:
        f.write('Accuracy on the test images with '+ type + ' : {:.2f} %'.format(acc))
        f.write('\n')
    print('Accuracy of the model on the test images: {:.2f} %'.format(acc))
    # print('now is {}'.format(type(model)))    
    model.train(True)
    return acc


def test(model):
    model.eval()
    # prepare data
    # infer   
    res = [0, 0, 0, 0]
    total = 0
    ite = 0
    for step, (test_x, test_label) in enumerate(test_loader):
        #print('start test')
        x, y = test_x, test_label
        x = x.cuda()
        y = y.cuda().long()
        x_clean = x.cuda()
        # generate l1, l2, linf adv data
        adv_x_Linf = x + pgd_linf(model, x, y)
        adv_x_L1 = x + pgd_l1_topk(model, x, y)
        adv_x_L2 = x + pgd_l2(model, x, y)
        x_l1, x_l2, x_inf = adv_x_L1.cuda(), adv_x_L2.cuda(), adv_x_Linf.cuda()
        
        # infer them by each branch       
        data = [x_clean, x_l1, x_l2, x_inf]
        for x_i in range(0, len(data)):
            with torch.no_grad():
                h = model(data[x_i], False, x_i)
            _, predicted = torch.max(h.data, 1)       
            res[x_i] += (predicted == y).sum().item()
        total += y.size(0)
        print(res)
        print(total)
    # calc acc
    print('infer finished')
    print(res)
    print(total)
    for x_i in range(0, len(data)):
        res[x_i] = 100 * res[x_i] / total
    
    print('final result:', res)   
        

if __name__ == "__main__":
    wbn_model_path = args.model_path + 'none.pkl'
    wbn_model = lenet5_conv1_fc()
    if os.path.exists(wbn_model_path):
        wbn_model.load_state_dict(torch.load(wbn_model_path))
        print('load wbn model.')
    else:
        print("load wbn failed.")               
            
    for k, v in wbn_model.named_parameters():
        if 'fc' in k:
            gate_params.append(v)
    wbn_model = wbn_model.cuda()
    print('the len of gate params:', len(gate_params))
    train_op(wbn_model)
    test(wbn_model)
