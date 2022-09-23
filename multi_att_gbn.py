from __future__ import print_function
import sys, os
sys.path.append('./model/')

import argparse
import random
from imfgsm_attack import _mim_whitebox
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import StepLR
import numpy as np
from lenet5_gate_conv import lenet5_conv1_fc
from torchvision.datasets import mnist
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from mnist_funcs import *
import time
import foolbox as fb

import tensorflow as tf
from realsafe.model.pytorch_wrapper import pytorch_classifier_with_logits
from realsafe.loss.cross_entropy import CrossEntropyLoss
from realsafe.loss.cw import CWLoss
from realsafe.attack.spsa import SPSA
from realsafe.attack.nattack import NAttack
from autoattack import AutoAttack


# Training settings
parser = argparse.ArgumentParser(description='attack implementation')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.1, help='Learning Rate')
parser.add_argument('--model_path', default="./w_bn/", help='model path')
parser.add_argument('--dataset_path', default="./MNIST/data/", help='dataset path')


args = parser.parse_args()

train_dataset = mnist.MNIST(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = mnist.MNIST(root='./data', train=False, download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)

gate_params = []


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
    return acc


def clip_l2_norm(cln_img, adv_img, eps):
    noise = adv_img - cln_img
    if torch.sqrt(torch.sum(noise**2)).item() > eps:
        clip_noise = noise * eps / torch.sqrt(torch.sum(noise**2))
        clip_adv = cln_img + clip_noise
        return clip_adv
    else:
        return adv_img

def clip_l1_norm(cln_img, adv_img, eps):
    noise = adv_img - cln_img
    if torch.sum(torch.abs(noise)).item() > eps:
        clip_noise = noise * eps / torch.sum(torch.abs(noise))
        clip_adv = cln_img + clip_noise
        return clip_adv
    else:
        return adv_img


def test(model):
    model.eval()
    f_model = fb.PyTorchModel(model, bounds=(0, 1))
    # prepare data
    each_pred = [[], [], [], [], [], [], [], [], [], [], [], [], []]
    attack = ['PGD-L1', 'BBA', 'PGD-L2', 'PGD-L2-fb', 'C&W', 'Gaussian', 'BA', 'PGD-Linf', 'PGD-Linf-fb', 'FGSM', 'BIM', 'MI-FGSM', 'clean']
    res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    total = 0
    ite = 0
    cnt = 0
    for step, (test_x, test_label) in enumerate(test_loader):
        x, y = test_x, test_label
        x = x.cuda()
        y = y.cuda().long()
        print('iter {}'.format(step))
        ite = ite + 1
        x_clean = x.cuda() 
        
        # generate adv data by multiple attacks
        adv_pgd_l1 = x + pgd_l1_topk(model, x, y)
        # BBA attack
        b_s = int(x.shape[0] / 4)
        init_batches = [(x[0:b_s], y[0:b_s]), (x[b_s:b_s*2], y[b_s:b_s*2]), (x[b_s*2:b_s*3], y[b_s*2:b_s*3]), (x[b_s*3:b_s*4], y[b_s*3:b_s*4])]
        init_attack = fb.attacks.DatasetAttack()
        init_attack.feed(f_model, init_batches[0][0])
        init_attack.feed(f_model, init_batches[1][0])
        init_attack.feed(f_model, init_batches[2][0])
        init_attack.feed(f_model, init_batches[3][0])
        bba_att = fb.attacks.L1BrendelBethgeAttack(init_attack=init_attack)
        adv_bba, _, success = bba_att(f_model, x, y, epsilons=10)
        adv_pgd_l2 = x + pgd_l2(model, x, y)
        pgdl2_att = fb.attacks.L2ProjectedGradientDescentAttack()
        adv_fbpgd_l2, _, success = pgdl2_att(f_model, x, y, epsilons=2.0)
        cw_att = fb.attacks.L2CarliniWagnerAttack()
        adv_cw, _, success = cw_att(f_model, x, y, epsilons=2.0)
        gauss_att = fb.attacks.L2AdditiveGaussianNoiseAttack()
        adv_guass, _, success = gauss_att(f_model, x, y, epsilons=2.0)
        ba_att = fb.attacks.BoundaryAttack()
        adv_ba, _, success = ba_att(f_model, x, y, epsilons=2.0)   
        adv_pgd_linf = x + pgd_linf(model, x, y)
        pgdlinf_att = fb.attacks.LinfProjectedGradientDescentAttack()
        adv_fbpgd_linf, _, success = pgdlinf_att(f_model, x, y, epsilons=0.3)       
        fgsm_att = fb.attacks.LinfFastGradientAttack()
        adv_fgsm, _, success = fgsm_att(f_model, x, y, epsilons=0.3)
        bim_att = fb.attacks.LinfBasicIterativeAttack()
        adv_bim, _, success = bim_att(f_model, x, y, epsilons=0.3)
        adv_mifgsm = _mim_whitebox(model, x, y, epsilon=0.3, num_steps=10, step_size=0.03, decay_factor=1.0)
        # clip bba, ba, and cw
        for i in range(0, x.shape[0]):  
            adv_bba[i] = clip_l1_norm(x_clean[i], adv_bba[i], 10)
            adv_cw[i] = clip_l2_norm(x_clean[i], adv_cw[i], 2.0)
            adv_ba[i] = clip_l2_norm(x_clean[i], adv_ba[i], 2.0)
        
        adv_pgd_l1, adv_bba, adv_pgd_l2, adv_fbpgd_l2, adv_cw, adv_guass, adv_ba, adv_pgd_linf, adv_fbpgd_linf, adv_fgsm, adv_bim, adv_mifgsm = adv_pgd_l1.cuda(), adv_bba.cuda(), adv_pgd_l2.cuda(), adv_fbpgd_l2.cuda(), adv_cw.cuda(), adv_guass.cuda(), adv_ba.cuda(), adv_pgd_linf.cuda(), adv_fbpgd_linf.cuda(), adv_fgsm.cuda(), adv_bim.cuda(), adv_mifgsm.cuda()
        # infer them by each branch       
        data = [adv_pgd_l1, adv_bba, adv_pgd_l2, adv_fbpgd_l2, adv_cw, adv_guass, adv_ba, adv_pgd_linf, adv_fbpgd_linf, adv_fgsm, adv_bim, adv_mifgsm, x_clean]
        for x_i in range(0, len(data)):
            with torch.no_grad():
                h = model(data[x_i], False, -1)
            _, predicted = torch.max(h.data, 1)       
            res[x_i] += (predicted == y).sum().item()
            # add
            each_pred[x_i] = each_pred[x_i] + (predicted == y).tolist()
            # add
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


@pytorch_classifier_with_logits(n_class=10, x_min=0.0, x_max=1.0, x_shape=(28, 28, 1), x_dtype=tf.float32, y_dtype=tf.int32)
class lenet5_tf(torch.nn.Module):
    def __init__(self, load_path):
        torch.nn.Module.__init__(self)
        self.model = lenet5_conv1_fc().cuda()
        wbn_model_path = load_path
        self.model.load_state_dict(torch.load(wbn_model_path))
        self.model.eval()

    def forward(self, x):
        x = x.transpose(1, 2).transpose(1, 3).contiguous()
        x = x.cuda()
        logits = self.model(x)
        return logits.cpu()
    
def trans(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return x
    
def test_aa_spsa_nattack(tf_model, torch_model):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.50
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    # add
    each_pred = [[], [], [], [], [], []]
    # add
    attack = ['SPSA-Linf', 'NATTACK-Linf', 'AA-Linf']
    res = [0, 0, 0, 0, 0, 0]
    total = 0
    ite = 0
    cnt = 0
    for step, (test_x, test_label) in enumerate(test_loader):
        x, y = test_x, test_label
        x = x.cuda()
        y = y.cuda().long()
        print('iter {}'.format(step))
        ite = ite + 1
        x_clean = x.cuda() 
        y_clean = y.cuda()
        # sapa, nattack
        spsa_att = SPSA(model=tf_model, loss=CrossEntropyLoss(tf_model), goal='ut', 
                        distance_metric='l_inf', session=session, samples_per_draw=8192, samples_batch_size=256)
        spsa_att.config(magnitude=0.3, max_queries=100*8192, sigma=0.01, lr=0.01)
        nattack_att = NAttack(model=tf_model, loss=CrossEntropyLoss(tf_model), goal='ut', 
                distance_metric='l_inf', session=session, samples_per_draw=300, samples_batch_size=25)
        nattack_att.config(magnitude=0.3, max_queries=300*600, sigma=0.01, lr=0.008)
        adv_spsa, adv_nattack = torch.zeros(x_clean.shape), torch.zeros(x_clean.shape)
        for i in range(0, x.shape[0]):
            adv_spsa[i] = trans(spsa_att.attack(x[i].permute(1, 2, 0).cpu(), y[i].cpu())).permute(2, 0, 1)
            adv_nattack[i] = trans(nattack_att.attack(x[i].permute(1, 2, 0).cpu(), y[i].cpu())).permute(2, 0, 1)
        # aa
        aa_att = AutoAttack(torch_model, norm='Linf', eps=0.3, version='standard')
        dict_adv_aa = aa_att.run_standard_evaluation_individual(x, y, bs=x.shape[0])
        
        adv_spsa, adv_nattack, adv_apgdce, adv_apgdt, adv_fabt, adv_square = adv_spsa.cuda(), adv_nattack.cuda(), dict_adv_aa['apgd-ce'].cuda(), dict_adv_aa['apgd-t'].cuda(), dict_adv_aa['fab-t'].cuda(), dict_adv_aa['square'].cuda()
        print('dist spsa', torch.max(torch.abs(adv_spsa - x)))
        print('dist nattack', torch.max(torch.abs(adv_nattack - x)))
        # infer them by each branch       
        data = [adv_spsa, adv_nattack, adv_apgdce, adv_apgdt, adv_fabt, adv_square]
        for x_i in range(0, len(data)):
            with torch.no_grad():
                h = torch_model(data[x_i], False, -1)
            _, predicted = torch.max(h.data, 1)       
            res[x_i] += (predicted == y).sum().item()
            # add
            each_pred[x_i] = each_pred[x_i] + (predicted == y).tolist()
            # add
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
    tf_wbnmodel = lenet5_tf(wbn_model_path)
    if os.path.exists(wbn_model_path):
        wbn_model.load_state_dict(torch.load(wbn_model_path))
        print('load wbn model.')
    else:
        print("load wbn failed.")               
            
    wbn_model = wbn_model.cuda()
    wbn_model.eval()
    test(wbn_model)
    test_aa_spsa_nattack(tf_wbnmodel, wbn_model)
