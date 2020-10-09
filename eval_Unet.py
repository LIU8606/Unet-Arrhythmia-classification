import sys
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
import wandb
from torch import optim
from tqdm import tqdm
from model.UNet import UNet
from model.UNet6 import UNet6
from model.UNet10 import UNet10
import numpy as np
import matplotlib.pyplot as plt
from show import show_onelead,show_twoleads

def eval_class(pred, ground_truth):

    ground_truth = ground_truth.cpu()
    pred = pred.cpu()

    confunsion_matrix = np.zeros((5,5))

    for i in range(pred.shape[0]):

        pred_class = pred[i].argmax(axis=0)
        ground_class = ground_truth[i].argmax(axis=0)

        pred_class = pred_class.cpu()
        ground_class = ground_class.cpu()


        for l in range(4):
                   
            check = pred_class[np.where(ground_truth[i][l] == 1)[0]]

            for p in range(5):
                confunsion_matrix[l][p] += len(np.where(np.array(check) == p)[0])


    return confunsion_matrix

def pred_peak_class(pred, ground_truth):

    ground_truth = ground_truth.cpu()
    pred = pred.cpu()

    confunsion_matrix = np.zeros((4,4))

    for i in range(pred.shape[0]):

        threshold = 0.5

        N_R_wave = np.where(pred[i][0] >= threshold)[0]
        V_R_wave = np.where(pred[i][1] >= threshold)[0]
        F_R_wave = np.where(pred[i][2] >= threshold)[0]


        N_peak = np.where(ground_truth[i][0] == 1)[0]
        V_peak = np.where(ground_truth[i][1] == 1)[0]
        F_peak = np.where(ground_truth[i][2] == 1)[0]

        N_pred_peak = []
        V_pred_peak = []
        F_pred_peak = []


        for p in N_R_wave:
            if p-1 in N_R_wave and p+1 in N_R_wave:
                if pred[i][0][p] >= pred[i][0][p-1] and pred[i][0][p] >= pred[i][0][p+1]:
                    N_pred_peak.append(p)
        
        pred_N = []

        if len(N_pred_peak) >0:
            pred_N.append(N_pred_peak[0])
            for j in range(len(N_pred_peak)-1):
                if N_pred_peak[j+1] - N_pred_peak[j] > 100:
                    pred_N.append(N_pred_peak[j+1])

        for p in V_R_wave:
            if p-1 in V_R_wave and p+1 in V_R_wave:
                if pred[i][1][p] >= pred[i][1][p-1] and pred[i][1][p] >= pred[i][1][p+1]:
                    V_pred_peak.append(p)
        
        pred_V = []

        if len(V_pred_peak) >0:
            pred_V.append(V_pred_peak[0])
            for j in range(len(V_pred_peak)-1):
                if V_pred_peak[j+1] - V_pred_peak[j] > 100:
                    pred_V.append(V_pred_peak[j+1])


        for p in F_R_wave:
            if p-1 in F_R_wave and p+1 in F_R_wave:
                if pred[i][2][p] >= pred[i][2][p-1] and pred[i][2][p] >= pred[i][2][p+1]:
                    F_pred_peak.append(p)
        
        pred_F = []

        if len(F_pred_peak) >0:
            pred_F.append(F_pred_peak[0])
            for j in range(len(F_pred_peak)-1):
                if F_pred_peak[j+1] - F_pred_peak[j] > 100:
                    pred_F.append(F_pred_peak[j+1])   

           

        # print(N_peak)
        # print(pred_N)

        

def Unet_validation(net, loader, device):

    net.eval()
    n_val = len(loader)

    C = np.zeros((5,5))

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False, ncols=100) as pbar:
        for batch in loader:
            x, ground_truth = batch[0], batch[1]
            x = x.to(device, dtype=torch.float32)
            ground_truth = ground_truth.to(device, dtype=torch.float32)

            with torch.no_grad():
                pred = net(x)
            
            C += eval_class(pred, ground_truth)

            pbar.update()
    
    TP = C[0][0] + C[1][1] + C[2][2] + C[3][3]

    return C, TP/np.sum(C), C[2][2]/np.sum(C[2])


def Unet_test(loader, device):

    test_net = UNet6(in_ch=1, out_ch=5)
    test_net.load_state_dict(torch.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/signal_classification/2020-10-08 17:34:30 0.97 0.56.pkl"))
    test_net.eval()
    test_net.to(device)
    n_val = len(loader)

    C = np.zeros((5,5))

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False, ncols=100) as pbar:
        for batch in loader:
            x, ground_truth = batch[0], batch[1]
            x = x.to(device, dtype=torch.float32)
            ground_truth = ground_truth.to(device, dtype=torch.float32)
 
            with torch.no_grad():
                pred = test_net(x)
            
            C += eval_class(pred, ground_truth)

            pbar.update()

            show_onelead(ground_truth,x,pred,0,"Image/AHA_test_MSE_distance_other/")
    
    TP = C[0][0] + C[1][1] + C[2][2] + C[3][3]

    return C, TP/np.sum(C), C[2][2]/np.sum(C[2])



