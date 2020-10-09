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
from Unet_data import get_data
import time

def show_onelead(ground_truth,x,pred,epoch,pos):

    peak = []
    ground_truth = ground_truth.cpu().detach().numpy()
    ground_class = ground_truth[0].argmax(axis=0)
    N_peak = np.where(ground_truth[0][0] == 1)[0]
    V_peak = np.where(ground_truth[0][1] == 1)[0]
    F_peak = np.where(ground_truth[0][2] == 1)[0]
    other_peak = np.where(ground_truth[0][3] == 1)[0]
    peak.extend(N_peak)
    peak.extend(V_peak)
    peak.extend(F_peak)
    peak.extend(other_peak)
    peak = sorted(peak)
    #print(ground_class[peak])


    plt.figure(figsize = (20,10))
    plt.subplot(211)
    x = x.cpu().detach().numpy()
    plt.plot([i for i in range(2560)],x[0].reshape(2560,))
    plt.plot(N_peak,x[0].reshape(2560,)[N_peak],"bo")
    plt.plot(V_peak,x[0].reshape(2560,)[V_peak],"ro")
    plt.plot(F_peak,x[0].reshape(2560,)[F_peak],"go")
    plt.plot(other_peak,x[0].reshape(2560,)[other_peak],"ko")

    plt.subplot(212)
    pred = pred.cpu().detach().numpy()
    plt.plot([i for i in range(2560)],pred[0][0])
    plt.plot([i for i in range(2560)],pred[0][1])
    plt.plot([i for i in range(2560)],pred[0][2])
    plt.plot([i for i in range(2560)],pred[0][3])
    #legend = plt.legend([N,V,F],["N","V","F"])
    plt.show()
    plt.savefig(pos+str(epoch)+" "+str(ground_class[peak]))

def show_twoleads(ground_truth,x,pred,epoch,pos):

    peak = []
    ground_truth = ground_truth.cpu().detach().numpy()
    ground_class = ground_truth[0].argmax(axis=0)
    N_peak = np.where(ground_truth[0][0] == 1)[0]
    V_peak = np.where(ground_truth[0][1] == 1)[0]
    F_peak = np.where(ground_truth[0][2] == 1)[0]
    other_peak = np.where(ground_truth[0][3] == 1)[0]
    peak.extend(N_peak)
    peak.extend(V_peak)
    peak.extend(F_peak)
    peak.extend(other_peak)
    peak = sorted(peak)
    #print(ground_class[peak])


    plt.figure(figsize = (20,10))
    plt.subplot(311)
    x = x.cpu().detach().numpy()
    plt.plot([i for i in range(2560)],x[0][0].reshape(2560,))
    plt.plot(N_peak,x[0][0].reshape(2560,)[N_peak],"bo")
    plt.plot(V_peak,x[0][0].reshape(2560,)[V_peak],"ro")
    plt.plot(F_peak,x[0][0].reshape(2560,)[F_peak],"go")

    plt.subplot(312)
    plt.plot([i for i in range(2560)],x[0][1].reshape(2560,))
    plt.plot(N_peak,x[0][1].reshape(2560,)[N_peak],"bo")
    plt.plot(V_peak,x[0][1].reshape(2560,)[V_peak],"ro")
    plt.plot(F_peak,x[0][1].reshape(2560,)[F_peak],"go")

    plt.subplot(313)
    pred = pred.cpu().detach().numpy()
    plt.plot([i for i in range(2560)],pred[0][0])
    plt.plot([i for i in range(2560)],pred[0][1])
    plt.plot([i for i in range(2560)],pred[0][2])
    #legend = plt.legend([N,V,F],["N","V","F"])
    plt.show()
    plt.savefig(pos+str(epoch)+" "+str(ground_class[peak]))

