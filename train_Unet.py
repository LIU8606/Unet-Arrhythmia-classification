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
from Unet_data_2 import get_data2
from eval_Unet import eval_class,pred_peak_class
from eval_Unet import Unet_validation,Unet_test
from show import show_onelead,show_twoleads
import time

wandb_config = {
        "epochs": 200,
        "batch_size": 32,
        "lr": 1e-4,
        }


def train_model(net, epochs=6000, batch_size=32, lr=1e-4, device=torch.device('cuda')):
    """
    training the UNet model
    Args:
        net: (nn.Module) UNet module
        epochs: (int) training epochs
        batch_size: (int) batch_size
        lr: (float) learning rate
        device: (torch.device) execute device. cuda/cpu
    """
    wandb.watch(net)

    #data = load_dataset_using_pointwise_labels()

    X, Y = get_data("AHA","train")
    #X = np.load("npy/Unet_AHA_X_removeallN.npy")
    #Y = np.load("npy/Unet_AHA_Y_removeallN.npy")

    #X = X.reshape(X.shape[0],X.shape[1],X.shape[2])
    X = X.reshape(X.shape[0],X.shape[1],1)
    X = np.swapaxes(X, 1, 2)
    
    print(X.shape)
    print(Y.shape)

    X = torch.Tensor(X)
    Y = torch.Tensor(Y)

    data = TensorDataset(X, Y)

    # calculate train, validation, test dataset size
    n_val = int(len(data) * 0.2)
    n_train = len(data) - n_val
    train, val= random_split(data, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    #test###

    X_test, Y_test = get_data("AHA","test")
    #X_test = np.load("npy/Unet_AHA_Xtest_removeallN.npy")
    #Y_test = np.load("npy/Unet_AHA_Ytest_removeallN.npy")
    
    #X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2])
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
    X_test = np.swapaxes(X_test, 1, 2)

    print(X_test.shape)
    print(Y_test.shape)

    X_test = torch.Tensor(X_test)
    Y_test = torch.Tensor(Y_test)

    test_data = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)


    val_confunsion_matrix, ACCU, F_ACCU = Unet_test(test_loader, device)
    print(np.array(val_confunsion_matrix,dtype = "int"))
    print(ACCU, F_ACCU)
    return 

    global_step = 0

    logging.info(f'''Start training:
        Epochs:         {epochs}
        Batch size:     {batch_size}
        Learning rate:  {lr}
        Training size:  {n_train}
        Validation size:{n_val}
        Device:         {device.type}
        ''')

    optimizer = optim.Adam(net.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    class_weight = [0.3,0.8,1,0.5,0.1]
    criterion = nn.MSELoss()

    # weight = torch.Tensor(np.array(weight))
    # weight = weight.to(device, dtype=torch.float32)
    # criterion = nn.CrossEntropyLoss(weight)
    # criterion = FocalLoss(alpha=wandb.config.focalloss_alpha,gamma=wandb.config.focalloss_gamma)

    max_ACCU = 0

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0

        confunsion_matrix = np.zeros((5,5))

        with tqdm(total=n_train, ncols=100) as pbar:
            for batch in train_loader:
                x = batch[0]
                ground_truth = batch[1]

                # # cross entropy
                # gt = ground_truth.cpu().detach().numpy()
                # label = np.zeros((gt.shape[0],gt.shape[2]))
                # for i in range(gt.shape[0]):
                #     for cl in range(4):
                #         pos = np.where(ground_truth[i][cl] > 0)[0]
                #         label[i][pos] = cl   

                # label = torch.Tensor(label)
                # label = label.to(device, dtype=torch.float32)


                # gt = ground_truth.cpu().detach().numpy()
                # weight = np.zeros((gt.shape[0],4,gt.shape[2])) + 0.1
                # for i in range(gt.shape[0]):
                #     for cl in range(4):
                #         pos = np.where(gt[i][cl] > 0)[0]
                #         weight[i][cl][pos] = class_weight[cl]

                # weight = torch.Tensor(weight)
                # weight = weight.to(device, dtype=torch.float32)


                x = x.to(device, dtype=torch.float32)
                ground_truth = ground_truth.to(device, dtype=torch.float32)

                pred = net(x)

                for i in range(4):
                    #class_loss = criterion(pred[:,i,:]* weight[:,i,:], ground_truth[:,i,:]* weight[:,i,:])
                    class_loss = criterion(pred[:,i,:], ground_truth[:,i,:]) * class_weight[i]
                    if i == 0:
                        loss = class_loss
                    else:
                        loss += class_loss 

                # label = label.long()
                # loss = criterion(pred, label)
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.set_description('Epoch %d / %d' % (epoch + 1, epochs))
                pbar.update(x.shape[0])
                tqdm._instances.clear()
                global_step += 1

                confunsion_matrix += eval_class(pred,ground_truth)
            #pred_peak_class(pred, ground_truth)

            show_onelead(ground_truth,x,pred,epoch,"Image/AHA_train_MSE_distance_other/")

            print()
            print("train:")
            print(np.array(confunsion_matrix,dtype = "int"))

            val_confunsion_matrix, ACCU, F_ACCU = Unet_validation(net, val_loader, device) 

            print("val: Accu =",ACCU)
            print(np.array(val_confunsion_matrix,dtype = "int"))

        if ACCU > max_ACCU:
            torch.save(net.state_dict(), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " "+str(round(ACCU,2)) + " " +str(round(F_ACCU,2)) +".pkl")
            max_ACCU = ACCU




def train():
    ex = wandb.init(project="PQRST-segmentation")
    ex.config.setdefaults(wandb_config)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet6(in_ch=1, out_ch=5)
    net.to(device)

    try:
        train_model(net=net, device=device, batch_size=wandb.config.batch_size, lr=wandb.config.lr, epochs=wandb.config.epochs)
    except KeyboardInterrupt:
        try:
            save = input("save?(y/n)")
            if save == "y":
                torch.save(net.state_dict(), 'net_params.pkl')
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == '__main__':
    train()
