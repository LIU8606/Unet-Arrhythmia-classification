import numpy as np


def adjust_Rpeak(X_signal,y_label,peak_index,DS_name):

    Rpeak = []

    for f in range(X_signal.shape[0]):

        r_peak = []
        for i in range(peak_index[f].shape[0]):
            if DS_name == "AHA":
                max_i = np.argmax(X_signal[f][peak_index[f][i]:peak_index[f][i]+100])
            elif DS_name == "MIT":
                max_i = 0
            r_peak.append(peak_index[f][i]+max_i)
                
        Rpeak.append(r_peak)    

    return np.array(Rpeak) 



def database(DS_name,set_name): 

    if DS_name == "AHA":
        if set_name == "train":
            X_signal = np.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/AHA database/AHA_train_data_denoise.npy")
            y_label = np.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/AHA database/AHA_train_label.npy")
            peak_index = np.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/AHA database/AHA_train_peak.npy")
        elif set_name == "test":
            X_signal = np.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/AHA database/AHA_test_data_denoise.npy")
            y_label = np.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/AHA database/AHA_test_label.npy")
            peak_index = np.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/AHA database/AHA_test_peak.npy")  

        X_signal = X_signal/40
        Rpeak = adjust_Rpeak(X_signal,y_label,peak_index,DS_name)

    elif DS_name == "MIT":
        if set_name == "train":
            X_signal = np.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/MIT database/MIT_train_data_denoise.npy")
            y_label = np.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/MIT database/MIT_train_label.npy")
            peak_index = np.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/MIT database/MIT_train_peak.npy")
        elif set_name == "test":
            X_signal = np.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/MIT database/MIT_test_data_denoise.npy")
            y_label = np.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/MIT database/MIT_test_label.npy")
            peak_index = np.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/MIT database/MIT_test_peak.npy")
        
        Rpeak = adjust_Rpeak(X_signal,y_label,peak_index,DS_name)

    elif DS_name == "ALL":
        AHA_X_signal = np.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/AHA database/AHA_train_data_denoise.npy")
        AHA_y_label = np.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/AHA database/AHA_train_label.npy")
        AHA_peak_index = np.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/AHA database/AHA_train_peak.npy")
        MIT_X_signal = np.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/MIT database/MIT_train_data_denoise.npy")
        MIT_y_label = np.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/MIT database/MIT_train_label.npy")
        MIT_peak_index = np.load("/mnt/nas/homes/柳妹/ECG-Heartbeat-Classification-seq2seq-model-master/MIT database/MIT_train_peak.npy")

        AHA_Rpeak = adjust_Rpeak(AHA_X_signal,AHA_y_label,AHA_peak_index,"AHA")
        MIT_Rpeak = adjust_Rpeak(MIT_X_signal,MIT_y_label,MIT_peak_index,"MIT")

        X_signal = []
        y_label = []
        Rpeak = []
        for i in range(AHA_X_signal.shape[0]):
            X_signal.append(AHA_X_signal[i])
            y_label.append(AHA_y_label[i])
            Rpeak.append(AHA_Rpeak[i])
        for i in range(MIT_X_signal.shape[0]):
            X_signal.append(MIT_X_signal[i])
            y_label.append(MIT_y_label[i])
            Rpeak.append(MIT_Rpeak[i])

        X_signal = np.array(X_signal)

    for f in range(X_signal.shape[0]):
        X_signal[f] = (X_signal[f] - X_signal[f].mean())/ X_signal[f].std()

    #Rpeak = adjust_Rpeak(X_signal,y_label,peak_index,DS_name) 

    return np.array(X_signal), np.array(y_label), np.array(Rpeak) 

def get_RR(X_signal,train_Rpeak):

    print("RR")

    RR_interval = []
    for f in range(X_signal.shape[0]):

        RR = np.zeros((len(X_signal[f]),1))
        tmp = []

        for i in range(len(train_Rpeak[f])-1):
            tmp.append(train_Rpeak[f][i+1]-train_Rpeak[f][i])
        
        RR_mean = np.mean(tmp)
        RR_std = np.std(tmp)

        for i in range(len(train_Rpeak[f])-1):
            RR[train_Rpeak[f][i]:train_Rpeak[f][i+1]] = (tmp[i] - RR_mean)/RR_std
        
        RR_interval.append(RR)

    return np.array(RR_interval)


def guassian(x, mu, sigma):
    return np.exp( - (x - mu)**2 / (2 * sigma**2) )

def get_mask(train_data,train_label,train_Rpeak):


    N_GT = []
    V_GT = []
    F_GT = []
    Other_GT = []
    background_GT = []

    for i in range(train_data.shape[0]):
        
        N_l = np.zeros(len(train_data[i]))
        V_l = np.zeros(len(train_data[i]))
        F_l = np.zeros(len(train_data[i]))
        other_l = np.zeros(len(train_data[i]))

        background_l = np.ones(len(train_data[i]))

        #background_l_square = np.ones(len(train_data[i]))

        for p in train_Rpeak[i]:

            s = p - 50
            e = p + 50

            if s < 0 :
                s = 0

            if e >= len(train_data[i]):
                e = len(train_data[i])-1

            guassian_tmp = [guassian(g, p, 10) for g in range(s,e+1,1)]
            
            if train_label[i][train_Rpeak[i].index(p)] == "N":

                N_l[s:e+1] = np.array(guassian_tmp)
            
            elif train_label[i][train_Rpeak[i].index(p)] == "V":

                V_l[s:e+1] = np.array(guassian_tmp)
            
            elif train_label[i][train_Rpeak[i].index(p)] == "F":

                F_l[s:e+1] = np.array(guassian_tmp)

            else:
                other_l[s:e+1] = np.array(guassian_tmp)
                

        background_l = background_l - N_l - V_l - F_l - other_l
        #background_l_square = background_l_square - N_l_square - V_l_square - F_l_square

        # for b_i in range(len(train_data[i])):

        #     if N_l[b_i] == 0 and V_l[b_i] == 0 and F_l[b_i] == 0:
        #         background_l[b_i] = 0.5

            
        N_GT.append(N_l)
        V_GT.append(V_l)
        F_GT.append(F_l)
        Other_GT.append(other_l)
        background_GT.append(background_l)


    return np.array(N_GT),np.array(V_GT),np.array(F_GT),np.array(Other_GT),np.array(background_GT)


def database_10s(train_data,N_GT,V_GT,F_GT,Other_GT,backdround_GT):

    length = 2560

    X_train = []
    y_mask = []

    for d in range(train_data.shape[0]):

        num = len(train_data[d]) // length


        for i in range(num):

            if all(i == 0 for i in V_GT[d][i*length:(i+1)*length]) and all(i == 0 for i in F_GT[d][i*length:(i+1)*length]):
                continue

            y = []
            x = []

            # print("t",train_data[d][i*length:(i+1)*length].shape)
            # print(RR_interval[d][i*length:(i+1)*length].shape)
            X_train.append(train_data[d][i*length:(i+1)*length].reshape(length,1))
            #x.append(RR_interval[d][i*length:(i+1)*length])
            y.append(N_GT[d][i*length:(i+1)*length])
            y.append(V_GT[d][i*length:(i+1)*length])
            y.append(F_GT[d][i*length:(i+1)*length])
            y.append(Other_GT[d][i*length:(i+1)*length])
            y.append(backdround_GT[d][i*length:(i+1)*length])
            
            #X_train.append(x)
            y_mask.append(y)
            

    return np.array(X_train), np.array(y_mask)



def get_data(database_name,set_name):

    X_signal, y_label, Rpeak = database(database_name,set_name)
    #RR_interval = get_RR(X_signal,Rpeak)
    N_GT,V_GT,F_GT,Other_GT,backdround_GT = get_mask(X_signal, y_label, Rpeak)
    X, y = database_10s(X_signal,N_GT,V_GT,F_GT,Other_GT,backdround_GT)
    #X, y = database_10s(X_signal,N_GT,V_GT,F_GT,backdround_GT,RR_interval)

    return X, y










        