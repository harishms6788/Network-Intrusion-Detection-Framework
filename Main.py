import numpy as np
import pandas as pd
import scipy
import pyhomogeneity as hg
import random as rn
from numpy import matlib
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.manifold import TSNE
from CMPA import CMPA
from Global_Vars import Global_Vars
from HGSO import HGSO
from Model_1DCNN import Model_1D_CNN
from Model_DBN import Model_DBN
from Model_MDDHN_AM import Model_MDDHN_AM
from Model_RNN import Model_RNN
from Model_TCNN import Model_TCNN
from Model_TCNN_RNN import Model_TCNN_RNN
from OOA import OOA
from Objective_Function import objfun_feat
from Plot_Results import *
from Proposed import Proposed
from TFMOA import TFMOA

no_of_dataset = 2

#  Read the dataset 1
an = 0
if an == 1:
    Dataset = './Dataset/Dataset_1/APA-DDoS-Dataset/APA-DDoS-Dataset.csv'
    Data = pd.read_csv(Dataset)
    Data.drop('frame.time', inplace=True, axis=1)
    data_1 = np.asarray(Data)
    data1 = data_1[:, 2:-1]

    tar = data_1[:, -1]
    Uni = np.unique(tar)
    uni = np.asarray(Uni)
    Target_1 = np.zeros((tar.shape[0], len(uni))).astype('int')
    for j in range(len(uni)):
        ind = np.where(tar == uni[j])
        Target_1[ind, j] = 1
    np.save('Data_1.npy', data1)  # Save the Dataset_1
    np.save('Target_1.npy', Target_1)  # Save the Target_1

# Read the dataset 2
an = 0
if an == 1:
    Dataset = './Dataset/Dataset_2/wustl-scada-2018.csv'  # Path of the dataset_5
    Data = pd.read_csv(Dataset)
    data_1 = np.asarray(Data)
    data1 = data_1[:, :-1]
    tar = data_1[:, -1]
    Uni = np.unique(tar)
    uni = np.asarray(Uni)
    Target_1 = np.zeros((tar.shape[0], len(uni))).astype('int')
    for j in range(len(uni)):
        ind = np.where(tar == uni[j])
        Target_1[ind, j] = 1
    np.save('Data_2.npy', data1)  # Save the Dataset_2
    np.save('Target_2.npy', Target_1)  # Save the Target_2

# Deep Feature from `1DCNN
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Data = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Feature = Model_1D_CNN(Data, Target)
        np.save('Deep_Feature_' + str(n + 1) + '.npy', Feature)

# TSNE Feature Extraction
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Data = np.load('Data' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        TSNE_Feat = []
        for i in range(len(Data)):
            print(n, i)
            tsne = TSNE(n_components=1, verbose=0, perplexity=13)
            tsne_pca_results = tsne.fit_transform(Data[i].reshape(-1, 1))
            TSNE_Feat.append(tsne_pca_results.reshape(-1))
        np.save('TSNE_Feature_' + str(n + 1) + '.npy', TSNE_Feat)

# Statistical Feature Extraction
an = 0
if an == 1:
    for n in range(no_of_dataset):
        data = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Dataset
        Feat = []
        for j in range(data.shape[0]):  # For all datas
            print(j)
            datas = data[j][1:].astype(float)
            h, cp, p, U, mu = hg.pettitt_test(datas)
            if h:
                homo = 1
            else:
                homo = 0
            # $$ Correlation $$#
            corr = np.corrcoef(datas)
            ## varience
            varience = np.var(datas)
            # calculate sample skewness
            ske = skew(datas, axis=0, bias=True)
            # calculate sample kurtosis
            kurto = kurtosis(datas, fisher=False)
            # $$ Contrast $$#
            min = np.min(datas)
            max = np.max(datas)
            contrast = (max - min) / (max + min)
            # $$ Entropy $$#
            ent = scipy.stats.entropy(datas)
            mean = np.mean(datas)
            median = np.median(datas)
            std = np.std(datas)
            Feat.append(np.asarray([min, max, mean, median, std, homo, corr, varience, ske, kurto, contrast, ent]))
        np.save('Statistical_Feature_' + str(n + 1) + '.npy', Feat)

# optimization for Weighted Feature Selection
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Feat1 = np.load('Deep_Feature_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat2 = np.load('TSNE_Feature_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat3 = np.load('Statistical_Feature_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat = [Feat1, Feat2, Feat3]
        Global_Vars.Feat = Feat
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 33
        xmin = matlib.repmat(np.append(np.zeros(Chlen - 3), 0.01 * np.ones(Chlen - 30)), Npop, 1)
        xmax = matlib.repmat(np.append((Feat.shape[1] - 1) * np.ones(Chlen - 3), 0.99 * np.ones(Chlen - 30)), Npop, 1)
        fname = objfun_feat
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = rn.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 25

        print("OOA...")
        [bestfit1, fitness1, bestsol1, time1] = OOA(initsol, fname, xmin, xmax, Max_iter)  # OOA

        print("TFMOA...")
        [bestfit2, fitness2, bestsol2, time2] = TFMOA(initsol, fname, xmin, xmax, Max_iter)  # TFMOA

        print("HGSO...")
        [bestfit3, fitness3, bestsol3, time3] = HGSO(initsol, fname, xmin, xmax, Max_iter)  # HGSO

        print("CMBA...")
        [bestfit4, fitness4, bestsol4, time4] = CMPA(initsol, fname, xmin, xmax, Max_iter)  # CMBA

        print("Proposed..")
        [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Proposed

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]

        np.save('BestSol_Feat_' + str(n + 1) + '.npy', BestSol)

# Optimized Weighted Feature Selection
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Feat1 = np.load('Deep_Feature_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat2 = np.load('TSNE_Feature_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat3 = np.load('Statistical_Feature_' + str(n + 1) + '.npy', allow_pickle=True)
        BestSol = np.load('BestSol_Feat_' + str(n + 1) + '.npy', allow_pickle=True)
        sol = BestSol[4, :]
        feat_col = sol[:round(len(sol) - 3)].astype(np.uint8)
        weight = sol[round(len(sol) - 3):]
        feat1 = Feat1[:, :len(feat_col) - 20]
        feat2 = Feat2[:, 10:len(feat_col) - 10]
        feat3 = Feat3[:, :len(feat_col) - 20]
        Feature = feat1 * weight[0] + feat2 * weight[1] + feat3 * weight[2]
        np.save('Selected_Features_' + str(n + 1) + '.npy', Feature)

# Intrusion Detection
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Feat1 = np.load('Deep_Feature_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat2 = np.load('TSNE_Feature_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat3 = np.load('Statistical_Feature_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat4 = np.load('Selected_Features_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat = [Feat1, Feat2, Feat3, Feat4]
        Tar = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Eval_all = []
        for m in range(len(Feat)):  # for all learning percentage
            EVAL = np.zeros((5, 14))
            per = round(len(Feat[n]) * 0.75)
            Train_Data = Feat[n][:per, :, :]
            Train_Target = Tar[:per, :]
            Test_Data = Feat[n][per:, :, :]
            Test_Target = Tar[per:, :]
            EVAL[0, :], pred1 = Model_DBN(Train_Data, Train_Target, Test_Data, Test_Target)  # DBN Model
            EVAL[1, :], pred2 = Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target)  # RNN model
            EVAL[2, :], pred3 = Model_TCNN(Train_Data, Train_Target, Test_Data, Test_Target)  # TCNN model
            EVAL[3, :] = Model_TCNN_RNN(Train_Data, Train_Target, Test_Data, Test_Target)  #  TCNN and RNN
            EVAL[4, :] = Model_MDDHN_AM(Train_Data, Train_Target, Test_Data, Test_Target)  #  MDDHNAM
            Eval_all.append(EVAL)
        np.save('Eval_all.npy', Eval_all)

plot_results_1()
Plot_Confusion()
plotConvResults()
Plot_ROC_Curve()
