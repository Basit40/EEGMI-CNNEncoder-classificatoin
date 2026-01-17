import numpy as np
from sklearn.covariance import EmpiricalCovariance

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
from scipy.io import loadmat
from scipy.linalg import sqrtm, inv


def euclidean_align (Xtr, Xts):     
     # Xtr: [trial, channel, time]
     # Xts: [trial, channel, time]

    covs=[np.cov(x) for x in Xtr]
    # print('=====',covs[0].shape) # 22X22
    # print('___',np.array(covs).shape) #287X22X22

    C_ref=np.mean(covs,axis=0)

    #print('++++',C_ref.shape) #22X22

    Xtr_aligned=[np.dot(inv(sqrtm(C_ref)),x) for x in Xtr]
    Xts_aligned=[np.dot(inv(sqrtm(C_ref)),x) for x in Xts]

    return np.array(Xtr_aligned), np.array(Xts_aligned)


if __name__=="__main__":
    subj1=loadmat('subjects_data2a/subject_03T.mat')
    subj1_na=loadmat('subjects_data2a_Nocov/subject_03T.mat')
    subj2=loadmat('subjects_data2a/subject_09T.mat')
    subj2_na=loadmat('subjects_data2a_Nocov/subject_09T.mat')

    data1=subj1['Data'][0][0]
    data1_na=subj1_na['Data'][0][0]
    data2=subj2['Data'][0][0]
    data2_na=subj2_na['Data'][0][0]


    print('_____z11_____',data1_na.shape)
    print('_____z22_____',data2_na.shape)

    data1_na=np.transpose(data1_na, [2,0,1])
    data2_na=np.transpose(data2_na, [2,0,1])

    print('_____z11_____',data1_na.shape)
    print('_____z22_____',data2_na.shape)



    xtr_ali, xts_ali=euclidean_align(data1_na,data2_na)

    print('xtr_ali=',xtr_ali.shape)
    print('xts_ali=',xts_ali.shape)