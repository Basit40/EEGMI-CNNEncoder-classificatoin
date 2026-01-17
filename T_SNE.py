from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
from scipy.io import loadmat
#from scipy.linalg import sqrtm, inv
from aligning_data import euclidean_align




def t_sne(X,prx, pca=True):

    tsne=TSNE(n_components=2, perplexity=prx, learning_rate=10)

    if pca :
        pca=PCA(n_components=19)
        X=pca.fit_transform(X)
        #Y=pca.fit_transform(Y)

    x_embedded=tsne.fit_transform(X)
    #y_embedded=tsne.fit_transform(Y)


    return x_embedded#,y_embedded
   

if __name__=="__main__":
    subj1=loadmat('subjects_data2a/subject_01T.mat')
    subj2=loadmat('subjects_data2a/subject_02T.mat')
    subj3=loadmat('subjects_data2a/subject_03T.mat')
    subj4=loadmat('subjects_data2a/subject_04T.mat')
    subj5=loadmat('subjects_data2a/subject_05T.mat')
    subj6=loadmat('subjects_data2a/subject_06T.mat')
    subj7=loadmat('subjects_data2a/subject_07T.mat')
    subj8=loadmat('subjects_data2a/subject_08T.mat')
    subj9=loadmat('subjects_data2a/subject_09T.mat')



    subj1_na=loadmat('subjects_data2a_Nocov/subject_01T.mat')
    subj2_na=loadmat('subjects_data2a_Nocov/subject_02T.mat')
    subj3_na=loadmat('subjects_data2a_Nocov/subject_03T.mat')
    subj4_na=loadmat('subjects_data2a_Nocov/subject_04T.mat')
    subj5_na=loadmat('subjects_data2a_Nocov/subject_05T.mat')
    subj6_na=loadmat('subjects_data2a_Nocov/subject_06T.mat')
    subj7_na=loadmat('subjects_data2a_Nocov/subject_07T.mat')
    subj8_na=loadmat('subjects_data2a_Nocov/subject_08T.mat')
    subj9_na=loadmat('subjects_data2a_Nocov/subject_09T.mat')

    data1=subj1['Data'][0][0]
    data2=subj2['Data'][0][0]
    data3=subj3['Data'][0][0]
    data4=subj4['Data'][0][0]
    data5=subj5['Data'][0][0]
    data6=subj6['Data'][0][0]
    data7=subj7['Data'][0][0]
    data8=subj8['Data'][0][0]
    data9=subj9['Data'][0][0]


    data1_na=subj1_na['Data'][0][0]
    data2_na=subj2_na['Data'][0][0]
    data3_na=subj3_na['Data'][0][0]
    data4_na=subj4_na['Data'][0][0]
    data5_na=subj5_na['Data'][0][0]
    data6_na=subj6_na['Data'][0][0]
    data7_na=subj7_na['Data'][0][0]
    data8_na=subj8_na['Data'][0][0]
    data9_na=subj9_na['Data'][0][0]

    #print('_____data1_____',data1.shape)
    #print('_____data1_na__',data1_na.shape)

    
    train=np.concatenate([data9 ,data6  ,data8 ,data4  ,data7  ,data5 ,data3 ,data1 ],axis=0)#.real*
    #train=np.transpose(train, [2,0,1])
    test=data2#.real*1000000
    #test=np.transpose(test, [2,0,1])
    print('_____train__',train.shape)
    print('_____test______',test.shape)

    train_a=train.reshape(train.shape[0],-1).real
    test_a=test.reshape(test.shape[0],-1).real
    print('_____train_a__',train_a.shape)
    print('_____test_a______',test_a.shape)

       


    train_na=np.concatenate([data9_na ,data6_na  ,data8_na ,data4_na  ,data7_na  ,data5_na ,data3_na ,data1_na ],axis=2)#.real*
    train_na=np.transpose(train_na, [2,0,1])
    test_na=data2_na#.real*1000000
    test_na=np.transpose(test_na, [2,0,1])


    train_na_na=train_na.reshape(train_na.shape[0],-1).real
    test_na_na=test_na.reshape(test_na.shape[0],-1).real

    print('_____train_na_na__',train_na_na.shape)
    print('_____test_na_na_____',test_na_na.shape)

    
    train_na_al,test_na_al= euclidean_align (train_na, test_na)  ########## Alignment ############

    print('_____train_na_al__',train_na_al.shape)
    print('_____test_na_al____',test_na_al.shape)
    
    #train_na_al=train_na_al.transpose([2,0,1])
    #test_na_al=test_na_al.transpose([2,0,1])

    print('_____train_na_al_____',train_na_al.shape)
    print('_____test_na_al_____',test_na_al.shape)

    

    train_al_na=train_na_al.reshape(train_na_al.shape[0],-1).real
    test_al_na=test_na_al.reshape(test_na_al.shape[0],-1).real



    print('_____train_na_al_____',train_al_na.shape)
    print('_____test_na_al_____',test_al_na.shape)
        


    train=  train.reshape(train.shape[0],-1).real
    test=  test.reshape(test.shape[0],-1).real

    print('____z01_____',train.shape)
    print('____z02_____',test.shape)
    



    for i in range(1):

        x=t_sne(train_al_na,5,False)
        x2=t_sne(test_al_na,5,False)
        plt.figure(figsize=(6,6))
        plt.scatter(x[:,0],x[:,1],label='train')
        plt.scatter(x2[:,0],x2[:,1],label='test')
        plt.title("Aligned")
        plt.legend()

        y=t_sne(train_na_na,5,False)
        y2=t_sne(test_na_na,5,False)
        plt.figure(figsize=(6,6))
        plt.scatter(y[:,0],y[:,1],label='train')
        plt.scatter(y2[:,0],y2[:,1],label='test')
        plt.title("Not Aligned")
        plt.legend()
    


    for i in range(1):

        xa=t_sne(train_a,5,False)
        x2a=t_sne(test_a,5,False)
        plt.figure(figsize=(6,6))
        plt.scatter(xa[:,0],xa[:,1],label='train')
        plt.scatter(x2a[:,0],x2a[:,1],label='test')
        plt.title("Aligned")
        plt.legend()
      

    plt.show()

