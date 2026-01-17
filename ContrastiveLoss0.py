import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
#from sklearn.metrics.pairwise import euclidean_distances

from NNN import ssl





def plotvect (z11,z22):   ## Draw vectors in 2D
    #z11=torch.tensor(z11, dtype=torch.float, device="cuda")        
    #z22=torch.tensor(z22, dtype=torch.float, device="cuda")
    #print('aaaaaa',z11.shape)

    print('bbbbbb',z11.shape)
    print('bbbbbbb',z22)
    o=np.array([[0,0],[0,0],[0,0],[0,0]])
    z11=np.array(z11.cpu())
    z22=np.array(z22.cpu())
    print('pppppppppp',z11.shape)
    print('pppppppppp',z22)    
    plt.figure(figsize=(6,6))
    for i in range(len(z11)):
        plt.quiver(*o[i],*z11[i],angles='xy', scale=1 ,scale_units='xy',color='r')
        plt.quiver(*o[i],*z22[i],angles='xy',scale=1 ,scale_units='xy',color='g')    
    plt.xlim([-1.25,1.25])
    plt.ylim([-1.25,1.25])
    plt.grid()
    #plt.quiver(x,y,u,v, color='red',scale=10, pivot='middle',angles='xy')

   

class ContrastiveLoss(nn.Module):

    def __init__(self,temp):
        super(ContrastiveLoss,self).__init__()

        self.temperature=temp



    def forward(self,z,criterion):
# 
        LARGE_NUM = 1e9
        temperature = 0.5  
        # print('___________z==',(z.shape))  #174X384 //  [(199*2) x 384] [patchsize*2, features..f(z)]
        z = F.normalize(z, dim=-1).to('cuda')
        

        #print('z======',(z.shape))  ##[(87*2)x384]  // [(199*2)x384]  [64*2, features]

        num = int(z.shape[0] / 3)

        #print('+++++++++++',num)
        
        hidden0, hidden1, hidden2 = torch.split(z, num)
        # print('hidd0========\n',(hidden0.shape))  ## [87X384] // [199x384] //[171x224] // [trials.trnsform1 X features(f(z))] 
        # print('hidd1========\n',(hidden1.shape))  ## [87X384] // [199x384] //[171x224] // [trials.trnsform1 X features(f(z))] 
        # print('hidd2========\n',(hidden2.shape))  ## [87X384] // [199x384] //[171x224] // [trials.trnsform2 X features(f(z))]     


            # yee= ye.detach().cpu().numpy()  #_________________________ye:labels
            # clss0=np.where(yee==0,1,0)  #_________________________class 0 # Binary mask for class 0 (1 if yee == 0, else 0)
            # clss1=np.where(yee==1,1,0)  #_________________________class 1
            # clss2=np.where(yee==2,1,0)  #_________________________class 2
            # clss3=np.where(yee==3,1,0)  #_________________________class 3
            # print('clss0.sum()==',clss0.sum())  ## Sum of class 0 instances in the batch
            # print('clss1.sum()==',clss1.sum())  ## Sum of class 1 instances in the batch
            # print('clss2.sum()==',clss2.sum())  ## Sum of class 2 instances in the batch
            # print('clss3.sum()==',clss3.sum())  ## Sum of class 3 instances in the batch 
          



        labels = torch.arange(0,num).to(z.device)  # Creating Seudo Labels [0,1,2,...,86]  // [0,1,2,..,198]//[0,1,...,170]
        #print('Lable_________\n',labels)
        
        # hidden0=F.normalize(hidden0,p=2,dim=1).to('cuda')
        # hidden1=F.normalize(hidden1,p=2,dim=1).to('cuda')
        # hidden2=F.normalize(hidden2,p=2,dim=1).to('cuda')
        #print('Hidd1\n',n_h1.shape)
        #print('Hidd2\n',n_h2.shape)
        #plotvect(hidden1,hidden2)

        masks = F.one_hot(torch.arange(0,num), num).to(z.device)  ## [87X87]  // [199x199]//[171x171]// (just like unit array)
        #print('masks',masks.shape)  ## 87X87
        
        logits_0 = (torch.matmul(hidden0, hidden0.T))/temperature # square matrix ## dot product approximate cosine similarity
        logits_00 = logits_0 - masks * LARGE_NUM
        logits_a = (torch.matmul(hidden1, hidden1.T))/temperature # square matrix ## dot product approximate cosine similarity
        logits_aa = logits_a - masks * LARGE_NUM
        logits_b = (torch.matmul(hidden2, hidden2.T))/temperature
        logits_bb = logits_b - masks * LARGE_NUM

        logits_ab = (torch.matmul(hidden1, hidden2.T))/temperature
        logits_ba = (torch.matmul(hidden2, hidden1.T))/temperature

        logits_a0 = (torch.matmul(hidden1, hidden0.T))/temperature
        logits_b0 = (torch.matmul(hidden2, hidden0.T))/temperature
        
        loss_01   = criterion(torch.cat([logits_a0, logits_00,logits_aa,logits_bb], 1),labels)
        loss_02   = criterion(torch.cat([logits_b0, logits_00,logits_aa,logits_bb], 1),labels)
        #loss_a = criterion(torch.cat([logits_ab, logits_aa], 1),labels)   ## Losses of one batch for T1   
        #loss_b = criterion(torch.cat([logits_ba, logits_bb], 1),labels) ## _________[199x (199*2)]
        
        #print('00000000',loss_a)  ## scalar
        #print('00000000',loss_b)  ## scalar        

        #xxx=torch.cat([logits_ab, logits_aa], 1)+torch.cat([logits_ba, logits_bb], 1)
        #print('xxx=\n',xxx)
        # zzz=criterion(xxx,labels)
        # print('zzz=\n',zzz)   
        # yyy=xxx.diagonal()
        # print('000000000000\n',yyy.mean())
        
        #loss = ( loss_a + loss_b)/2

        loss = ( loss_01+loss_02)/2#+ loss_a + loss_b)#/4

        #____________________________________________________________________________________________

        #def euclidean_distances(v1,v2):
           # return torch.dist(v1,v2, p=2)

        #m=5
        #d1=euclidean_distances(hidden0,hidden1)
        #d2=euclidean_distances(hidden0,hidden2)
        #d3=euclidean_distances(hidden1,hidden2)

        #Loss=0.5*((d1+d2)**2)+0.5*(max(0,m-d3)**2) ## Euclidean distance
        #loss=max(0,d1+d2-d3+m)  ## Triple distance
        #Triple_loss=nn.TripletMarginWithDistanceLoss(margin=.1)
        #loss=Triple_loss(logits_00,logits_a0,logits_ba)

        return loss, labels, logits_ab

if __name__=="__main__":
    batch_size=512
    embedding_dim=112

    criterion = nn.CrossEntropyLoss().to('cuda')    
        
    #______________________________________________________________________________

    # Generate a dataset
    X, y = make_classification(n_samples=1536, n_features=112, n_classes=4, n_clusters_per_class=1,n_informative=2, weights=[0.25, 0.25, 0.25,0.25],random_state=42)

    # Plot the dataset
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Synthetic Classification Data")
    plt.show()

    #check the original class distribution
    print("\n--- Original class distribution")
    unique, counts=np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Class{u}:{c}samples({c/len(y)*100:.2f}%)")
    print("\n--- Random Split (without stratify)---")
    x_train_rand, x_test_rand, y_train_rand, y_test_rand=train_test_split(X,y,test_size=0.66, random_state=42)

    # Checking class distribution in random split
    unique_train, counts_train = np.unique(y_train_rand, return_counts=True)
    unique_test, counts_test = np.unique(y_test_rand, return_counts=True)

    print("\nTraining Set Distribution:")
    for u, c in zip(unique_train, counts_train):
        print(f"Class {u}: {c} samples ({c/len(y_train_rand)*100:.2f}%)")

    print("\nTesting Set Distribution:")
    for u, c in zip(unique_test, counts_test):
        print(f"Class {u}: {c} samples ({c/len(y_test_rand)*100:.2f}%)")

    # Stratified Split (ensuring class proportions remain consistent)
    print("\n--- Stratified Split ---")
    X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(X, y, test_size=0.66, stratify=y, random_state=42)

    # Checking class distribution in stratified split
    unique_train_strat, counts_train_strat = np.unique(y_train_strat, return_counts=True)
    unique_test_strat, counts_test_strat = np.unique(y_test_strat, return_counts=True)

    print("\nStratified Training Set Distribution:")
    for u, c in zip(unique_train_strat, counts_train_strat):
        print(f"Class {u}: {c} samples ({c/len(y_train_strat)*100:.2f}%)")

    print("\nStratified Testing Set Distribution:")
    for u, c in zip(unique_test_strat, counts_test_strat):
        print(f"Class {u}: {c} samples ({c/len(y_test_strat)*100:.2f}%)")


    

    Trans=np.concatenate([X_train_strat,X_train_strat,X_train_strat],0)
    #________________________________________________________________________________  
    

   # trans = np.array(Trans)  ###___________________________________________- This was np.concatenate(trans)
    #trans=trans[:,None,:,:]

    print('____Input___',Trans.shape)

    
    x = torch.tensor(Trans, dtype=torch.float, device="cuda")


    # timepoints=x.shape[2]
    # print('==========timepoints=',timepoints)
    # kkk
    # Net = ssl(timepoints,classification=False).to('cuda')

    

    contrastive_loss=ContrastiveLoss(temp=0.5)

    

    loss, labels, logits=contrastive_loss(x,criterion)

    #loss=np.array(loss)

    print("Contrastive Loss:", loss)
    print("labels:", labels.shape)
    print('logits=',logits.shape)

    plt.show()
