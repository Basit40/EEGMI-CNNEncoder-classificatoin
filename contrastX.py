


#from NNN_EggNet import ssl
from NNN import ssl
from ContrastiveLoss0 import ContrastiveLoss
from ContrastiveLoss4 import ContrastiveLossx


from aligning_data import euclidean_align
from choice_random import INDEX
from transform_rrr import Transform
from T_SNE import t_sne


import torch.nn.functional as F
import torch
from torchinfo import summary
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
from scipy import signal
# import pandas as pd
import tqdm
import mit_utils as utils
# import analytics
import time
import os, shutil

import matplotlib.pyplot as plt
#from mail import mail_it
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


import random

from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

import argparse


plt.close('all') 

print('torch version',torch.__version__)

parser = argparse.ArgumentParser(description='Need for Transformation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# FFparser.add_argument('-d', '--dataset', type=int)
# parser.add_argument('-g', '--gpu_id', type=str, default=0)
parser.add_argument('-F1', '--transform_function_1', type=str)#,metavar='',required=True, help='Transformation 1')
parser.add_argument('-F2', '--transform_function_2', type=str)#,metavar='',required=True, help='Transformation 2')
# parser.add_argument('-e', '--epoch', type=int, default=60)


arg = parser.parse_args()

torch.set_default_tensor_type(torch.FloatTensor)




device = "cuda"  

#save parameters
log_dir = "logs"
model_name = 'sscl'
model_save_dir = '%s/%s_%s' % (log_dir, model_name, time.strftime("%m%d%H%M"))
os.makedirs(model_save_dir, exist_ok=True)

#save pretrain figures
model_name = "A_pretrain_figs"
model_save_figs = '%s/%s' % (log_dir,model_name)
os.makedirs(model_save_figs, exist_ok=True)

model_name = "A_Test_figs"
test_save_figs = '%s/%s' % (log_dir,model_name)
os.makedirs(test_save_figs, exist_ok=True) 

# save testing results
model_name = "A_Results"
test_results_dir = '%s/%s' % (log_dir, model_name)
os.makedirs(test_results_dir, exist_ok=True)

# save logs
log_file = "%s_%s_%s.log" % (arg.transform_function_1, arg.transform_function_2, time.strftime("%m%d%H%M"))
log_templete = {"acc": None, "cm": None,"f1": None,"per F1":None, "epoch":None, }




#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&      start     &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#__________________________ Cutting data________________
data_start=0  #----------------------------------
data_length=550  #--------------------------------

#________________________________________________________________ Motor Imagery  (channles,sampls,trials)

#____________________________ Data Competition IV 2b two classes: 0,1   (left hand, right hand)  with all subjects ok.....

subj1=loadmat(r'C:\Users\aelha\Desktop\ssl-sleepstaging-Copy\subjects_bessel2b_data\subject_01T.mat')  


print(subj1.keys())

Data1=subj1['Data'] 
y1=subj1['Labels']-10  #[0][1].squeeze()

subj2=loadmat(r'C:\Users\aelha\Desktop\ssl-sleepstaging-Copy\subjects_bessel2b_data\subject_02T.mat')  
Data2=subj2['Data']
y2   =subj2['Labels']-10
subj3=loadmat(r'C:\Users\aelha\Desktop\ssl-sleepstaging-Copy\subjects_bessel2b_data\subject_03T.mat') 
Data3=subj3['Data']
y3=subj3['Labels']-10
subj4=loadmat(r'C:\Users\aelha\Desktop\ssl-sleepstaging-Copy\subjects_bessel2b_data\subject_04T.mat')
Data4=subj4['Data']
y4=subj4['Labels']-10
subj5=loadmat(r'C:\Users\aelha\Desktop\ssl-sleepstaging-Copy\subjects_bessel2b_data\subject_05T.mat')
Data5=subj5['Data']
y5=subj5['Labels']-10
subj6=loadmat(r'C:\Users\aelha\Desktop\ssl-sleepstaging-Copy\subjects_bessel2b_data\subject_06T.mat')
Data6=subj6['Data']
y6=subj6['Labels']-10
subj7=loadmat(r'C:\Users\aelha\Desktop\ssl-sleepstaging-Copy\subjects_bessel2b_data\subject_07T.mat')
Data7=subj7['Data']
y7=subj7['Labels']-10
subj8=loadmat(r'C:\Users\aelha\Desktop\ssl-sleepstaging-Copy\subjects_bessel2b_data\subject_08T.mat')
Data8=subj8['Data']
y8=subj8['Labels']-10
subj9=loadmat(r'C:\Users\aelha\Desktop\ssl-sleepstaging-Copy\subjects_bessel2b_data\subject_09T.mat')
Data9=subj9['Data']
y9=subj9['Labels']-10




#_________________________________________________________________________________________________

 #######   DATA COMPETITION IV 2A  (channles,sampls,trials)     with sujbect 4 ok.....


# subj1=loadmat(r'subjcts_bessel2a_data\subject_01T.mat')    
# subj2=loadmat(r'subjcts_bessel2a_data\subject_02T.mat') 
# subj3=loadmat(r'subjcts_bessel2a_data\subject_03T.mat') 
# subj4=loadmat(r'subjcts_bessel2a_data\subject_04T.mat')
# subj5=loadmat(r'subjcts_bessel2a_data\subject_05T.mat')
# subj6=loadmat(r'subjcts_bessel2a_data\subject_06T.mat')
# subj7=loadmat(r'subjcts_bessel2a_data\subject_07T.mat')
# subj8=loadmat(r'subjcts_bessel2a_data\subject_08T.mat')
# subj9=loadmat(r'subjcts_bessel2a_data\subject_09T.mat')



# Data1=subj1['Data']
# Data1=Data1[:,:,data_start:data_length]
# y1  =subj1['Labels'].squeeze()-7
# Data2=subj2['Data']
# Data2=Data2[:,:,data_start:data_length]
# y2   =subj2['Labels'].squeeze()-7
# Data3=subj3['Data']
# Data3=Data3[:,:,data_start:data_length]
# y3   =subj3['Labels'].squeeze()-7
# Data4=subj4['Data']
# Data4=Data4[:,:,data_start:data_length]
# y4   =subj4['Labels'].squeeze()-5
# Data5=subj5['Data']
# Data5=Data5[:,:,data_start:data_length]
# y5   =subj5['Labels'].squeeze()-7
# Data6=subj6['Data']
# Data6=Data6[:,:,data_start:data_length]
# y6   =subj6['Labels'].squeeze()-7
# Data7=subj7['Data']
# Data7=Data7[:,:,data_start:data_length]
# y7   =subj7['Labels'].squeeze()-7
# Data8=subj8['Data']
# Data8=Data8[:,:,data_start:data_length]
# y8   =subj8['Labels'].squeeze()-7
# Data9=subj9['Data']
# Data9=Data9[:,:,data_start:data_length]
# y9   =subj9['Labels'].squeeze()-7



#_________________________________________________________________________________________________


 #  Data 2a four classes: 0,1,2,3  (left hand, right hand, foot, tongue)
 #  Data 2b two classes: 0,1   (left hand, right hand)


#target_class = np.array(['Lh', 'Rh', 'Ft', 'Tg'])   # change number classes=4, number of channels=22  @ NNN

target_class = np.array(['Lh', 'Rh'])              # change number classes=2, number of channels=3  @ NNN



#_________________________________________________________________________________________________


s=9 ## (1-9) Subject Numbers for Validation and Test
subj=f"subject_0{s}T"  #_________________________change subject number
DTt=f"Data{s}"
DT=eval(DTt).real  #_________________________change subject number





LTt=f"y{s}"
LT=eval(LTt)  #_________________________change subject number

LT=LT.squeeze()  #### for datasset 2b 

total_indices=DT.shape[0]

num_indices_for_V = 30 # Example: pick 40 random indices for valid  //////////// For data competition IV 2b: 30
num_indices_for_T = 30 # Example: pick 47 random indices for test


random_indices_V=INDEX(LT, target_class.shape[0], num_indices_for_V)
data_val = DT[random_indices_V,:,:]
label_val=LT[random_indices_V]

remaining_indices = np.setdiff1d(np.arange(total_indices), random_indices_V)  # Indices not selected


DT = DT[remaining_indices,:,:]
LT=LT[remaining_indices]
total_indices=DT.shape[0]


random_indices_T=INDEX(LT,target_class.shape[0],num_indices_for_T)
data_test = DT[random_indices_T,:,:]
label_test=LT[random_indices_T]


remaining_indices = np.setdiff1d(np.arange(total_indices), random_indices_T)  # Indices not selected
DT = DT[remaining_indices,:,:]
LT=LT[remaining_indices]



print("DT>>>>>",DT.shape)
print("data_val>>>>>>>",data_val.shape)
print("data_test>>>>>>",data_test.shape)




## Prepare Training Data (all subjects except subject s)


train_data_list = []
train_label_list = []

for subj_idx in range(1, 10):
    if subj_idx == s:
        continue  # Skip current subject (used for val/test)
    train_data_list.append(eval(f"Data{subj_idx}"))
    train_label_list.append(eval(f"y{subj_idx}"))

orig_x = np.concatenate(train_data_list, axis=0)

y = np.concatenate(train_label_list, axis=1).squeeze() ### for dataset 2b   

# print('orig_x shape before squeeze=',orig_x.shape)
# print('y shape before squeeze=',y.shape)
# print('LT shape after squeeze=',LT.shape)



orig_x=np.concatenate([orig_x,DT],axis=0)
y=np.concatenate     ([y ,LT],axis=0)


# orig_x=DT
# y=LT





x=orig_x.real

#x=np.transpose(x, [2,0,1])
#data_test=np.transpose(data_test, [2,0,1])

print('orig_x shape1=',(x.shape))


print('data_test=',(data_test.shape))



#____________________________________________________ Alignment ____________________________________________

x, data_test=euclidean_align(x,data_test)

#___________________________________________________________________________________________________________

timepoints=len(x[0][0])


print('==========timepoints=',timepoints)


# labels=filename['Data'][0][1]  ## for A01T.mat
# y=labels.squeeze()

#print('Labels=',(y.shape))


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
from sklearn.model_selection import train_test_split

x=(x-np.mean(x,axis=0))/np.std(x,axis=0) #%%% NOrmalization 
data_val=(data_val-np.mean(data_val,axis=0))/np.std(data_val,axis=0)
data_test=(data_test-np.mean(data_test,axis=0))/np.std(data_test,axis=0)




# #__________________________________________________________________________

x_train = torch.tensor(x, dtype=torch.float).to(device)
y_train = torch.tensor(y, dtype=torch.long).to(device)
x_val = torch.tensor(data_val, dtype=torch.float).to(device)
y_val = torch.tensor(label_val, dtype=torch.long).to(device)

x_test= torch.tensor(data_test, dtype=torch.float).to(device)
y_test = torch.tensor(label_test, dtype=torch.long).to(device)

print('==============',x_test.shape)
print('==============',y_test.shape)

data_sscl= np.concatenate((x, data_test),axis=0)
y_sscl=np.concatenate((y, label_test),axis=0)

#data_sscl= data_test
#y_sscl= label_test

data_sscl = torch.tensor(data_sscl, dtype=torch.float).to(device)
y_sscl = torch.tensor(y_sscl, dtype=torch.long).to(device)

print("data_sscl=",data_sscl.shape)
print("y_sscl",y_sscl.shape)


#________________________________________________________________________________
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def save_ckpt(state, is_best, model_save_dir, message='best_w.pth'):
    current_w = os.path.join(model_save_dir, 'latest_w.pth')
    best_w    = os.path.join(model_save_dir, message)
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def transform(x, mode):
        
    x_ = x.cpu().numpy()

    Trans = Transform()
    

    if mode == 'time_warp':
        pieces = random.randint(5,20)
        stretch = random.uniform(1.5,4)
        squeeze = random.uniform(0.25,0.67)
        x_ = Trans.time_warp(x_, 100, pieces, stretch, squeeze)


    elif mode == 'noise':
        factor = random.uniform(10,20)
        #print('xx___',x_.shape)
        x_ = Trans.add_noise(x_,factor)
        #print('xx___',x_.shape)

    elif mode == 'scale':
        x_ = Trans.scaled(x_,[2,4])

    elif mode == 'DC':
        x_ = Trans.DC(x_,[1,4])

    elif mode == 'negate':
        x_ = Trans.negate(x_)
    elif mode == 'hor_flip':
        x_ = Trans.hor_flip(x_)
    elif mode == 'permute':
        pieces = random.randint(5,20)
        #print('inside permute',x_.shape)
        x_ = Trans.permute(x_,pieces)
    elif mode == 'cutout_resize':
        pieces = random.randint(5, 20)
        x_ = Trans.cutout_resize(x_, pieces)
    elif mode == 'cutout_zero':
        pieces = random.randint(4, 10)
        x_ = Trans.cutout_zero(x_, pieces)
    elif mode == 'crop_resize':
        size = random.uniform(0.4,0.8)
        x_ = Trans.crop_resize(x_, size)
    elif mode == 'move_avg':
        n = random.randint(3, 10)
        x_ = Trans.move_avg(x_,n, mode="same")
    #     to test
    elif mode == 'lowpass':
        order = random.randint(3, 10)
        cutoff = random.uniform(5,20)
        #print('===============================',x_.shape)
        x_ = Trans.lowpass_filter(x_, order, [cutoff])
    elif mode == 'highpass':
        order = random.randint(3, 10)
        cutoff = random.uniform(5, 10)
        x_ = Trans.highpass_filter(x_, order, [cutoff])
    elif mode == 'bandpass':
        order = random.randint(3, 10)
        cutoff_l = random.uniform(1, 5)
        cutoff_h = random.uniform(20, 40)
        cutoff = [cutoff_l, cutoff_h]
        x_ = Trans.bandpass_filter(x_, order, cutoff)

    else:
        print("Error ########################")   

    x_ = x_.copy()
    #x_ = x_[:,None,:]

    #print('x__________',x_.shape)
    return x_
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def contrast_loss(x, criterion):

    contrastive_loss=ContrastiveLoss(temp=0.5) ## For All  loss functions

    
    # num = int(x.shape[0] / 2)
    # hidden1, hidden2 = torch.split(x, num)
    # loss, labels, logits_ab=contrastive_loss(hidden1,hidden2)    # loss function 1

    
    loss, labels, logits_ab=contrastive_loss(x,criterion)    # loss functions 0 & 2

    return loss, labels, logits_ab
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



#=====================================================================================================================   Encoder Training  =============================================================
Net = ssl(timepoints,classification=False).to(device)
#summary(Net,input_size=(1,22,1000))

#Net = nn.DistributedDataParallel(Net)  
# from torch.nn.parallel import DistributedDataParallel as DDP
# Net = DDP(Net)

criterion = nn.CrossEntropyLoss().to(device)

batch_size = 64
epochs = 10# Number of Epochs


#train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_dataset = torch.utils.data.TensorDataset(data_sscl, y_sscl)

train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

optimizer = torch.optim.Adam(Net.parameters(), lr=0.001, weight_decay=0.001)

train_acc_list = []
train_err_list = []

iter_per_epoch = np.ceil(x_train.shape[0] / batch_size)  #______x_train.shape[0] == Number of trials
best_acc = -1
best_err = 1

for epoch in range(epochs):                          #________________(Number of Epochs:rounding over all data)
    Net.train()
    loss_sum = 0
    evaluation = []
   
    with tqdm.tqdm(total=iter_per_epoch) as pbar:     #________________(Number of Batches in one epoch)  


        correct=0
        totals=0

        for ii, (Xe, ye) in  enumerate(train_iter):                      


           # print('xssss',len(train_iter)) ###_________________(Number of trials in one epoch)

            #print('Contrast tranin shape',Xe.shape,'oooooooooo') #______X=[BathSizex22x1000] , ye:labels
            trans1 = []
            trans2 = []
                        
            for i in range(Xe.shape[0]): #__________________________________[BatchSize]
                t1= transform(Xe[i], arg.transform_function_1)   #_______ transformation function type
                trans1.append(t1)
                   
                            
            for i in range(Xe.shape[0]):
                t2 = transform(Xe[i], arg.transform_function_2)  #_________ transformation function type
                trans2.append(t2)
            
            #trans0=np.array(Xe)          
            trans1=np.array(trans1)
            trans2=np.array(trans2)
            trans0=Xe.detach().cpu().numpy()

           
            Trans=np.row_stack([trans1,trans2])       #[342, 22, 501]
                 
            trans = torch.tensor(Trans, dtype=torch.float, device=device) 

            tTrans=torch.row_stack([Xe,trans])       #[342, 22, 501]
            #print('trans ===========================',tTrans.shape)  ##[(batchsize*2) x 22 x 1000]    
                 
            output = Net(tTrans)          #__________________________________________________________________________________Encoder output
            #print('+++output ===',output.shape)  ##tensor type [(batchSize*2) xFeatures No.]
              
            L_cnst, lab_con, logit_con = contrast_loss(output, criterion) #____________________________ comtrast_loss function   7777777777777777777777777777777777777777777777777777777777
            #L_cnst, lab_con, logit_con = ContrastiveLossx(output) #____________________________ comtrast_loss function 



            loss_sum += L_cnst
            
                   
            _, logit_prdct = torch.max(logit_con.detach(),1) #predicted  
            #print("tttt",logit_prdct)  ## [199 elements], but not orderd
            #print("tttt",lab_con)  ## [199 elements], but not orderd
            evaluation.append((logit_prdct == lab_con).tolist()) 
            #print('.................>>>>>>',evaluation) 
            #          
            aaa=(logit_prdct == lab_con).sum().item()
            correct+=aaa
            ttt=logit_prdct.size(0)
            totals+=ttt
            run_acc=f"{100*aaa/ttt:.2f}"
                     
            #______________________________________________________________________


                
#____________________________________________________________________________
            optimizer.zero_grad()  ##_________for clear contents     
            L_cnst.backward()  #____________________________________backprobagation_______
            optimizer.step()    
        
            pbar.set_description("Epoch %d, Loss/Batch = %.2f" % (epoch, L_cnst))
            pbar.update(1)

        #print('logits_ab>>>>>>>>>>>>>>\n',logit_con[0:5,0:5])  ## [199x199]
        #plt.show()

        #plt.show()


    evaluation = [item for sublist in evaluation for item in sublist]
    #print('.................>>>>>>',evaluation)


    

    train_acc = sum(evaluation) / len(evaluation)
      
    run_error = 1 - train_acc


    train_acc_list.append(train_acc)
    train_err_list.append(run_error)
    #train_loss_list.append(loss_sum)

   
    print(epoch,"Encoder", " Loss/Epoch= %.2f"%loss_sum, "Error_Cntst: %.2f"% run_error,"Acc_Contst= %.2f"%train_acc)#, "Train_Acc_Cntst=%.3f"%train_acc)
     
    print('======================================================================================================================')
   
    #save file
    state = {"state_dict": Net.state_dict(), "epoch": epoch}
    #print(Net.state_dict().keys())
    # print(Net.state_dict()['temporal_filtering.conv.weight'][0:1])    
    save_ckpt(state, best_err > run_error, model_save_dir)

    best_err = min(best_err, run_error)


#print(Net.state_dict()['temporal_filtering.conv.weight'][0:1])  
#print('Train_loss_list>>>>>>>>>>>>>>>>>>',train_loss_list)
print('\nComplete of  Train Encoder =============================================')
print("Hghst Acc_Amng_epchs:%.2f"% max(train_acc_list),'@ epoch Num:%d'%train_acc_list.index(max(train_acc_list)))
print('=/////////////////////////////////////////////////////////////////////////////////////////////////////////////=')


Trn_err=np.array((train_err_list))
Trn_acc=np.array((train_acc_list))


# plt.plot(Trn_acc)
# plt.plot(Trn_err)

# plt.show()

#==================================================================================================================================== Classifier  Trainging ============================================================

net = ssl(timepoints,classification=True).to(device) 
#print(net)
#summary(net,input_size=(342,22,1000)) #171x2==342

#ccriterion = MultiClassHingeLoss(margin=1.0).to(device)
ccriterion = nn.CrossEntropyLoss().to(device)
#ccriterion = nn.NLLLoss().to(device)

checkpoint = torch.load(os.path.join(model_save_dir,'latest_w.pth'))
# print(checkpoint['state_dict'].keys())

net.load_state_dict(checkpoint['state_dict'])

#print(net.state_dict()['temporal_filtering.conv.weight'][0:1])

#summary(net,input_size=(batch_size,22,1000)) #171x2==342

batch_size =64
epochs_t = 300#

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=True)



optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001) ####################################################

valid_acc_list = []
Val_loss_list=[]
train_acc_list=[]
Trin_loss_list=[]


iter_per_epoch = np.ceil(x_train.shape[0] /batch_size )

best_acc = -1

for epoch in range(epochs_t):            ##_ Number of Epochs
    net.train()  ##______________________invoking the model for training
    loss_sum = 0
    totals=0
    corrects=0 

    evaluation = []
  

    with tqdm.tqdm(total=iter_per_epoch) as pbar:  ##___ Number of Batches in one epoch
       
        for gg, (Xtr, ytr) in enumerate(train_iter):

            # Forward pass
            output = net(Xtr)
            L_trin = ccriterion(output, ytr)

            #Backward pass
            optimizer.zero_grad()
            L_trin.backward()  ##_____________ For backprobagation ____________________________
            optimizer.step()    

            # Track loss and accuracy
            loss_sum += L_trin.item()
            _, predicted = torch.max(output.detach(), 1)                  
            totals+=ytr.size(0)
            corrects+=(predicted == ytr).sum().item()

            # Update progress bar       
            pbar.set_description("Epoch_: %d, loss/batch = %.2f" % (epoch, L_trin.item()))
            pbar.update(1)


    # End of epoch: calculate average
    ave_train_loss=loss_sum/len(train_iter)        
    train_acc=100*corrects/totals


    # Append to lists
    train_acc_list.append(train_acc)
    Trin_loss_list.append(loss_sum)


    print(epoch,":Epoch_Classifier_train.", " Loss_Train/Epoch=%.3f "% ave_train_loss, " Accuracy_Train=%.2f"% train_acc)
    print('------------------------------------------------------------------------------------------------------')
    
 #-------------------------------------------------------------------------------------------------------------------------  Validatoin  Classifier during each epoch ---------------------------------------------
    val_loss = 0
    pred_v = []
    true_v = []
    totalsv=0
    correctsv=0

    with torch.no_grad():  ###_validation with no backprobagation (Evalution)...
        net.eval()   ##________________envoking the model for validation inside each epoch__________
        for Xv, yv in val_iter:
            #print('Valid Classifier shape',X.shape,'++++++++')                
            output = net(Xv)
            L_val = ccriterion(output, yv)
            val_loss += L_val.item()

            _, predictedv = torch.max(output, 1)
            totalsv+=yv.size(0)
            correctsv+=(predictedv == yv).sum().item()
            
            pred_v.extend(predictedv.cpu().numpy())  #append((predictedv).tolist())
            true_v.extend(yv.cpu().numpy())  #append(yv.tolist())#______________________________y_label_val

        
    ######################################################################################            

    ave_val_loss=val_loss/len(val_iter)  #_________________Average Validation Loss
    running_acc=100*correctsv/totalsv


    valid_acc_list.append(running_acc)
    Val_loss_list.append(val_loss)

    print("Loss_val_runninng=%.3f"% ave_val_loss, " Accuracy_val_running =%.2f"% running_acc)
    print('===========================================================================================================')
    
    state = {"state_dict": net.state_dict(), "epoch": epoch}
    save_ckpt(state, best_acc < running_acc, model_save_dir, 'best_w.pth') ## ______Saveing Model best eval_______
    best_acc = max(best_acc, running_acc)




    
    




print('=============================================================================================================')
print('\nTraining  Classifier Complete: ')
#print('--------------',train_acc_list)
print("Train_Accuracy =%.2f"% max(train_acc_list),  '@ Best_Train_Accuracy_epoch Number:',train_acc_list.index(max(train_acc_list)))
print("Valid_Accuracy =%.2f"% max(valid_acc_list),  '@ Best_Valid_Accuracy_epoch Number:',valid_acc_list.index(max(valid_acc_list)))
print('==============================================================================================================')

Trn_loss=np.array(Trin_loss_list)
train_acc=np.array(train_acc_list)

Val_loss=np.array(Val_loss_list)
valid_acc=np.array(valid_acc_list)

# Trn_loss=(Trn_loss-np.mean(Trn_loss,axis=0))/np.std(Trn_loss,axis=0)
# Val_loss=(Val_loss-np.mean(Val_loss,axis=0))/np.std(Val_loss,axis=0)

Trn_loss=(Trn_loss-min(Trn_loss))/(max(Trn_loss)-min(Trn_loss))
Val_loss=(Val_loss-min(Val_loss))/(max(Val_loss)-min(Val_loss))



###Plotting the results
plt.figure(figsize=(6,6))
plt.plot(train_acc, label='Train ')
plt.plot(valid_acc, label='Valid')
plt.title(f"Accuracy. Train Acc: {f"{max(train_acc):.2f}%"}, Val Acc: {f"{max(valid_acc):.2f}%" }")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')  
plt.legend()

plt.figure(figsize=(6,6))
plt.plot(Trn_loss, label='Train ')
plt.plot(Val_loss, label='Valid')
plt.title(f"Losses") 
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()





# =====================================================================  Testing Stage  =======================================================


test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)

nett = ssl(timepoints,classification=True).to(device) 
#
checkpoint = torch.load(os.path.join(model_save_dir,'best_w.pth'))  
epoch_b = checkpoint['epoch']

#print(checkpoint['state_dict'].keys())

nett.load_state_dict(checkpoint['state_dict'])
nett.eval()  ###_____________ Testing stage____________Evaluation _____


#print(nett.state_dict().keys())

# test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
# test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)

pred_v = []
true_v = []
all_features=[]
correct=0
totals=0
with torch.no_grad():      ##_____________ No backprobagation_________
    nett.eval()  ##________________envoking the model for testing__________
    for Xts, yts in test_iter:

        # print('Testing ALL shape',Xts.shape)      #[numExamlps, channls, time] 
        #  
        output = nett(Xts)                        # [NumExampls, NumClasses]
        #print ('====//////====',output.shape)
        _, predicted = torch.max(output, 1)  #indeces of max class via columns
        totals+=yts.size(0)
        correct+=(predicted == yts).sum().item()

        
        pred_v.append((predicted).tolist())
        true_v.append(yts.tolist())


        # pred_v.extend(predicted.cpu().numpy())#append((predicted).tolist())
        # true_v.extend(yts.cpu().numpy())#append(yts.tolist())#______________________________y_label_test
        all_features.extend(output.cpu().numpy())


pred_v     = [item for sublist in pred_v for item in sublist]
true_v     = [item for sublist in true_v for item in sublist]
all_features=np.array(all_features)

#Test_Accuracy = 100*sum(np.array(pred_v) == np.array(true_v)) / len(true_v)
Test_Accuracy = 100*correct/totals


print('\nTestin Results:')
print("Test_accuracy = %.2f"%Test_Accuracy," Data test.:", subj )
print('===================================================================================')



# Plotting the results
prdct_figs=plt.figure(figsize=(6,6))
features_2d=t_sne(all_features,2,False)
for i in range(len(target_class)):
    idx=np.where(np.array(pred_v)==i)[0]
    plt.scatter(features_2d[idx,0],features_2d[idx,1],cmap='bwr', edgecolors='k',label=target_class[i],s=100) # c= oo[idx]
plt.legend( )
plt.title(f"Predicted Features {subj}, Acc.: {Test_Accuracy:.2f}%" )
plt.savefig(os.path.join(test_save_figs, f"prdct_{subj}.png"))
#plt.show()

tru_figs=plt.figure(figsize=(6,6))
for i in range(len(target_class)):
    idx=np.where(np.array(true_v)==i)[0]
    plt.scatter(features_2d[idx,0],features_2d[idx,1],cmap='bwr', edgecolors='k',label=target_class[i],s=100) # c= oo[idx]
plt.legend( )
plt.title(f"True Features {subj} , Acc.: {Test_Accuracy:.2f}%" )
plt.savefig(os.path.join(test_save_figs, f"tru_{subj}.png"))

###================================================================================================= =================================================Evalution Tools ================================================================================= 
def calculate_label_prediction(confMatrix, labelidx):
    
    label_total_sum = confMatrix.sum(axis=0)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    prediction = 0
    if label_total_sum != 0:
        prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return prediction


def calculate_label_recall(confMatrix, labelidx):
    
    label_total_sum = confMatrix.sum(axis=1)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    recall = 0
    if label_total_sum != 0:
        recall = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return recall


def calculate_f1(prediction, recall):
    if (prediction + recall) == 0:
        return 0
    return round(2 * prediction * recall / (prediction + recall), 2)

print('____________________________________________________________________________________________________')
cm=utils.plot_confusion_matrix(true_v, pred_v, target_class, normalize=True, title=None, cmap=plt.cm.Blues)
#
#cmm = confusion_matrix(true_v,pred_v)
#print('============\n',cmm)
plt.savefig(os.path.join(test_save_figs, f"cf{subj}.png"))
# plt.show()
f1_macro = f1_score(true_v, pred_v, average='macro')
print(' ____________________________________________________________________________________________________ ')


i=0
f1 = []
for i in range(len(target_class)):
    r = calculate_label_recall(cm,i)
    p = calculate_label_prediction(cm,i)
    f = calculate_f1(p,r)
    f1.append(f)

log_templete["acc"] = '{:.3%}'.format(Test_Accuracy)
log_templete["epoch"] = epoch_b
#log_templete["cm"] = str(cmm)
log_templete["f1"] = str(f1_macro)
log_templete["per F1"] = str(f1)

log = log_templete

print('______________________________________________The End________________________________________________')     
# 
# 


#save figures
#plt.show()



