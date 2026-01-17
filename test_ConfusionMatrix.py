

# import warnings
# import numpy as np
# from scipy.signal import resample
import mit_utils as utils
# import pywt
# from sklearn.preprocessing import scale
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
# # ===========================================
# warnings.filterwarnings("ignore")
# import torch
import numpy as np
# import time,os
from sklearn.metrics import f1_score
# from torch import nn



import torch

print('_______________________________________________')
print(torch.version.cuda)  # Check CUDA version in PyTorch
print(torch.cuda.is_available())  # Should
print('_______________________________________________')





def calculate_all_prediction(confMatrix):
    '''
    计算总精度：对角线上所有值除以总数
    '''
    total_sum = confMatrix.sum()
    correct_sum = (np.diag(confMatrix)).sum()
    prediction = round(100 * float(correct_sum) / float(total_sum), 2)
    return prediction


def calculate_label_prediction(confMatrix, labelidx):
    '''
    计算某一个类标预测精度：该类被预测正确的数除以该类的总数
    '''
    label_total_sum = confMatrix.sum(axis=0)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    prediction = 0
    if label_total_sum != 0:
        prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return prediction


def calculate_label_recall(confMatrix, labelidx):
    '''
    计算某一个类标的召回率：
    '''
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



# def plot_confusion_matrix(y_true, y_pred, classes,
#                           normalize=False,
#                           title=None,
#                           cmap=plt.cm.Blues):
#     """
#     绘制混淆矩阵图，来源：
#     https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
#     """
#     if not title:
#         if normalize:
#             title = 'Normalized confusion matrix'
#         else:
#             title = 'Confusion matrix, without normalization'

#     cm = confusion_matrix(y_true, y_pred)

#     classes = classes[unique_labels(y_true, y_pred)]
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
 
#     fmt = '.2f'# if normalize else 'd'
#     formatted_cm = np.array([[format(value, fmt) for value in row] for row in cm])

#     print(formatted_cm)

#     fig, ax = plt.subplots()
#    # for i in range(3):
#     #    cm[i,i] = 0
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            xticklabels=classes, yticklabels=classes,
#            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')
    
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
    
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()

    
#     return cm




# Example true and predicted labels
y_true = [0, 1, 2, 2, 0, 1, 2, 0, 1, 2,3,0,3,1,2,0,1,3,1,4,3,1,4,2,2,4]
y_pred = [0, 0, 2, 2, 0, 1, 2, 0, 1, 1,3,1,3,1,2,0,1,3,1,4,3,4,2,4,1,4]

# Define class names
classes = np.array(['Class 0', 'Class 1', 'Class 2','Class 3', 'Class 4'])




log_templete = {"acc": None,
                    "cm": None,
                    "f1": None,
                "per F1":None,
                "epoch":None,
                    }


cm=utils.plot_confusion_matrix(y_true, y_pred, classes,normalize=False,cmap=plt.cm.cividis  )
f1_macro = f1_score(y_true, y_pred, average='macro')



highest_acc=89.23
epoch_b=45


# Plot the confusion matrix
utils.plot_confusion_matrix(y_true, y_pred, classes, normalize=True,cmap=plt.cm.plasma )
plt.show()



i=0
f1 = []
for i in range(3):
    r = calculate_label_recall(cm,i)
    p = calculate_label_prediction(cm,i)
    f = calculate_f1(p,r)
    f1.append(f)





log_templete["acc"] = '{:.3%}'.format(highest_acc)
log_templete["epoch"] = epoch_b
log_templete["cm"] = str(cm)
log_templete["f1"] = str(f1_macro)
log_templete["per F1"] = str(f1)

log = log_templete

print('---------------------------')
print(log)
print('log',(log.keys()))
print('log',(log.values()))



# ss=np.arange(25).reshape(5,-1)

# print(ss)


# bb=ss.astype('float')
# print(bb)

# vv1=ss.sum(axis=1)
# print(vv1)

# vv2=ss.sum(axis=1)[:,np.newaxis]
# print(vv2)

# vv3=ss.sum(axis=1)[np.newaxis,:]
# print(vv3)


# vv4=vv2.T
# print(vv4.shape)


# print(bb/vv2)