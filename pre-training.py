import numpy as np
import pywt
import torch
from sklearn.metrics import classification_report, confusion_matrix
from skorch import NeuralNetClassifier

from main import MyModule, load_data
import matplotlib.pyplot as plt
import seaborn as sn 

if __name__ == "__main__":
    sampling_rate = 360

    wavelet = "mexh"  # mexh, morl, gaus8, gaus4
    scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101, 1)

    (x1_train, x2_train, y_train, groups_train), (x1_test, x2_test, y_test, groups_test) = load_data(
        wavelet=wavelet, scales=scales, sampling_rate=sampling_rate)

    net = NeuralNetClassifier(
        MyModule,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    net.initialize()
    net.load_params(f_params="./models/model_{}.pkl".format(wavelet))

    y_true, y_pred = y_test, net.predict({"x1": x1_test, "x2": x2_test})

    # print("PRE TRAIN - Confusion Matrix :")\
    
    # print(confusion_matrix(y_true, y_pred))
    
    # confusion_mtrx=confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(10,8))
    # sn.heatmap(confusion_mtrx,annot=True,fmt='d',cmap='Blues')
    # plt.show()

    # print("PRE TRAIN - Classifiation Report :")
    # print(classification_report(y_true, y_pred, digits=4))
    

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred == i)
        roc_auc[i] = np.trapz(tpr[i], fpr[i])

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange', lw=lw, label='Class N (area = %0.2f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], color='green', lw=lw, label='Class S  (area = %0.2f)' % roc_auc[1])
    plt.plot(fpr[2], tpr[2], color='blue', lw=lw, label='Class V  (area = %0.2f)' % roc_auc[2])
    plt.plot(fpr[3], tpr[3], color='red', lw=lw, label='Class F (area = %0.2f)' % roc_auc[3])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    np.save("y_true.npy", y_true)
    np.save("y_pred.npy", y_pred)
