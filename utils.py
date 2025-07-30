import numpy as np
import torch
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, \
    matthews_corrcoef, roc_auc_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import label_binarize

def sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred)


def specificity(y_true, y_pred):
    tn, fp = confusion_matrix(y_true, y_pred)[0]
    if tn + fp != 0:
        return tn / (tn + fp)
    else:
        return 1


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def plot_confusion_matrix(yv, pred, tag):
    confusion = confusion_matrix(yv, pred)
    print(f"Confusion Matrix in Fold {tag}: \n{confusion}")
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for Fold {tag}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def plot_roc_curve(y_val, pred, tag):

    y_scores = pred[:, 1].detach().numpy().flatten()

    fpr, tpr, thresholds = roc_curve(y_val, y_scores)

    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='#800000', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

    plt.xlabel('False Positive Rate', fontsize=15, fontname='Arial')
    plt.ylabel('True Positive Rate', fontsize=15, fontname='Arial')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.xticks(fontsize=15, fontname='Arial')
    plt.yticks(fontsize=15, fontname='Arial')

    plt.show()
    roc_data = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'Thresholds': thresholds
    })

    roc_data.to_csv(f"./roc_curve/roc_curve_{tag}.csv", index=False)

def plot_ovr_roc_curve(y_val, pred, tag):
    n_classes = pred.shape[1]
    y_val_bin = label_binarize(y_val, classes=np.arange(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()


    all_roc_data = []

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        

        roc_data = pd.DataFrame({
            'False Positive Rate': fpr[i],
            'True Positive Rate': tpr[i],
            'Class': i,
        })
        all_roc_data.append(roc_data)


    combined_roc_data = pd.concat(all_roc_data, ignore_index=True)
    

    combined_roc_data.to_csv(f'./ovr_roc_curve/roc_curve_ovr_{tag}.csv', index=False)


    plt.figure()
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for 3-Class(One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.show()


def binary(yv, predict, tag):
    fpr, tpr, th = roc_curve(yv, predict)
    pred = np.ones(len(predict))
    for i in range(len(predict)):
        if predict[i] < th[np.argmax(tpr - fpr)]: pred[i] = 0.0
    
    res = [
        roc_auc_score(yv, predict),
        sensitivity(yv, pred),
        specificity(yv, pred),
        precision_score(yv, pred),
        accuracy_score(yv, pred),
        f1_score(yv, pred),
        matthews_corrcoef(yv, pred)
    ]
    return res


def neighborhood(feat, k, spec_ang=False):
    # compute C

    featprod = np.dot(feat.T, feat)

    smat = np.tile(np.diag(featprod), (feat.shape[1], 1))
    if spec_ang:
        dmat = 1 - featprod / np.sqrt(smat * smat.T)  # 1 - spectral angle
    else:
        dmat = smat + smat.T - 2 * featprod
    dsort = np.argsort(dmat)[:, 1:k + 1]
    C = np.zeros((feat.shape[1], feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i, j] = 1.0

    return C



def normalized(wmat):

    deg = np.diag(np.sum(wmat, axis=0))
    # D^-0.5
    degpow = np.power(deg, -0.5)

    degpow[np.isinf(degpow)] = 0
    # W= D^-0.5 * W * D^-0.5
    W = np.dot(np.dot(degpow, wmat), degpow)
    return W



def norm_adj(feat):
    C = neighborhood(feat.T, k=6, spec_ang=False)
    norm_adj = normalized(C.T * C + np.eye(C.shape[0]))
    g = torch.from_numpy(norm_adj).float()
    return g


def adjacency_to_edge_index(adj_matrix):

    rows, cols = torch.nonzero(adj_matrix, as_tuple=True)
    

    edge_index = torch.stack([rows, cols], dim=0)
    
    return edge_index