import numba as nb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve, DetCurveDisplay, roc_curve, RocCurveDisplay, precision_recall_curve
import seaborn as sns


def generate_threshold_values(n: int):
    return ["{:.9f}".format(i) for i in np.arange(0.01, 1, 1 / n)]


def generate_dataframe(columns, data_list, index_values=None):
    df = pd.DataFrame(columns=columns)
    for i, elem in enumerate(columns):
        df[elem] = data_list[i]
    if index_values is not None:
        df.index = index_values
    return df


def compute_precision_recall_f1(score, y_true, thresholds=False):
    precision, recall, thresholds_values = precision_recall_curve(y_true=score, probas_pred=y_true)

    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * (precision * recall) / (precision + recall)
    if thresholds:
        return precision, recall, f1, thresholds_values
    else:
        return precision, recall, f1


@nb.njit(fastmath=True)
def calculate_accuracy(genuine, imposter, thresholds=None):
    """
    TP : genuine user correctly identified
    FN : genuine user incorrectly rejected
    TN : imposter correctly rejected
    FP : imposter passes as a genuine user

    :param genuine: genuine score
    :param imposter: imposter score
    :param thresholds: threshold values to calculate accuracy from (provided or generated)
    :return: accuracy list in function of threshold values
    """
    accuracy_list = []
    # if not thresholds.any():
    #     thresholds = generate_threshold_values(50)

    for t in thresholds:
        TP, FP, TN, FN = 0, 0, 0, 0
        for score in genuine:
            if score >= t:
                TP += 1
            else:
                FN += 1

        for score in imposter:
            if score >= t:
                FP += 1
            else:
                TN += 1

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracy_list.append(accuracy)
    return accuracy_list


def compute_fpr_fnr_tpr_from_det_curve(score, y_true):
    fpr, fnr, threshold = det_curve(score, y_true, pos_label=None, sample_weight=None)
    tpr = 1 - fnr
    return fpr, fnr, tpr, threshold,


def compute_fpr_fnr_tpr_from_roc_curve(score, y_true):
    fpr, tpr, threshold = roc_curve(score, y_true, pos_label=None, sample_weight=None)
    fnr = 1 - tpr
    return fpr, fnr, tpr, threshold,


#####################
####### PLOTS  ######
#####################

def plot_roc_curve(fpr, tpr, ax, title_add=""):
    display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    display.plot(ax=ax)
    ax.set_title("Receiver Operating Characteristic (ROC) curves " + title_add)
    ax.set_ylim(0)


def plot_det_curve(fpr, fnr, ax, title_add=""):
    display = DetCurveDisplay(fpr=fpr, fnr=fnr)
    display.plot(ax=ax)
    ax.set_title("Detection Error Tradeoff (DET) curves " + title_add)


def plot_eer_roc(fpr, tpr, ax, result, title_add=""):
    plot_roc_curve(fpr, tpr, ax, title_add)
    ax.plot(result['fpr'], result['tpr'], 'ro')
    ax.plot([0, 1], [1, 0])


def plot_eer_det(fpr, fnr, ax, result, title_add=""):
    ax.plot(fpr, fnr)
    ax.plot(result['fpr'], result['fnr'], 'ro')
    ax.plot([0, 1], [0, 1])


def plot_score_distribution(ax, imposter, genuine, title):
    ax.set_title("Raw scores " + title, fontsize=12)
    ax.hist(imposter,
            label='Impostors', density=True,
            color='C1', alpha=0.5, bins=50)
    ax.hist(genuine,
            label='Genuine', density=True,
            color='C0', alpha=0.5, bins=50)
    ax.legend(fontsize=10)
    ax.set_yticks([], [])


def plot_decision_threshold_far_frr(df, index, axes):
    p = sns.lineplot(data=df, ax=axes)
    p.set(
        xlabel="Decision Threshold", ylabel="Error Rate %",
        title="FAR and FRR as a function of the decision threshold for " + index)
    # idx = np.argwhere(np.diff(np.sign(df_roc_curves['fpr'] - df_roc_curves['fnr']))).flatten()
    # index = df_roc_curves.index.values[idx]
    # plt.plot(df_roc_curves.index.values[idx], df_roc_curves['fpr'][index], 'ro')


def plot_decision_threshold_f1_acc(df, index, axes):
    p = sns.lineplot(data=df, ax=axes)
    p.set(xlabel="Decision Threshold", ylabel="%",
          title="F1 and accuracy as a function of the decision thresholds on the similarity score for " + index)


@nb.njit(fastmath=True, parallel=True)
def compute_similarity_matrix(array):
    K = np.zeros((array.shape[0], array.shape[0]))
    print(K)
    for i in nb.prange(array.shape[0]):
        x_i = array[i]
        for j in range(array.shape[0]):
            x_j = array[j]
            K[i, j] = np.exp(-np.linalg.norm(x_i - x_j, 2) ** 2)

    return K
