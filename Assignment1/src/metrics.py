import numba as nb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve, DetCurveDisplay, roc_curve, RocCurveDisplay, precision_recall_curve, auc
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


def compute_cmc(similarity_matrix):
    ranked_li = []
    for elem in list(similarity_matrix.columns):
        sorted = similarity_matrix[elem].sort_values(ascending=False)
        print(sorted)
        print()
        previous_val = sorted[0]
        index = 0
        for i, index_ in enumerate(sorted.index):
            # if previous !
            if index_ == elem:
                ranked_li.append(i)
                break
    s = pd.Series(ranked_li).value_counts(normalize=True).sort_index(ascending=True)
    li_ranked = s.cumsum()
    return li_ranked


def compute_auc(precision, recall):
    return auc(precision, recall)


#####################
####### PLOTS  ######
#####################

def plot_roc_curve(fpr, tpr, ax, title_add=""):
    display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    display.plot(ax=ax)
    ax.set_title("Receiver Operating Characteristic (ROC) curves " + title_add)
    ax.set_ylim(0)
    ax.set_xscale("log")


def plot_det_curve(fpr, fnr, ax, title_add=""):
    display = DetCurveDisplay(fpr=fpr, fnr=fnr)
    display.plot(ax=ax)
    ax.set_title("Detection Error Tradeoff (DET) curves " + title_add)


def plot_eer_roc(fpr, tpr, ax, result, title_add=""):
    plot_roc_curve(fpr, tpr, ax, title_add)
    ax.plot(result['fpr'], result['tpr'], 'ro')
    ax.plot([0, 1], [1, 0])


def plot_eer_det(fpr, fnr, ax, result, title_add=""):
    ax.plot(fpr, fnr, label='DET')
    ax.plot(result['fpr'], result['fnr'], 'ro', label="EER")
    ax.plot([0, 1], [0, 1])
    ax.set_title("Detection Error tradeoff (DET) curves for " + title_add)
    ax.set_xlabel("False positive rate (fpr)")
    ax.set_ylabel("False negative rate (fnr)")
    ax.legend()


def plot_recall_precision(recall, precision, ax, title_add=""):
    ax.plot(recall, precision)
    ax.set_title("Precision-recall curve for " + title_add)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")


def plot_score_distribution(ax, imposter, genuine, title):
    bins = np.linspace(0, 1, 150)
    ax.set_title("Raw scores " + title, fontsize=12)
    ax.hist(imposter,
            label='Impostors', density=True,
            color='C1', alpha=0.5, bins=bins, range=(0, 0.5))
    ax.hist(genuine,
            label='Genuine', density=True,
            color='C0', alpha=0.5, bins=bins, range=(0, 0.5))
    ax.legend(fontsize=10)
    ax.set_yticks([], [])
    ax.set_xlabel("Scores")
    ax.set_xlim(0, 0.3)


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
    p.set(xlabel="Decision Threshold", ylabel="",
          title="F1 and accuracy as a function of the decision \n thresholds on the similarity score for " + index)


def plot_cmc(df, axes, index):
    p = sns.lineplot(data=df, ax=axes)
    p.set(xlabel="Rank", ylabel="Recognition rate",
          title="Cumulative Matching Characteristic Curve for " + index)


# Not used
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
