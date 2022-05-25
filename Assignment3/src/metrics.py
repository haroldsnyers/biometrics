import logging

import numba as nb
import numpy as np
import pandas as pd
import seaborn as sns
from numba import prange
from sklearn.metrics import det_curve, DetCurveDisplay, roc_curve, RocCurveDisplay, precision_recall_curve, auc

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


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
    logging.info('Computing precision recall')
    precision, recall, thresholds_values = precision_recall_curve(y_true=score, probas_pred=y_true)
    logging.info('End precision recall computation curve')
    precision, recall = precision[:-1], recall[:-1]
    logging.info('End precision recall computation')
    f1 = 2 * (precision * recall) / (precision + recall)
    logging.info('End f1')
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

    for x in prange(len(thresholds)):
        t = thresholds[x]
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
    logging.info('Computing roc curve')
    fpr, tpr, threshold = roc_curve(score, y_true, pos_label=None, sample_weight=None)
    logging.info('End roc curve computation')
    fnr = 1 - tpr
    return fpr, fnr, tpr, threshold,


def compute_cmc(similarity_matrix):
    ranked_li = []
    for elem in list(similarity_matrix.columns):
        sorted = similarity_matrix[elem].sort_values(ascending=False)
        # print(sorted)
        # print()
        for i, index_ in enumerate(sorted.index):
            if index_ == elem:
                ranked_li.append(i)
                break
    print(ranked_li)
    s = pd.Series(ranked_li).value_counts(normalize=True).sort_index(ascending=True)
    li_ranked = s.cumsum()
    if len(li_ranked) == 1:
        li_ranked.loc[1] = 1
    print(ranked_li)
    return li_ranked


def compute_auc(precision, recall):
    return auc(precision, recall)


def get_f1_and_acc_dataframe(genuine, imposter, score, y_true):
    """Plot F1 and accuracy as a function of the decision thresholds on the similarity score."""
    # Hint: evaluating for ± 50 threshold values should suffice
    precision, recall, f1, thresholds_values = compute_precision_recall_f1(score, y_true, True)
    logging.info('Computing accuracy')
    accuracy = calculate_accuracy(genuine.to_numpy(), imposter.to_numpy(), thresholds_values)
    logging.info('Generate dataframe')
    df_classification_metrics = generate_dataframe(
        columns=['f1', 'acc'], data_list=[f1, accuracy], index_values=thresholds_values)
    logging.info('dataframe generated')
    return df_classification_metrics, precision, recall


#####################
####### PLOTS  ######
#####################

def plot_roc_curve(fpr, tpr, ax, title_add=""):
    display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    display.plot(ax=ax)
    ax.set_title("Receiver Operating Characteristic (ROC) curves " + title_add)
    ax.set_ylim(0)
    # ax.set_xscale("log")


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
    # ax.set_yticks([], [])
    ax.set_xlabel("Scores")
    # ax.set_xlim(0, 0.3)


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


def plot_sim_matrix(matrix, ax, desc):
    # Draw a heatmap with the numeric values in each cell
    sns.heatmap(matrix, linewidths=.5, ax=ax, cmap="gray")
    ax.set_title('Similarity matrix heatmap for ' + str(desc))
