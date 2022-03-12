import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay, DetCurveDisplay


def generate_threshold_values(n: int):
    return ["{:.9f}".format(i) for i in np.arange(0.01, 1, 1/n)]


def generate_dataframe(columns, data_list, index_values=None):
    df = pd.DataFrame(columns=columns)
    for i, elem in enumerate(columns):
        df[elem] = data_list[i]
    if index_values.any():
        df.index = index_values
    return df


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
    if not thresholds.any():
        thresholds = generate_threshold_values(50)

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

        accuracy = (TP + TN)/(TP + TN + FP + FN)
        accuracy_list.append(accuracy)
    return accuracy_list


def plot_roc_curve(fpr, tpr, ax, title_add=""):
    display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    display.plot(ax=ax)
    ax.set_title("Receiver Operating Characteristic (ROC) curves " + title_add)


def plot_det_curve(fpr, fnr, ax, title_add=""):
    display = DetCurveDisplay(fpr=fpr, fnr=fnr)
    display.plot(ax=ax)
    ax.set_title("Detection Error Tradeoff (DET) curves " + title_add)
