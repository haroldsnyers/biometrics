import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm


def read_img(image_path):
    # load the image and convert it to grayscale
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # ROI, and resize it to a canonical size
    try:
        imagePathStem = str(image_path.stem)
        k = int(imagePathStem[imagePathStem.rfind("_") + 1:][:4]) - 1
    except:
        pass

    return gray, image_path.parent.name


def chi2(hist_a, hist_b, eps=1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum(((hist_a - hist_b) ** 2) / (hist_a + hist_b + eps))

    # return the chi-squared distance
    return d


def compute_similarity_matrix(embedded, dist_metric, labels):
    data = []
    similarity_matrix = np.zeros((len(embedded), len(embedded)))
    for i, img1 in tqdm(enumerate(embedded), total=len(embedded)):
        for j, img2 in enumerate(embedded):
            score = dist_metric(img1, img2)
            if i != j:
                genuine = 1 if labels[i] == labels[j] else 0
                data.append([labels[i], labels[j], genuine, score])
            similarity_matrix[i][j] = score

    df = pd.DataFrame(data, columns=['p1', 'p2', 'genuine', 'score'])
    df['scores_norm'] = min_max_norm(df['score'].values)
    similarity_mat_norm = min_max_norm(similarity_matrix)
    return df, pd.DataFrame(similarity_mat_norm)


def min_max_norm(df_col):
    return (np.max(df_col) - df_col) / (np.max(df_col) - np.min(df_col))


def get_image_db_statistic(db):
    # extract number of samples and image dimensions (for later display)
    n_samples, h, w, n_channels = db.images.shape
    imshape = (h, w, n_channels)

    # count number of individuals
    n_classes = db.target.max() + 1

    n_features = db.data.shape[1]

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_classes: %d" % n_classes)
    print("n_features: %d" % n_features)
    print("imshape : " + str(imshape))

    return imshape, n_samples, n_classes, n_features
