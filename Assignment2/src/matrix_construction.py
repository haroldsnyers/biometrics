import pandas as pd
import numpy as np
import cv2

from scipy.stats import hmean
from sklearn.metrics.pairwise import pairwise_distances
from src.image_processing import detect_keypoints, brute_force_matcher, compute_local_descriptor, warp_points, get_reduced_set, estimate_affine_transform_by_kps


distance_sklearn = lambda x, y, dist_fct_n: 1/pairwise_distances(x, y, metric=dist_fct_n).mean()


def construct_similarity_table(org_img, img_db, labels, dist_func, **kwargs):
    """
    Constructs a similarity table [ids, scores]
    :param org_img: image to compare to
    :param img_db: image database
    :param labels: image label (or id)
    :param dist_func: function that computes the distance between two images
    :param kwargs: extra arguments for distances functions
    """
    data = []
    for i, img in enumerate(img_db):
        data.append(
            [labels[i],
             dist_func(org_img, img, **kwargs)])
    assert (len(data) == len(img_db))
    return pd.DataFrame(data, columns=['id', 'score'])


def local_img_similarity(matches, metric, top_n=30):
    # feature distance

    # given these matches we have to come up with a metric, note that some of the matches will be successful
    # while others will be faulty, it is important that we only take reliable matches as to not skew the similarity
    dist = [i.distance for i in matches]
    dist.sort()
    print(dist)
    if top_n is not None:
        best_n_matches = dist[:top_n]
        if metric == 'mean':
            scores =1/(np.mean(best_n_matches)+np.finfo(float).eps)
        elif metric == 'sum':
            scores =1/(np.sum(best_n_matches)+np.finfo(float).eps)
        elif metric == 'harmonic_mean':
            scores =1/(hmean(best_n_matches)+np.finfo(float).eps)
    else:
        best_matches = []
        for i in dist:
            if i < 30:
                best_matches.append(i)
        scores = len(best_matches)
    return scores


def local_similarity(masked_fp1, masked_fp2, detector=cv2.ORB_create(), kp_erosion_ksize=(5, 5), metric='mean',
                     top_n=30):
    # separate the segmentation from the image
    fp1, mask1 = masked_fp1[..., 0], masked_fp1[..., 1]
    fp2, mask2 = masked_fp2[..., 0], masked_fp2[..., 1]

    # detect the keypoints
    kp1 = detect_keypoints(fp1, mask1, detector, kernel_size=kp_erosion_ksize)
    kp2 = detect_keypoints(fp2, mask2, detector, kernel_size=kp_erosion_ksize)

    # compute descriptor for each keypoints
    try:
        kp1, local_des1 = compute_local_descriptor(fp1, kp1, detector)
        kp2, local_des2 = compute_local_descriptor(fp2, kp2, detector)
    except:
        brief_detector = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp1, local_des1 = compute_local_descriptor(fp1, kp1, brief_detector)
        kp2, local_des2 = compute_local_descriptor(fp2, kp2, brief_detector)

    # find matches between keypoints based on local feature descriptor
    try:
        local_k1_k2_matches = brute_force_matcher(local_des1, local_des2, cv2.NORM_HAMMING)
    except:
        local_k1_k2_matches = brute_force_matcher(local_des1, local_des2, cv2.NORM_L2)

    # distances between matches that are below threshold
    return local_img_similarity(local_k1_k2_matches, metric, top_n)


def global_img_similarity(matches, reg_kp1, kp2, dist_fct_n, thresholds):
    kp1_reduced,kp2_reduced = get_reduced_set(matches, reg_kp1, kp2)
    best_scores = []
    for i, match in enumerate(matches):
        score = distance_sklearn(kp1_reduced[i].reshape(-1, 1), kp2_reduced[i].reshape(-1, 1), dist_fct_n)
        if score < thresholds[dist_fct_n]:
            best_scores.append(score)
    best_scores = np.array(best_scores)
    # if len(best_scores) > 0:
    #     print("highest score ", best_scores.max(), best_scores.argmax())
    #     print("lowest score ", best_scores.min(), best_scores.argmin())
    # print("lower than ", len(best_scores))

    return len(best_scores)


def global_similarity(masked_fp1, masked_fp2, dist_fct_n='euclidean', detector=cv2.ORB_create(),
                      kp_erosion_ksize=(5, 5), thresholds=None):
    # separate the semgentation from the image
    fp1, mask1 = masked_fp1[..., 0], masked_fp1[..., 1]
    fp2, mask2 = masked_fp2[..., 0], masked_fp2[..., 1]

    # detect the keypoints
    kp1 = detect_keypoints(fp1, mask1, detector, kernel_size=kp_erosion_ksize)
    kp2 = detect_keypoints(fp2, mask2, detector, kernel_size=kp_erosion_ksize)

    # compute descriptor for each keypoints
    try:
        kp1, local_des1 = compute_local_descriptor(fp1, kp1, detector)
        kp2, local_des2 = compute_local_descriptor(fp2, kp2, detector)
    except:
        brief_detector = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp1, local_des1 = compute_local_descriptor(fp1, kp1, brief_detector)
        kp2, local_des2 = compute_local_descriptor(fp2, kp2, brief_detector)

    # find matches between keypoints based on local feature descriptor
    try:
        local_k1_k2_matches = brute_force_matcher(local_des1, local_des2, cv2.NORM_HAMMING)
    except:
        local_k1_k2_matches = brute_force_matcher(local_des1, local_des2, cv2.NORM_L2)

    # get source and target index for each match
    local_pt_source, local_pt_target = np.array([
        (match.queryIdx, match.trainIdx) for match in local_k1_k2_matches]).T

    # use matching keypoints to estimate an affine transform between keypoints
    M, inliers = estimate_affine_transform_by_kps(
        cv2.KeyPoint.convert(kp1[local_pt_source]),
        cv2.KeyPoint.convert(kp2[local_pt_target]))

    # if no inliers can be found
    if M is None:
        return 0

    # warp the keypoints according to the found transform
    kp1_reg = warp_points(cv2.KeyPoint.convert(kp1), M)

    # subset the keypoints, inliers are considered good keypoints
    # since they were used in finding the transformation
    global_k1_k2_matches = local_k1_k2_matches[inliers == 1]

    # compute global similarity based aligned matching global keypoints
    return global_img_similarity(global_k1_k2_matches, kp1_reg, kp2, dist_fct_n, thresholds)


def hybrid_img_similarity(matches, kp1_reg, kp2, dist_fct_n, thresholds):
    kp1_reduced,kp2_reduced = get_reduced_set(matches, kp1_reg, kp2)
    best_scores = []
    for i, match in enumerate(matches):
        score = distance_sklearn(kp1_reduced[i].reshape(-1, 1), kp2_reduced[i].reshape(-1, 1), dist_fct_n)
        # if sift : 100, otherwise 55
        print('dist', match.distance)
        # print('score', score)
        if score < thresholds[dist_fct_n] and match.distance < 100:
            best_scores.append(score)
    best_scores = np.array(best_scores)
    if len(best_scores) > 0:
        print("highest score ", best_scores.max(), best_scores.argmax())
        print("lowest score ", best_scores.min(), best_scores.argmin())
    print("lower than ", len(best_scores))

    return len(best_scores)


def hybrid_similarity(masked_fp1, masked_fp2, dist_fct_n=None, detector=cv2.ORB_create(), kp_erosion_ksize=(5, 5),
                      thresholds=None):
    # separate the semgentation from the image
    fp1, mask1 = masked_fp1[..., 0], masked_fp1[..., 1]
    fp2, mask2 = masked_fp2[..., 0], masked_fp2[..., 1]

    # detect the keypoints
    kp1 = detect_keypoints(fp1, mask1, detector, kernel_size=kp_erosion_ksize)
    kp2 = detect_keypoints(fp2, mask2, detector, kernel_size=kp_erosion_ksize)

    # compute descriptor for each keypoints
    try:
        kp1, local_des1 = compute_local_descriptor(fp1, kp1, detector)
        kp2, local_des2 = compute_local_descriptor(fp2, kp2, detector)
    except:
        brief_detector = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp1, local_des1 = compute_local_descriptor(fp1, kp1, brief_detector)
        kp2, local_des2 = compute_local_descriptor(fp2, kp2, brief_detector)

    # find matches between keypoints based on local feature descriptor
    try:
        local_k1_k2_matches = brute_force_matcher(local_des1, local_des2, cv2.NORM_HAMMING)
    except:
        local_k1_k2_matches = brute_force_matcher(local_des1, local_des2, cv2.NORM_L2)

    # get source and target index for each match
    local_pt_source, local_pt_target = np.array([
        (match.queryIdx, match.trainIdx) for match in local_k1_k2_matches]).T

    # use matching keypoints to estimate an affine transform between keypoints
    M, inliers = estimate_affine_transform_by_kps(
        cv2.KeyPoint.convert(kp1[local_pt_source]),
        cv2.KeyPoint.convert(kp2[local_pt_target]))

    # if no inliers can be found
    if M is None:
        return 0

    # warp the keypoints according to the found transform
    kp1_reg = warp_points(cv2.KeyPoint.convert(kp1), M)

    # subset the keypoints, inliers are considered good keypoints
    # since they were used in finding the transformation
    global_k1_k2_matches = local_k1_k2_matches[inliers == 1]

    # compute hybrid similarity based on aligned matching keypoints
    return hybrid_img_similarity(global_k1_k2_matches, kp1_reg, kp2, dist_fct_n, thresholds)