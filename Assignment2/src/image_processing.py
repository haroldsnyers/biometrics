import os
import warnings
import matplotlib as mpl  # Setting the default colormap for pyplot
import numpy as np  # Standard array processing package

from pathlib import Path  # File path processing package
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm
import cv2

import src.fprmodules.enhancement as fe
from src.irismodules.iris_recognition.python.fnc import segment, normalize

mpl.rc('image', cmap='gray')
warnings.filterwarnings('ignore')

# FINGER Processing ####################################################


def read_db(path):
    images = []
    labels = []
    imagePaths = sorted(Path(path).rglob("*.png"))
    for imagePath in tqdm_notebook(imagePaths):
        image = cv2.imread(path + imagePath.name)
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(image)
        label = imagePath.stem[0:3]
        labels.append(label)
    return images, labels


# Calculate the enhanced images and the associated segmentation masks
def enhance_images(images):
    """

    :param images:
    :return:
    """
    images_e_u = []
    masks = []
    for i, image in enumerate(tqdm_notebook(images)):
        try:
            # Gabor filtering
            img_e, mask, orientim, freqim = fe.image_enhance(image)
            # Normalize in the [0,255] range
            img_e_u = cv2.normalize(img_e, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=0)
            images_e_u.append(img_e_u)
        except:
            print('error for: ', i)
            images_e_u.append(image)
        masks.append(mask)
    return np.array(images_e_u), np.array(masks)


# IRIS #####################################


def read_iris_db(path):
    images = []
    labels = []
    imagePaths = sorted(os.listdir('data\\CASIA1'))

    for imagePath in imagePaths:
        if imagePath.endswith('.png'):
            label = imagePath.split('.')[0]
            image = cv2.imread(path + '\\' + imagePath)
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels)


def segment_iris(img, radial_res=200, angular_res=500):
    """
    we use the same hyper parameters as in the enroll-casia1.py example
    :param img:
    :param radial_res:
    :param angular_res:
    :return:
    """

    # Segment the iris region from the eye image. Indicate the noise region.
    ciriris, cirpupil, imwithnoise = segment.segment(img)

    # Normalize iris region by unwraping the circular region into a rectangular block of constant dimensions.
    polar_iris, mask = normalize.normalize(
        imwithnoise, ciriris[1], ciriris[0], ciriris[2], cirpupil[1], cirpupil[0], cirpupil[2], radial_res, angular_res)

    return polar_iris, (mask == 0)


def get_filter_bank(ksize=5, sigma=4, theta_range=np.arange(0, np.pi, np.pi / 16), lambd=10, gamma=0.5, psi=0):
    # this filterbank comes from https://cvtuts.wordpress.com/
    filters = []
    for theta in theta_range:
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters
    cv2.getGaborKernel()


def enhance_iris(img, eps=1.e-15, agg_f=np.max):
    # get the gabor filters
    filters = get_filter_bank()

    # apply filters to image
    enhanced_image = np.array([cv2.filter2D(img, ddepth=-1, kernel=k) for k in filters])

    # Normalize in the [0,255] range
    enhanced_image = cv2.normalize(enhanced_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=0)

    # aggregate features
    return agg_f(enhanced_image, 0)


def enhance_and_segment_irises(images, radial_res=200, angular_res=500):
    enhanced_images, masks = [], []
    for img in tqdm(images):
        normalised_img, mask = segment_iris(img, radial_res, angular_res)
        enhanced_img = enhance_iris(normalised_img)

        masks.append(mask)
        enhanced_images.append(enhanced_img)

    return np.array(enhanced_images), np.array(masks)


# Common processing ############################################################################


def detect_keypoints(img, mask, keypoint_detector, kernel_size=(5, 5)):
    """
        Detects keypoints in an image.

        Note: Many false keypoints will be generated at the edge of the foreground mask, since ridges seem to terminate due to the clipping.
        we remove these by a morpholigical erosion (shrinking) of the foreground mask and deleting the keypoints outside.
        :param img:
        :param mask:
        :param keypoint_detector:
        :param kernel_size:
        :return: keypoints detected by detector
    """
    # find the keypoints with ORB
    kp = keypoint_detector.detect(img)

    # convert mask to an unsigned byte
    mask_b = mask.astype(np.uint8)
    # morphological erosion
    mask_e = cv2.erode(mask_b * 255, kernel=np.ones(kernel_size, np.uint8), iterations=5)
    # remove keypoints and their descriptors that lie outside this eroded mask
    kpn = [kp[i] for i in range(len(kp)) if mask_e.item(int(kp[i].pt[1]), int(kp[i].pt[0])) == 255]
    return kpn


def compute_local_descriptor(img, kp, detector):
    """

    :param img:
    :param kp:
    :param detector:
    :return:
    """
    kp, des = detector.compute(img, kp)
    return np.array(kp), des


def brute_force_matcher(des1, des2, dist=cv2.NORM_HAMMING):
    """
      Brute Force matcher on a pair of KeyPoint sets using the local descriptor for similarity

      returns all pairs of best matches
      :param des1:
      :param des2:
      :param dist:
      :return:
    """

    # crossCheck=True only retains pairs of keypoints that are each other best matching pair
    bf = cv2.BFMatcher(dist, crossCheck=True)
    matches = bf.match(des1, des2)

    # sort matches based on descriptor distance
    matches.sort(key=lambda x: x.distance, reverse=False)

    return np.array(matches)


def estimate_affine_transform_by_kps(src_pts, dst_pts):
    """
        Returns the Affine transformation that aligns two sets of points
    """
    transform_matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts,
                                                            method=cv2.RANSAC,
                                                            confidence=0.9,
                                                            ransacReprojThreshold=10.0,
                                                            maxIters=5000,
                                                            refineIters=10)
    return transform_matrix, inliers[:, 0]


def warp_points(pts, M):
    mat_reg_points = cv2.transform(pts.reshape(-1, 1, 2), M)

    # return transformed keypoint list
    return cv2.KeyPoint.convert(mat_reg_points)


def warp_img(img, M):
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def get_reduced_set(matches, reg_kp1, kp2):
    # get reduced keypoint set from matches
    reg_kp1 = cv2.KeyPoint_convert(reg_kp1)
    kp2 = cv2.KeyPoint_convert(kp2)
    kp1_reduced, kp2_reduced = [], []
    for i in matches:
        kp1_reduced.append(reg_kp1[i.queryIdx])
        kp2_reduced.append(kp2[i.trainIdx])
    return kp1_reduced, kp2_reduced


#####################################################
# Iris neural network

def read_iris_casia_db(path):
    images = []
    labels = []
    labels_id = []
    ids = []
    imagePaths = sorted(os.listdir(path))
    for i, classes in enumerate(imagePaths):
        for imagePath in os.listdir(path + classes):
            # change depending on file type
            if imagePath.endswith('.jpg'):
                id = imagePath.split('.')[0]
                label = classes
                image = cv2.imread(path + classes + '\\' + imagePath)
                if len(image.shape) > 2:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                images.append(image)
                labels.append(label)
                labels_id.append(i)
                ids.append(id)

    return np.array(images), np.array(labels), np.array(labels_id), np.array(ids)
