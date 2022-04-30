import math
import numpy as np # Standard array processing package
import cv2
import matplotlib as mpl # Setting the default colormap for pyplot

from matplotlib import pyplot as plt # Plotting library
from tqdm.notebook import tqdm as tqdm_notebook, trange  # A visual progress bar library
from IPython.display import Markdown # Allows us to generate markdown using python code
from icecream import ic
from sklearn.manifold import TSNE           # used for dimensionality reduction

from kneed import KneeLocator # Simple module to find the knee in a series

from src.matrix_construction import construct_similarity_table

mpl.rc('image', cmap='gray')


# Helper functions
def plot_image_sequence(data, n, imgs_per_row = 10, cmap = None, titles = None):
    """
    Plotting function to simplify displaying images in a certain amount of rows and columns.
    :param data: data matrix
    :param n: number of images you want to display in the grid
    :param imgs_per_row: number of images to be displayed in each row
    :param cmap: colour map
    :param titles: list of image titles
    """

    n_rows = math.ceil(n / imgs_per_row)
    n_cols = min(imgs_per_row, n)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 10 * n_rows))
    for i in trange(n):
        if n == 1:
            if titles is not None:
                ax.set_title(titles[i])
            ax.imshow(data[i], cmap=cmap)
        elif n_rows > 1:
            if titles is not None:
                ax[int(i / imgs_per_row), int(i % imgs_per_row)].set_title(titles[i][0], size = titles[i][1], pad=titles[i][2])
            ax[int(i / imgs_per_row), int(i % imgs_per_row)].imshow(data[i], cmap=cmap)
        else:
            if titles is not None:
                ax[int(i % n)].set_title(titles[i])
            ax[int(i % n)].imshow(data[i], cmap=cmap)

    print('Plotting images...')
    plt.show()


def plot_multiple_distance_fct_scores(org_img, images_db, labels_db, dist_fct, dist_fcts_n, imgs_per_row=4):
    """

    :param org_img: image to compare to
    :param img_db: image database
    :param labels: image label (or id)
    :param dist_func: function that computes the distance between two images
    :param dist_fcts_n: list of names of distance functions
    :param imgs_per_row:
    """
    n = len(dist_fcts_n)
    n_rows = math.ceil(n / imgs_per_row)
    n_cols = min(imgs_per_row, n)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 10 * n_rows))
    for i in trange(n):
        sim_tb0 = construct_similarity_table(org_img, images_db, labels_db, dist_fct, dist_fct_n=dist_fcts_n[i])
        # sim_tb0.sort_values(by='score', ascending=False).values[:]
        ids,scores = sim_tb0.sort_values(by='score', ascending=False).values[:,0], sim_tb0.sort_values(by='score', ascending=False).values[:,1]
        if n == 1:
            ax.plot(ids,scores)
            ax.set_xticks(np.arange(0,100,5),ids[np.arange(0,100,5)])
            ax.set_title('Highest Score ID for ' + dist_fcts_n[i] + ' : '+ids[0])
        elif n_rows > 1:
            ax[int(i / imgs_per_row), int(i % imgs_per_row)].set_title('Highest Score ID for ' + dist_fcts_n[i] + ' : '+ids[0])
            ax[int(i / imgs_per_row), int(i % imgs_per_row)].set_xticks(np.arange(0,100,5))
            ax[int(i / imgs_per_row), int(i % imgs_per_row)].set_xticklabels(ids[np.arange(0,100,5)])
            ax[int(i / imgs_per_row), int(i % imgs_per_row)].plot(ids,scores)
        else:
            ax[int(i % n)].set_title('Highest Score ID for ' + dist_fcts_n[i] + ' : '+ids[0])
            ax[int(i % n)].set_xticks(np.arange(0,100,5),ids[np.arange(0,100,5)])
            ax[int(i % n)].plot(ids,scores)

    print('Plotting images...')
    plt.show()


def plot_similarity_scores(similarity_m, similarity_name='mss'):
    ids,scores = similarity_m.sort_values(by='score', ascending=False).values[:,0], similarity_m.sort_values(by='score', ascending=False).values[:,1]
    plt.plot(ids,scores)
    plt.xticks(np.arange(0,100,5),ids[np.arange(0,100,5)],rotation = 45)
    plt.title('Highest Score ID for ' + similarity_name + ' : '+ids[0])

    Markdown(f'''We can see that the highest match score {scores[0]:.4f}
    belongs to subject '''+ids[0]+'. However, subject '+ids[1]+f''' is not far behind with
    a score of {scores[1]:.4f}! Surely we can do a better job than that. Or can we..?''')


def draw_keypoints(images_db, test_ids, kp_list, detector_names, flag, imgs_per_row=4):

    n = len(kp_list) * len(test_ids)
    n_rows = math.ceil(n / imgs_per_row)
    n_cols = min(imgs_per_row, n)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 10 * n_rows), sharey=True)
    fig.tight_layout()

    m = 0
    for i in trange(len(kp_list)):
        for j, t_id in enumerate(test_ids):
            img = cv2.drawKeypoints(images_db[t_id], kp_list[i][j], None, color=(0, 255, 0), flags=flag)

            if n == 1:
                ax.set_title('Keypoint from ' + detector_names[i] + ' for img id ' + str(t_id))
                ax.imshow(img)
            elif n_rows > 1:
                ax[int(m / imgs_per_row), int(m % imgs_per_row)].set_title('Keypoint from ' + detector_names[i] + ' for img id ' + str(t_id))
                ax[int(m / imgs_per_row), int(m % imgs_per_row)].axis('off')
                ax[int(m / imgs_per_row), int(m % imgs_per_row)].imshow(img)
            else:
                ax[int(m % n)].set_title('Keypoint from ' + detector_names[i] + ' for img id ' + str(t_id))
                ax[int(m % n)].axis('off')
                ax[int(m % n)].imshow(img)
            m+=1

    print('Plotting images...')
    plt.show()


def plot_TSNE(features, ax, n_components=2, verbose=1, title="ORB"):
    tsne = TSNE(n_components=n_components, verbose=verbose)
    tsne_results = tsne.fit_transform(features)
    x,y = tsne_results[:, 0], tsne_results[:, 1]

    ax.scatter(x, y)
    ax.set_title(f't-SNE plot using {title}')
    ax.set_ylabel("comp 1")
    ax.set_xlabel("comp 2")


def plot_vector_descriptor(local_desc, detector_name='ORB'):
    plt.figure(figsize=(10, 5))
    plt.imshow(local_desc[1].reshape(1, -1), interpolation='nearest')
    plt.colorbar(orientation='horizontal')
    plt.yticks([])
    plt.title(detector_name + ' local description vector')


def draw_matches(kp1, kp2, local_matches, enhanced_db, test_nr1, test_nr2, ax):
    # Visualize the matched keypoints

    imMatches = cv2.drawMatches(enhanced_db[test_nr1],kp1,
                                enhanced_db[test_nr2],kp2,local_matches, None)
    ax.imshow(imMatches)
    # ax.set_size_inches(18,9)
    ax.axis('off');

    # plt.show()


def plot_highest_score_and_knee(sim_tb, similarity_metric='global', norm=False):
    sim_tb = sim_tb.sort_values(by='score', ascending=False)
    ids,scores = sim_tb.values[:,0],sim_tb.values[:,1]

    [print(i+1, sim_tb.iloc[i,0], f"{sim_tb.iloc[i,1]:.5f}") for i in range(10)];

    plt.plot(ids,scores)
    plt.xticks(np.arange(0,100,5),ids[np.arange(0,100,5)],rotation = 45)
    plt.title('Highest Score ID: '+ids[0]);

    # First normalize the values (min-max)
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min())

    kneedle = KneeLocator(list(range(len(ids))),scores_norm,S=1, curve='convex', direction='decreasing')
    print(f'Knee at: {kneedle.knee}\t Treshold: ',kneedle.knee_y)
    kneedle.plot_knee()
    Markdown(f" With the threshold of {kneedle.knee_y:.3f}, we can see a more clear divide between the subjects. However, the best 6 matches are all pretty close... We should be able to come up with a better discriminator by using a " + similarity_metric + " similarity metric." )

    if norm:
        return sim_tb, ids, scores, kneedle, scores_norm
    else:
        return sim_tb, ids, scores, kneedle


def get_image_by_label(ind, db_label, db_images):
    for i in range(len(db_label)):
        if db_label[i] == ind:
            return db_images[i]
    raise ValueError("No image with this label is found.")


def plot_images_given_knee(ids, kneedle, db_label, db_images):
    imgs = [get_image_by_label(i, db_label, db_images) for i in ids[:kneedle.knee]]

    plot_image_sequence(imgs,kneedle.knee if kneedle.knee<=7 else 7)
