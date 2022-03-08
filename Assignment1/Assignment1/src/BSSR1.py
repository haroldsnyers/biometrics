# function to read BSSR1 scores from file and convert to pandas dataframe
import numpy as np

# package for data analysis with fast and flexible data structures
import pandas as pd

# package for flexible pathname manipulation
import glob

# package for reading xml files
import xml.etree.ElementTree as ET

# package to show a nice graphical progress-bar for lengthy calculations
# docu and installation on https://tqdm.github.io
# if you have difficulties installing this package then just leave it out
from tqdm.notebook import tqdm as tqdm_notebook

import os



def read_BSSR1_scores_from_file(enrollees_id_filepath, users_id_filepath, path):


    # parse the XML files
    enrollees = ET.parse(enrollees_id_filepath)
    users = ET.parse(users_id_filepath)

    dataframe = []
    files = glob.glob(path)

#     for filepath in glob.iglob(path):
# replace the following two lines of code with the previous line if the tqdm package is not installed
    for i in tqdm_notebook(range(len(files))):
        filepath = files[i]
        file = open(filepath,'r')

        file_name = os.path.split(filepath)[-1]

        read_data = np.array(file.read().split('\n'))
        sims = read_data[2:-2].astype(np.str)
        n_cmp = int(read_data[1])
        assert sims.shape[0] == n_cmp
        assert sims.shape[0] == 6000
        # "The order of the elements in the similarity file are fixed for all similarity files in the tree.
        # They are not sorted on similarity value. The order corresponds to the entries in the enrollees.xml
        # file."

        # grab the subject_id of current user
        try:
            subject_id = users.find("./*[@name='{}']".format(file_name)).attrib['subject_id']
        except:
            print('Could not find user: {}, from file: {}'.format(file_name, filepath))
            raise

        sims = np.insert(sims, 0, subject_id)
        dataframe.append(sims)

        file.close()

    # extract the column names for later indexing
    column_names = [e.attrib['subject_id'] for e in enrollees.findall("./*")]
    column_names_ex = column_names.copy()
    column_names_ex.insert(0,'subject_id')

    # convert to pandas dataframe
    df = pd.DataFrame(dataframe, columns=column_names_ex)

    # set index to subject_id and organise rows according to column order
    df = df.set_index('subject_id')

    # show initial rows
    # df.head(10)


    return(df, enrollees, users, column_names)

#-------------------------------------------------------------------------------
## subsample the scores to a manageable number of individuals

def df2sim_subsample(df, column_names, nr_individuals = 1000):
    random_names = np.array(column_names)

    # for reproducible results we need to generate same random sequences
    # this can be obtained through the RandomState function
    from numpy.random import RandomState
    RS = RandomState(1234567890)
    RS.shuffle(random_names)

    # in case you want each time different random selections uncomment the following line
    # np.random.shuffle(random_names)
    random_names = random_names[:nr_individuals]

    # store in the similarity matrix
    similarity_matrix = df.loc[random_names,random_names].astype(np.float)
    # similarity_matrix
    return(similarity_matrix)
#-------------------------------------------------------------------------------

## convert to genuine and imposter scores

def sim2scores(similarity_matrix):
    # use .values to access as numpy array
    np_similarity_matrix = similarity_matrix.values

    # grab elements on the diagonal
    genuine_scores = np.diag(np_similarity_matrix)

    # mask elements that are on the diagonal, retain non-diagonal elements
    imposter_scores =  np_similarity_matrix[~np.eye(np_similarity_matrix.shape[0],dtype=bool)]

    # store in one single list of scores as required for the classification validation procedures
    scores = np.append(np.array(genuine_scores), np.array(imposter_scores))

    # normalize to [0,1] range, 0 corresponding to minimal similarity
    scores = (scores - scores.min())/(scores.max()-scores.min())

    ## add the genuine and imposter labels

    # tag genuine combinations as label 1 and imposter combinations as 0

    genuine_id = np.zeros_like(scores)
    genuine_id[0:genuine_scores.shape[0]] = 1

    return(genuine_id, scores)
