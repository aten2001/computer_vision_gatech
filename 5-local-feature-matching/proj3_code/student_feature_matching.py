from typing import Tuple
import torch
import numpy as np


def compute_feature_distances(features1: np.ndarray, 
                              features2: np.ndarray) -> np.ndarray:
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)

    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """
    
    dists = np.zeros((features1.shape[0], features2.shape[0]))

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    for i in range(len(features1)):
        for j in range(len(features2)):
#             dists[i,j] = distance(features1[i,:], features2[j,:])
            dists[i,j] = np.linalg.norm(features1[i,:] -  features2[j,:])
#             dists[i,j] = torch.cdist(features1[i,:], features2[j,:], p=2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features(features1: np.ndarray, 
                   features2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform nearest-neighbor matching with ratio test.

    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    The results should be sorted in descending order of confidence.

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second
      set of features (m not necessarily equal to n)


    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    matches =[]
    confidences = []
    dists = compute_feature_distances(features1, features2)
    for i in range(dists.shape[0]):
        sdist = dists[i,:]
        ind = np.argsort(sdist)[:2]          
#         out, ind = torch.topk(dists[i,:], 2, largest=False)
        ratio = sdist[ind[0]]/sdist[ind[1]]
        if ratio <0.80:
            matches.append([i,ind[0]])
            confidences.append(dists[i,ind[0]])
            
#     matches = torch.Tensor(matches).to(dtype = torch.LongTensor)
    matches = np.asarray(matches)
    confidences = np.asarray(confidences)
    indices = np.argsort(confidences)
    confidences = confidences[indices[::-1]]
    matches = matches[indices[::-1],:]
    print(matches.shape)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences
