import numpy as np
import scipy as sp
import scipy.io
import matplotlib.pyplot as pyplot

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import linear_model, datasets
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

import math

'''run various scikit-learn methods on GIST dataset'''



def importGistData():
    '''first obtarin a numpy array from the mat file and label vector from csv'''
    train_mat = scipy.io.loadmat('train.mat')
    public_val_mat = scipy.io.loadmat('val.mat')
    train_GIST = train_mat['test']
    public_val_GIST = public_val_mat['val']

    train_labels = np.loadtxt('train.csv', delimiter=',', skiprows=1, usecols=[1])
    # val_predictions_baseline = np.loadtxt('sample_submission.csv', delimiter=',', skiprows=1, usecols=[1])
    # val_predictions_baseline = val_labels_baseline[0:970] #only up to 970 labelled

    return train_GIST, public_val_GIST, train_labels

def proportionalDivide(train_GIST, train_labels, train_prop):
    #initialize the arrays to be created
    split_train_GIST = []
    split_val_GIST = []
    split_train_labels = np.array([])
    split_val_labels = np.array([])

    for i in np.unique(train_labels):
        num_class_samples = np.sum(train_labels==i)
        train_size = math.floor(num_class_samples*train_prop)
        val_size = num_class_samples - train_size

        #append labels based on proportionally calculated sizes
        split_train_labels = np.append(split_train_labels, np.ones(train_size)*i)
        split_val_labels = np.append(split_val_labels, np.ones(val_size)*i)

        #shuffle training examples, then select first train_prop, leaving rest for val
        train_GIST_to_be_split = np.random.permutation(train_GIST[train_labels==i]) #shuffle
        split_train_GIST.append(train_GIST_to_be_split[0:train_size])
        split_val_GIST.append(train_GIST_to_be_split[train_size:])

    #merge list of GIST train and val sets
    split_train_GIST = np.concatenate((split_train_GIST), axis=0)
    split_val_GIST = np.concatenate((split_val_GIST), axis=0)

    return split_train_GIST, split_val_GIST, split_train_labels, split_val_labels

def LogReg(split_train_GIST, split_val_GIST, split_train_labels, split_val_labels):
    import pdb; pdb.set_trace()
    clf = SGDClassifier(
        loss='LR', #change to hinge for linear SVM
        penalty='l2',
        alpha=1e-1,
        n_iter=100,
        n_jobs=-1,
        learning_rate='optimal',
        eta0=1e-7,
        )

    alpha_tests = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    eta0_tests = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    for alphas in alpha_tests:
        clf.alpha = alphas
        clf.fit(split_train_GIST, split_train_labels)
        score = clf.score(split_val_GIST, split_val_labels)
        print(alphas, score)

    for eta0s in eta0_tests:
        clf.eta0 = eta0s
        clf.fit(split_train_GIST, split_train_labels)
        score = clf.score(split_val_GIST, split_val_labels)
        print(eta0s, score)
    import pdb; pdb.set_trace()
    # logreg = linear_model.LogisticRegression(
    #     C=1e5,
    #     max_iter=100,
    #     solver='liblinear',
    #     multi_class='ovr',
    #     tol=1e-4)
    #
    # C_list = [1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1]
    # for Cs in C_list:
    #     logreg.C = Cs
    #     logreg.fit(split_train_GIST, split_train_labels)
    #     score = logreg.score(split_val_GIST, split_val_labels)
    #     print(Cs, score)
    # import pdb; pdb.set_trace()

def SVM(split_train_GIST, split_val_GIST, split_train_labels, split_val_labels):
    import pdb; pdb.set_trace()
    clf = SGDClassifier(
        loss='hinge', #change to log for LR
        penalty='l2',
        alpha=1e-1,
        n_iter=200,
        n_jobs=-1,
        learning_rate='optimal',
        eta0=1e-7,
        )

    alpha_tests = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    eta0_tests = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    for alphas in alpha_tests:
        clf.alpha = alphas
        clf.fit(split_train_GIST, split_train_labels)
        score = clf.score(split_val_GIST, split_val_labels)
        print(alphas, score)

    for eta0s in eta0_tests:
        clf.eta0 = eta0s
        clf.fit(split_train_GIST, split_train_labels)
        score = clf.score(split_val_GIST, split_val_labels)
        print(eta0s, score)

def linearSVC(split_train_GIST, split_val_GIST, split_train_labels, split_val_labels):
    lsvc = svm.LinearSVC(
        C=1.0,
        loss='squared_hinge',
        tol=1e-4,
        max_iter=1000
    )

    Clist = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    for Cs in Clist:
        lsvc.C = Cs
        lsvc.fit(split_train_GIST, split_train_labels)
        score = lsvc.score(split_val_GIST, split_val_labels)
        print (Cs, score)

def GNB(split_train_GIST, split_val_GIST, split_train_labels, split_val_labels):
    import pdb; pdb.set_trace()
    prior = np.bincount(split_train_labels.astype(np.int64))[1:]
    prior = prior/np.sum(prior)

    clf = GaussianNB(priors=prior)
    clf.fit(split_train_GIST, split_train_labels)
    print(clf.score(split_val_GIST, split_val_labels))


if __name__ == '__main__':

    train_GIST, public_val_GIST, train_labels = importGistData()
    split_train_GIST, split_val_GIST, split_train_labels, split_val_labels\
        = proportionalDivide(train_GIST, train_labels, 0.8)

    GNB(split_train_GIST, split_val_GIST, split_train_labels, split_val_labels)

    linearSVC(split_train_GIST, split_val_GIST, split_train_labels, split_val_labels)

    # neigh = KNeighborsClassifier(n_neighbors=136, weights='distance')
    # neigh.fit(split_train_GIST, split_train_labels)


    # filler = np.zeros(2000)
    # predictions = np.concatenate((neigh.predict(val_GIST),filler))
    # prediction_labels = np.arange(970+2000)+1
    # prediction_outtxt = np.concatenate((prediction_labels.reshape(-1,1), predictions.reshape(-1,1)), axis=1)
    # prediction_outtxt = prediction_outtxt.astype(int)
    # np.savetxt('knn_predictions.csv', prediction_outtxt, comments='',
    #     delimiter=',', header='Id, Prediction', fmt='%.i')

    # score = neigh.score(val_GIST, val_labels_baseline)
    #
    #
    # # hyperparam_test = np.zeros((150, 2))
    #
    # # for i in range(30):
    # #     neigh = KNeighborsClassifier(n_neighbors=1+i*5, weights='distance')
    # #     neigh.fit(train_GIST, train_labels)
    # #     score = neigh.score(val_GIST, val_labels_baseline)
    # #     print((1+i*5), score)
    # #     hyperparam_test[i, 0] = 1+i*5
    # #     hyperparam_test[i, 1] = score
    #
    # import pdb; pdb.set_trace()
    # np.savez(hyperparam_test)
