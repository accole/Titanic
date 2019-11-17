"""
Description : Titanic
"""

## IMPORTANT: Use only the provided packages!

## SOME SYNTAX HERE.   
## I will use the "@" symbols to refer to some variables and functions. 
## For example, for the 3 lines of code below
## x = 2
## y = x * 2 
## f(y)
## I will use @x and @y to refer to variable x and y, and @f to refer to function f

import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics


######################################################################
# classes
######################################################################

class Classifier(object) :

    ## THIS IS SOME GENERIC CLASS, YOU DON'T NEED TO DO ANYTHING HERE. 

    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) : ## INHERITS FROM THE @CLASSIFIER

    def __init__(self) :
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        # n,d = X.shape ## get number of sample and dimension
        y = [self.prediction_] * X.shape[0]
        return y


class RandomClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- an array specifying probability to survive vs. not 
        """
        self.probabilities_ = None ## should have length 2 once you call @fit

    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        # in simpler wordings, find the probability of survival vs. not

        #initialize self.probabilities[]
        self.probabilities_ = []
        #count survival numbers
        surv = (y==1).sum()
        #count dead numbers
        dead = (y==0).sum()
        #calculate the probability and append to probabilities
        self.probabilities_.append(float(dead)/len(y))
        self.probabilities_.append(float(surv)/len(y))

        ### ========== TODO : END ========== ###

        return self

    def predict(self, X, seed=1234) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)

        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (check the arguments of np.random.choice) to randomly pick a value based on the given probability array @self.probabilities_

        #initialize prediction probability array
        y = []
        #for every element of set X
        for i in range(0, X.shape[0]):
            #append a random prediction to the array
            y.append(np.random.choice(2, 1, replace=True, p=self.probabilities_))

        ### ========== TODO : END ========== ###

        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')

    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend()
        #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use @train_test_split to split the data into train/test set 
    # xtrain, xtest, ytrain, ytest = train_test_split (X,y, test_size = test_size, random_state = i)
    # now you can call the @clf.fit (xtrain, ytrain) and then do prediction

    train_error = 0 ## average error over all the @ntrials
    test_error = 0
    train_scores = []; test_scores = []; ## tracking the error for each of the @ntrials, these array should have length 100 once you're done.

    #find the errors over all the trials
    for i in range(0, ntrials):
        #split the data into testing and training data based on argument
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state= i)

        #fit the training data using the classifer
        clf.fit(X_train, y_train)

        #use classifier to make predictions with training data and record error
        y_pred = clf.predict(X_train)
        train_err = 1 - metrics.accuracy_score(y_train, y_pred, normalize=True)
        train_scores.append(train_err)
        train_error += train_err

        #use classifier to make predictions with test data and record error
        y_pred = clf.predict(X_test)
        test_err = 1 - metrics.accuracy_score(y_test, y_pred, normalize=True)
        test_scores.append(test_err)
        test_error += test_err

    #average the errors
    train_error = train_error / ntrials
    test_error = test_error / ntrials

    ### ========== TODO : END ========== ###

    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features



    #========================================
    # part a: plot histograms of each feature
    print('Plotting...')
    for i in range(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)


    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)



    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')

    #classify using random classifier
    clf = RandomClassifier()
    #fit the data using the classifier
    clf.fit(X,y)
    #run the classifier on training data
    y_pred = clf.predict(X)
    #collect the error created with the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain
    print('Classifying using Decision Tree...')
    # call the function @DecisionTreeClassifier

    clf = DecisionTreeClassifier(criterion='entropy')
    #fit the data using the classifier
    clf.fit(X,y)
    #use the classifier on the training data
    y_pred = clf.predict(X)
    #collect the error for the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)


    ### ========== TODO : END ========== ###



    # note: uncomment out the following lines to output the Decision Tree graph
    
    # save the classifier -- requires GraphViz and pydot
    """
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf")
    """
    
    #for python3
    """
    import io, pydot
    from sklearn import tree
    dot_data = io.StringIO()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=Xnames)
    (graph, ) = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf")
    """


    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors
    print('Classifying using k-Nearest Neighbors...')
    # call the function @KNeighborsClassifier

    #test the training data on KNeighbors model with 3 neighbors
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X,y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error k=3: %.3f' % train_error)

    #and with 5 neighbors
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X,y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error k=5: %.3f' % train_error)

    #and with 7 neighbors
    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(X,y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error k=7: %.3f' % train_error)

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    # call your function @error

    #record training and test errors using majority vote classification
    clf = MajorityVoteClassifier()
    train_error, test_error = error(clf, X, y)
    print('\t-- MajorityVote: training error = %.3f, test error = %.3f' % (train_error , test_error))

    #record training and test errors using random classification
    clf = RandomClassifier()
    train_error, test_error = error(clf, X, y)
    print('\t-- RandomClassifier: training error = %.3f, test error = %.3f' % (train_error , test_error))

    #record training and test errors using Decision Tree classification
    clf = DecisionTreeClassifier(criterion='entropy')
    train_error, test_error = error(clf, X, y)
    print('\t-- DecisionTreeClassifier: training error = %.3f, test error = %.3f' % (train_error , test_error))

    #record training and test errors using k=5 KNeighbors classification
    clf = KNeighborsClassifier(n_neighbors=5)
    train_error, test_error = error(clf, X, y)
    print('\t-- KNeighborsClassifier: training error k=5 = %.3f, test error = %.3f' % (train_error , test_error))

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    # hint: use the function @cross_val_score
    k = list(range(1,50,2))

    #keep a list of k values tested
    kvalues = []
    cv_score = [] ## track accuracy for each value of $k, should have length 25 once you're done
    for i in k:
        #add the tested k to values
        kvalues.append(i)
        #create a classifier for each k
        clf = KNeighborsClassifier(n_neighbors=i)
        #record the cross validation accuracy score
        cv_score.append(1 - np.average(cross_val_score(clf, X, y, cv=10)))

    #return value of k that has the minimum cross validation score
    #plt.xlabel('K')
    #plt.ylabel('10-fold cross validation error')
    #plt.plot(kvalues, cv_score, label='Validation Error')
    #plt.legend()
    #plt.savefig('partF.pdf')
    #plt.show()
    print('\t-- the value of k that minimizes cross validation error is %d' % kvalues[cv_score.index(min(cv_score))])

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')

    #record the depths tested
    depths = []
    #create error arrays to record errors
    train_e = []
    test_e = []
    #test depths 1-20
    for i in range(1, 21):
        #create a decision tree classifier for each depth
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=i)
        #record training and testing errors
        train_error, test_error = error(clf, X, y)
        depths.append(i)
        train_e.append(train_error)
        test_e.append(test_error)

    #return depth that has the smallest cross validation score
    #plt.xlabel('depth')
    #plt.ylabel('10-fold cross validation error rate')
    #plt.plot(depths, test_e, marker='o', label='Test Error')
    #plt.plot(depths, train_e, marker='x', label='Training Error')
    #plt.legend()
    #plt.savefig('partG.pdf')
    #plt.show()
    #print(test_e)
    print('\t-- the depth that minimizes cross validation error is %d' % depths[test_e.index(min(test_e))])    

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')

    #new error function to ease functionality of part H computation
    def split_error(clf, X, y, numtrials, frac):
        train_error = 0 ## average error over all the @ntrials
        test_error = 0
        
        #find the errors over all the trials
        for i in range(0, numtrials):
            #split the data into testing and training data based on argument
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, random_state= i)

            X_train = X_train[:int(len(X_train)*frac)]
            y_train = y_train[:int(len(y_train)*frac)]

            #fit the training data using the classifer
            clf.fit(X_train, y_train)

            #use classifier to make predictions with training data and record error
            y_pred_train = clf.predict(X_train)
            train_error += 1 - metrics.accuracy_score(y_train, y_pred_train, normalize=True)

            #use classifier to make predictions with training data and record error
            y_pred_test = clf.predict(X_test)
            test_error += 1 - metrics.accuracy_score(y_test, y_pred_test, normalize=True)

        #average the errors
        train_error = train_error / numtrials
        test_error = test_error / numtrials

        return train_error, test_error


    #use 10% of the data to test overall
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    #initialize error arrays
    tree_tr_error = []
    tree_test_error = []
    knn_tr_error = []
    knn_test_error = []

    #increment the training size by 10% each trial
    increments = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    #values that minimize cross validation
    min_depth = 6
    min_K = 7

    for i in increments:
        #split the training data into various sizes
        if i < 1:
            x_tr, x_tst, y_tr, y_tst = train_test_split(X_train, y_train, test_size=1-i)
        else:
            x_tr = X_train
            y_tr = y_train

        #initialize error arrays
        train_error = 0
        test_error = 0

        #create both models
        #decision tree
        DT_clf = DecisionTreeClassifier(criterion='entropy', max_depth=min_depth)
        train_error, test_error = split_error(DT_clf, X, y, numtrials=100, frac=i)
        tree_tr_error.append(train_error)
        tree_test_error.append(test_error)

        #K Neighbors
        K_clf = KNeighborsClassifier(n_neighbors=min_K)
        train_error, test_error = split_error(K_clf, X, y, numtrials=100, frac=i)
        knn_tr_error.append(train_error)
        knn_test_error.append(test_error)

    #plot the decision trees
    #plt.plot(increments, tree_tr_error, 'b', label="Decision Tree Training Error")
    #plt.plot(increments, tree_test_error, 'g', label="Decision Tree Testing Error")
    #plot the KNN plots
    #plt.plot(increments, knn_tr_error, 'y', label="KNeighbors Training Error")
    #plt.plot(increments, knn_test_error, 'r', label="KNeighbors Testing Error")

    #create a label and show the plot
    #plt.legend()
    #plt.xlabel("Training Sample Fraction")
    #plt.ylabel("Error")
    #plt.savefig("partH.pdf")
    #plt.show()

    ### ========== TODO : END ========== ###


    print('Done')


if __name__ == "__main__":
    main()
