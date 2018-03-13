import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import scikitplot as skplt


class ROC:
    """
    ROC curve builder class.
    Classes are assumed to be binary

    """
    # results is an numpy array formed by stacking together fpr, tpr and corresponding thresholds.
    # use results for analysis
    results = None

    def __init__(self, proba, true_labels, pos_label_value, pos_label=1):
        """
        Use these values in calc_tpr_fpr() method

        :param proba: numpy array of class probabilities
        :param true_labels: numpy array of true labels
        :param pos_label_value: The value of the positive label (usually 1)
        :param pos_label: The relative order of positive label in proba
        """
        self.proba = proba
        self.true_labels = true_labels
        self.pos_label_value = pos_label_value
        self.pos_label = pos_label

    def plot(self):
        """
        Plots an ROC curve using True Positive Rate and False Positive rate lists calculated from __calc_tpr_fpr
        Calculates and outputs AUC score on the same graph
        """
        tpr, fpr, thresholds = self.__calc_tpr_fpr()
        self.results = np.column_stack((tpr, fpr, thresholds))

        # %%% TODO START YOUR CODE HERE %%%
        plt.plot(fpr, tpr) 
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive rate')
        plt.show()
        
        index1 = [i for i in np.arange(len(tpr)) if thresholds[i] >= 0.5][0]
        index2 = [j for j in np.arange(len(tpr)) if tpr[j] >= 1][0]
        
        print("Answer question1: (tpr = ", tpr[index1], ", fpr = ", fpr[index1],")")
        
        print("Answer question2: threshold value = ", thresholds[index2])
        
        # %%% END YOUR CODE HERE %%%

    def __calc_tpr_fpr(self):
        """
        Calculates True Positive Rate, False Positive Rate and thresholds lists

        First, sorts probabilities of positive label in decreasing order
        Next, moving towards the least probability locates a threshold between instances with opposite classes
        (keeping instances with the same confidence value on the same side of threshold),
        computes TPR, FPR for instances above threshold and puts them in the lists

        :return:
        tpr: list
        fpr: list
        thresholds: list
        """
        # %%% TODO START YOUR CODE HERE %%%
        tpr = []
        fpr = []
        
        sorted_prob = self.proba[:,1 - self.pos_label]
        
        indexes = np.argsort(sorted_prob)
        sorted_prob = sorted_prob[indexes]

        sorted_true_labels = self.true_labels[indexes]
        sorted_proba = self.proba[indexes]
        
        n = len(sorted_prob)
        
        thresholds = [i for i in np.arange(1, n) if (sorted_true_labels[i] != sorted_true_labels[i - 1])]
        thresholds.append(n - 1)
        thr_values = sorted_prob[thresholds]
        
        n_positive = len([v for v in sorted_true_labels if v == 1])
        n_negative = n - n_positive
        
        for i in range(len(thresholds)):
            thr = thresholds[i]
            thr_pos = len([v for v in sorted_true_labels[0:thr] if v == 1])
            tpr.append(thr_pos / n_positive)
            fpr.append((thr - thr_pos) / n_negative)
        return(tpr, fpr, thr_values)
        # %%% END YOUR CODE HERE %%%


def stratified_train_test_split(X, Y, test_size, random_seed=None):
    """
    Performs the stratified train/test split
    (with the same (!) inter-class ratio in train and test sets as compared to original set)
    input:
        X: numpy array of size (n,m)
        Y: numpy array of size (n,)
        test_size: number between 0 and 1, specifies the relative size of the test_set
        random_seed: random_seed

    returns:
        X_train
        X_test
        Y_train
        Y_test
    """
    if test_size < 0 or test_size > 1:
        raise Exception("Fraction for split is not valid")

    np.random.seed(random_seed)

    # %%% TODO START YOUR CODE HERE %%%
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    Y_pos = []
    X_pos = []
    Y_neg = []
    X_neg = []
    for i in range(len(Y)):
        if Y[i] == 1:
            Y_pos.append(Y[i])
            X_pos.append(X[i])
        else:
            Y_neg.append(Y[i])
            X_neg.append(X[i])
    
    test_size_pos = int(len(Y_pos) * test_size)
    test_sample = np.random.random_integers(len(Y_pos), size=(1,test_size_pos))

    for i in range(len(Y_pos)):
        if i in test_sample:
            X_test.append(X_pos[i])
            Y_test.append(Y_pos[i])
        else:
            X_train.append(X_pos[i])
            Y_train.append(Y_pos[i])
            
    test_size_neg = int(len(Y_neg) * test_size)
    test_sample = np.random.random_integers(len(Y_neg), size=(1,test_size_neg))
    for i in range(len(Y_neg)):
        if i in test_sample:
            X_test.append(X_neg[i])
            Y_test.append(Y_neg[i])
        else:
            X_train.append(X_neg[i])
            Y_train.append(Y_neg[i])

    return(np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test))
    
    # %%% END YOUR CODE HERE %%%


data = load_breast_cancer()

# Pre-processing: Exchange labels - make malignant 1, benign 0
data['target'] = np.array(data['target'], dtype=int) ^ 1

X_train, X_test, y_train, y_test = stratified_train_test_split(data['data'], data['target'], 0.3, 10)

# Check that the ratio is preserved
print("Inter-class ratio in original set:", len(np.argwhere(data['target'] == 1))/len(np.argwhere(data['target'] == 0)))
print("Inter-class ratio in train set:", len(np.argwhere(y_train == 1))/len(np.argwhere(y_train == 0)))
print("Inter-class ratio in test set:", len(np.argwhere(y_test == 1))/len(np.argwhere(y_test == 0)))
print('\n')

# We pick Logistic Regression because it outputs probabilities
# Try different number of iterations to change ROC curve
model = LogisticRegression(max_iter=5)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
y_pred = model.predict(X_test)
print("Classifier's Accuracy:", accuracy_score(y_test, y_pred))

# Build an ROC curve
roc = ROC(probabilities, y_test, 1)
roc.plot()
# Explore the results
results = roc.results

# Use scikitplot library to compare ROC curve with the one you are getting
skplt.metrics.plot_roc_curve(y_test, probabilities)
plt.show()


# ROC analysis questions:
# 1. What are fpr, tpr rates if we choose 0.5 as a threshold?
# %%% TODO Answer HERE %%%
#tpr = 0.8888888888888888
#fpr = 0.0898876404494 

# 2. Let's suppose this is a second cancer check for those who have high probability of cancer.
#    What threshold value will you use in this case and why?
# %%% TODO Answer HERE %%%
#threshold value = 0.944540796518
