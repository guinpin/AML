import numpy as np
from sklearn.metrics import f1_score
from utils import load_data, train_test_split
import copy, math

X, Y = load_data("iris.csv")

X_tr, Y_tr, X_t, Y_t = train_test_split(X,Y,.7)


class KNN:
    def __init__(self):
        """
        Your initialization procedure if required
        """
        self.X_tr = []
        self.Y_tr = []

    def fit(self,X,Y):
        """
        KNN algorithm in the simples implementation can work only with
        continuous features

        X: training data, numpy array of shape (n,m)
        Y: training labels, numpy array of shape (n,1)
        """

        # Hint: make sure the data passed as input are of the type float
        # Hint: make sure to create copies of training data, not copies of
        #       references
        self.X_tr = copy.deepcopy(X)
        
        self.Y_tr = copy.deepcopy(Y)

    def get_distances(self, test_instance):
        dist = []
        for j in range(len(self.X_tr)):
            training_instance = self.X_tr[j]
            d = 0
            for i in range(len(test_instance)):
                d += math.pow((training_instance[i] - test_instance[i]), 2)
            d = math.sqrt(d)
            inst = []
            inst.append(d)
            inst.append(self.Y_tr[j])
            dist.append(inst)
        dist = np.array(dist)
        return(dist)
        
        
    def predict(self, X, nn=5):
        """
        X: data for classification, numpy array of shape (k,m)
        nn: number of nearest neighbours that determine the final decision

        returns
        labels: numpy array of shape (k,1)
        """
        ans = []
        for test_instance in X:
            distances_with_labels = []
            labels = []
            
            distances_with_labels = self.get_distances(test_instance)
            
            sorted_ind = np.argsort(distances_with_labels, axis = 0)
            sorted_ind = np.array(sorted_ind[:,0])
            
            col_idx = np.array([0, 1])
            distances_with_labels = distances_with_labels[sorted_ind[:, None], col_idx]
            distances_with_labels = distances_with_labels[:nn]
            
            labels = distances_with_labels[:,1]
            labels = list(labels)
            ans.append((max(labels, key=labels.count)))
        return(ans)
        # Hint: make sure the data passed as input are of the type float
        
    def weighted_predict(self, X, nn=5):
        ans = []
        for test_instance in X:
            distances_with_labels = []
            labels = []
                
            distances_with_labels = self.get_distances(test_instance)
                
            sorted_ind = np.argsort(distances_with_labels, axis = 0)
            sorted_ind = np.array(sorted_ind[:,0])
                
            col_idx = np.array([0, 1])
            distances_with_labels = distances_with_labels[sorted_ind[:, None], col_idx]
            distances_with_labels = distances_with_labels[:nn]
                
            labels = distances_with_labels[:,1]
            labels = list(labels)
                
            scores = dict()
            for i in range(nn):
                if (labels[i][0] not in scores.keys()):
                    scores[labels[i][0]] = 1/(distances_with_labels[i][0] + 0.000001)
                else: 
                    scores[labels[i][0]] += 1/(distances_with_labels[i][0] + 0.000001)
                
            current_ans = max(scores)
                
            ans.append(current_ans)
        return(ans)
# Task:
# 1. Implement function fit in the class KNN
# 2. Implement function predict in the class KNN, where neighbours are weighted
#     according to uniform weights
# 3. Test your algorithm on iris dataset according to
#     f1_score (expected: 0.93)
# 4. Test your algorithm on mnist_small dataset according to
#     f1_score (expected: 0.7)
# 5. Test your algorithm on mnist_large dataset according to
#     f1_score (expected: 0.86)
# 6. Implement function predict in the class KNN, where neighbours are weighted
#     according to their distance to the query instance

np.random.seed(1)

c = KNN()
c.fit(X_tr,Y_tr)
label_p = c.predict(X_t)
f1 = f1_score(Y_t, label_p, average = 'micro')
print("Test score for iris dataset %.2f"%(f1))
label_p = c.weighted_predict(X_t)
f1 = f1_score(Y_t, label_p, average = 'micro')
print("Test score for iris dataset with weighted predict %.2f"%(f1))

X, Y = load_data("mnist_small.csv")
X_tr, Y_tr, X_t, Y_t = train_test_split(X,Y,.7)
c = KNN()
c.fit(X_tr,Y_tr)
label_p = c.predict(X_t)
f1 = f1_score(Y_t, label_p, average = 'micro')
print("Test score for mnist small %.2f"%(f1))

X, Y = load_data("mnist_large.csv")
X_tr, Y_tr, X_t, Y_t = train_test_split(X,Y,.7)
c = KNN()
c.fit(X_tr,Y_tr)
label_p = c.predict(X_t)
f1 = f1_score(Y_t, label_p, average = 'micro')
print("Test score for mnist large %.2f"%(f1))
