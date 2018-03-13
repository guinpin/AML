from pandas import read_csv
import numpy as np
import random
import copy

def load_data(path_to_csv, has_header=True):
    """
    Loads a csv file, the last column is assumed to be the output label
    All values are interpreted as strings, empty cells interpreted as empty
    strings

    returns: X - numpy array of size (n,m) of input features
             Y - numpy array of output features
    """
    if has_header:
        data = read_csv(path_to_csv, header='infer', dtype=str)
    else:
        data = read_csv(path_to_csv, header=None, dtype=str)
    data.fillna('', inplace=True)
    data = data.as_matrix()
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y


class DTree:
    """
    Simple decision tree classifier for a training data with categorical
    features
    """
    _model = None

    def fit(self, X, Y):
        self._model = create_branches({'attr_id': -1,
                                       'branches': dict(),
                                       'decision': None}, X, Y)

    def predict(self, X):

        if X.ndim == 1:
            return traverse(self._model, X)
        elif X.ndim == 2:
            answers = []
            for x in X:
                answers.append(traverse(self._model, x))
            return answers
        else:
            print("Dimensions error")

    def prune(self, node, X_train, Y_train, X_test, Y_test):
        """
        Implement pruning to improve generalization
        """
        max_err  = 1
        
        if node['attr_id'] == -1:
            return max_err
        
        if (node == self._model):
            Y_pred = self.predict(X_test)
            best_err = measure_error(Y_test,Y_pred)
        else:
            old_node = copy.deepcopy(node)
            node['attr_id'] = -1
            Y_pred = self.predict(X_test)
            best_err = measure_error(Y_test,Y_pred)
            node = old_node
        
        
        for attr in node['branches']:
            res_err = self.prune(node['branches'][attr], X_train, Y_train, X_test, Y_test)           
            if (res_err < best_err):
                best_err = res_err
            
        
            
        return best_err
        


def elem_to_freq(values):
    """
    input: numpy array
    returns: The counts of unique elements, unique elements are not returned
    """
    # hint: check numpy documentation for how to count unique values
    unique_values = np.unique(values)
    return len(unique_values)


def entropy(elements):
    """
    Calculates entropy of a numpy array of instances
    input: numpy array
    returns: entropy of the input array based on the frequencies of observed
             elements
    """
    # hint: use elem_to_freq(arr) 
    entr = 0
    unique_values = np.unique(elements)
    for elem in unique_values:
        probab = len(elements) / len([v for v in elements if v == elem])
        entr = (-1) * probab * np.log2(probab) 
    return entr


def information_gain(A, S):
    """
    input:
        A: the values of an attribute A for the set of training examples
        S: the target output class

    returns: information gain for classifying using the attribute A
    """
    # hint: use entropy(arr)
    gain = entropy(S)
    unique_values = np.unique(A)
    for value in unique_values:
        A_ind = [ind for ind in np.arange(len(A)) if A[ind] == value]
        Sv = S[A_ind]
        gain -= entropy(Sv) * len(Sv) / len(S)
    return gain


def choose_best_attribute(X, Y):
    """
    input:
        X: numpy array of size (n,m) containing training examples
        Y: numpy array of size (n,) containing target class

    returns: the index of the attribute that results in maximum information
             gain. If maximum information gain is less that eps, returns -1
    """

    eps = 1e-10

    columns = []
    for i in range(X.shape[1]):
        columns.append(X[:,[i]])
    gains = list(map(lambda x: information_gain(x, Y), columns))
    best_attr = np.argmax(gains)
    if (gains[best_attr] < eps):
        return -1
    
    return best_attr


def most_common_class(Y):
    """
    input: target class values
    returns: the value of the most common class
    """
    max_amount = 0
    for value in np.unique(Y):
        amount = len([y for y in Y if y == value])
        if (amount > max_amount):
            max_amount = amount
            max_value = value
    
    return max_value


def create_branches(node, X, Y):
    """
    create branches in a decision tree recursively
    input:
        node: current node represented by a dictionary of format
                {'attr_id': -1,
                 'branches': dict(),
                 'decision': None},
              where attr_id: specifies the current attribute index for branching
                            -1 mean the node is leaf node
                    braches: is a dictionary of format {attr_val:node}
                    decision: contains either the best guess based on
                            most common class or an actual class label if the
                            current node is the leaf
        X: training examples
        Y: target class

    returns: input node with fields updated
    """
    # choose best attribute to branch
    attr_id = choose_best_attribute(X,Y)
    node['attr_id'] = attr_id
    # record the most common class
    node['decision'] = most_common_class(Y)

    if attr_id != -1:
        # find the set of unique values for the current attribute
        attr_vals = np.unique(X[:, [attr_id]])

        for a_val in attr_vals:
            # compute the boolean array for slicing the data for the next
            # branching iteration
            # hint: use logical operation on numpy array
            # for more information about slicing refer to numpy documentation
            sel = [ind for ind in np.arange(len(Y)) if X[ind][attr_id] == a_val]
            # perform slicing
            X_branch = X[sel, :]
            Y_branch = Y[sel]
            node_template = {'attr_id': -1,
                             'branches': dict(),
                             'decision': None}
            # perform recursive call
            node['branches'][a_val] = create_branches(node_template, X_branch, Y_branch)
    return node


def traverse(model,sample):
    """
    recursively traverse decision tree
    input:
        model: trained decision tree
        sample: input sample to classify

    returns: class label
    """
    if model['attr_id'] == -1:
        decision = model['decision']
    else:
        attr_val = sample[ model['attr_id'] ]
        if attr_val not in model['branches']:
            decision = model['decision']
        else:
            decision = traverse(model['branches'][attr_val], sample)
    return decision


def train_test_split(X, Y, fraction):
    """
    perform the split of the data into training and testing sets
    input:
        X: numpy array of size (n,m)
        Y: numpy array of size (n,)
        fraction: number between 0 and 1, specifies the size of the training
                data

    returns:
        X_train
        Y_train
        X_test
        Y_test
    """
    if fraction < 0 or fraction > 1:
        raise Exception("Fraction for split is not valid")
    
    # do random sampling for splitting the data
    batch_size = int (len(Y) * fraction)
    train_ind = random.sample(range(len(Y)), batch_size)
    test_ind = [ind for ind in np.arange(len(Y)) if ind not in train_ind]
    
    X_train = X[train_ind, :]
    Y_train = Y[train_ind]
    X_test = X[test_ind, :]
    Y_test = Y[test_ind]
    
    
    return X_train, Y_train, X_test, Y_test


def measure_error(Y_true, Y_pred):
    """
    returns an error measure of your choice
    """
    errors = len([ind for ind in np.arange(len(Y_true)) if Y_true[ind] != Y_pred[ind]])
    measure_error = errors/len(Y_true)
    return (measure_error)


def recall(Y_true, Y_pred):
    """
    returns recall value
    """
    positive_right_results = len([ind for ind in np.arange(len(Y_true)) if Y_true[ind] == Y_pred[ind] and Y_true[ind] == 'Yes'])
    recall = positive_right_results/len(Y_true)
    
    return recall


# 1.  test your implementation on data_1.csv
#     refer to lecture slides to verify the correctness
# 2.  test your implementation on mushrooms_modified.csv
# 3.  test your implementation on titanic_modified.csv
print("For data_1: ")
X,Y = load_data("data_1.csv")

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, .8)

d_tree = DTree()
d_tree.fit(X_train,Y_train)
Y_pred = d_tree.predict(X_test)
current_error = measure_error(Y_test,Y_pred)
print("Correctly classified: %.2f%%" % ((1 - current_error) * 100))

new_error = d_tree.prune(d_tree._model, X_train, Y_train, X_test, Y_test)
print("Correctly classified after pruning: %.2f%%" % ((1 - new_error) * 100))

print("Recall %.4f" % recall(Y_test, Y_pred))

print("For mushrooms: ")

X,Y = load_data("mushrooms_modified.csv")

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, .8)

d_tree = DTree()
d_tree.fit(X_train,Y_train)
Y_pred = d_tree.predict(X_test)
current_error = measure_error(Y_test,Y_pred)
print("Correctly classified: %.2f%%" % ((1 - current_error) * 100))

new_error = d_tree.prune(d_tree._model, X_train, Y_train, X_test, Y_test)
print("Correctly classified after pruning: %.2f%%" % ((1 - new_error) * 100))

print("Recall %.4f" % recall(Y_test, Y_pred))


print("For titanic: ")

X,Y = load_data("titanic_modified.csv")

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, .8)

d_tree = DTree()
d_tree.fit(X_train,Y_train)
Y_pred = d_tree.predict(X_test)
current_error = measure_error(Y_test,Y_pred)
print("Correctly classified: %.2f%%" % ((1 - current_error) * 100))

new_error = d_tree.prune(d_tree._model, X_train, Y_train, X_test, Y_test)
print("Correctly classified after pruning: %.2f%%" % ((1 - new_error) * 100))

print("Recall %.4f" % recall(Y_test, Y_pred))
