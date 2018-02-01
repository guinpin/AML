from pandas import read_csv
import numpy as np


def load_data(path_to_csv, has_header=True):
    if has_header:
        data = read_csv(path_to_csv, header='infer')
    else:
        data = read_csv(path_to_csv, header=None)
    data = data.as_matrix()
    X = data[:, 0:-1]
    Y = data[:, -1]
    return X, Y


class CandidateElimination:

    # candidate elimination algorithm
    def fit(self, training_data, labels):
        S = self.initialize_to_first_positive(training_data, labels)
        G = self.initialize_to_most_general(training_data)
        training_examples = len(training_data)
        for i in range(training_examples):
            if labels[i] == "yes":
                G = [g for g in G if self.is_consistent(training_data[i], g, True)]
                not_consistent = [s for s in S if not self.is_consistent(training_data[i], s, True)]
                S = [s for s in S if self.is_consistent(training_data[i], s, True)]
                for n in not_consistent:
                    h = n.copy()
                    self.add_min_generalization(h, training_data[i], G, S)
                S = [s for s in S if not self.is_more_general_than_any(s, S)]
            else:
                S = [s for s in S if self.is_consistent(training_data[i], s, False)]
                not_consistent = [g for g in G if not self.is_consistent(training_data[i], g, False)]
                G = [g for g in G if self.is_consistent(training_data[i], g, False)]
                for n in not_consistent:
                    h = n.copy()
                    self.add_min_specialization(h, training_data[i], S, G, training_data)
                G = [g for g in G if not self.is_less_general_than_any(g, G)]
        print("Final Version Space:")
        print("S: ", S)
        print("G: ", G)

    def initialize_to_first_positive(self, training_data, labels):
        """"
        Returns list with one hypothesis which is equal to the first positive example
        """
        for i in range(len(labels)):
            if labels[i] == 'yes':
                init_set = [training_data[i, :]]
                return init_set

    def initialize_to_most_general(self, training_data):
        """"
        Returns list with one most general hypothesis - ['?', '?', '?', '?'...]
        """
        hypothesis = []
        for i in range(training_data.shape[1]):
            hypothesis.append("?")
        return [np.array(hypothesis, dtype=object)]

    def is_consistent(self, training_example, hypothesis, is_positive):
        """"
        Returns True if the hypothesis classifies the training_example as:
            - positive if it's positive
            - negative if it's negative
        """
        # %%% TODO START YOUR CODE HERE %%%
        #print(training_example)
        #print(hypothesis)
        is_passed = True        
        for i in range(len(training_example)):
            if hypothesis[i] != '?':
                if training_example[i] != hypothesis[i]:
                    is_passed = False

        if is_positive == True and is_passed == True:
            #print(True)
            return True
        elif is_positive == False and is_passed == False:
            #print(True)
            return True
        else: 
            #print(False)
            return False
        # %%% END YOUR CODE HERE %%%


    def add_min_generalization(self, hypothesis, training_example, G, S):
        """
        Makes the hypothesis consistent with training_example
        Adds it to S if some member of G is more general
        """
        # %%% TODO START YOUR CODE HERE %%%
        for i in range(len(hypothesis)):
            if hypothesis[i] != training_example[i]:
                hypothesis[i] = '?'
        for hyp in G:
            if self.is_equal(hypothesis, hyp):
                S.append(hypothesis)
        if (self.is_less_general_than_any(hypothesis, G)):
            S.append(hypothesis)

        return(S)

        # %%% END YOUR CODE HERE %%%

    def is_more_general_than_any(self, hypothesis, set):
        """
        Checks if the hypothesis is more general than any hypothesis in the set
        """
        for h in set:
            if self.is_more_general(hypothesis, h):
                return True
        return False
    
    def find_attr_variants(self, training_data):
        variants = []
        for ex in training_data:
            for attr in ex:
                at = []
                variants.append(at)
                
        for ex in training_data:
            for i in range(len(ex)):
                if ex[i] not in variants[i]:
                    variants[i].append(ex[i])
        return(variants)
            
    def add_min_specialization(self, hypothesis, training_example, S, G, training_data):
        """
        Generates all possible minimal specializations by replacing '?' by all possible values except for value in negative example
        Adds each such specialization to G if some member of S is more specific than the specialization
        """
        # %%% TODO START YOUR CODE HERE %%%
        specs = []
        variants = self.find_attr_variants(training_data)
        for i in range(len(training_example)):
            for j in range(len(variants[i])):
                h = hypothesis.copy()
                if variants[i][j] != training_example[i]:
                    h[i] = variants[i][j]
                    specs.append(h)
                    
        for spec in specs:
            if (self.is_more_general_than_any(spec, S)):
                G.append(spec)
        
        # %%% END YOUR CODE HERE %%%
        return(G)
        


    def is_less_general_than_any(self, hypothesis, set):
        """
        Checks if the hypothesis is less general than any hypothesis in the set
        """
        for h in set:
            if self.is_more_general(h, hypothesis):
                return True
        return False

    def is_more_general(self, hypothesis1, hypothesis2):
        """
        Returns True if hypothesis1 is more general than hypothesis2
        """
        # %%% TODO START YOUR CODE HERE %%%
        equal = False
        if self.is_equal(hypothesis1, hypothesis2):
            return False
        for i in range(len(hypothesis1)):
            if(hypothesis1[i] != hypothesis2[i] and hypothesis1[i] != '?'):
                return False
            if hypothesis2[i] == '?' and hypothesis1[i] !='?':
                return False
            if hypothesis1[i] == '?' and hypothesis2[i] != '?':
                equal = True
        return equal

        # %%% END YOUR CODE HERE %%%


    def is_equal(self, hypothesis1, hypothesis2):
        """
        Returns True if hypotheses are equal
        """
        for i in range(len(hypothesis1)):
            if hypothesis1[i] != hypothesis2[i]:
                return False
        return True

print("For data1")
CE = CandidateElimination()

X, Y = load_data("data_1.csv")
CE.fit(X, Y)

print("For cars")

CE = CandidateElimination()

X, Y = load_data("cars.csv")
CE.fit(X, Y)