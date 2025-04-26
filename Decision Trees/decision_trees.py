import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold = None, left = None, right = None, *, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None #if value exists it is a leaf node

class DecisionTree:
    def __init__(self, min_samples_split = 2, max_depth = 100, n_features=None):

        #stopping cretaria
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
        self.n_features = n_features
        self.root=None #Root Node

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features) #Ensuring n_features do not exceed the input shape
        self.root = self._grow_tree(X,y)
 
    def _grow_tree(self, X,y, depth=0): 
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        #check the stopping cretiria
        if(depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y) #if stopping criteria is there, we will return the value using most common label
            return Node(value=leaf_value) #returning the value of the leaf node
        

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False) #Getting the features for checking best split
        #find the best split
        best_feature, best_threshold= self._best_split(X, y, feat_idxs) #feat_idxs to identify the features from which the next split will be

        #create child nodes
        left_idxs, right_idxs = self._split_children(X[:, best_feature], best_threshold)
        left_child = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right_child = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_threshold, left_child, right_child)

    def _best_split(self, X, y, feat_idxs): #Finding all possible thresholds and splits to identify the best one
        best_gain = -1 
        split_idx, split_threshold = None, None
        
        for feat_idx in feat_idxs: 
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

        for thr in thresholds: 
            #calculate the information gain 
            gain = self._information_gain(y, X_column, thr)

            if gain>best_gain: 
                best_gain = gain
                split_idx = feat_idx 
                split_threshold = thr

        return split_idx, split_threshold
         

    def _information_gain(self, y, X_column , threshold):  #Entropy(parent) - weightedavg.Entropy(children)
        #calculate parent entropy
        parent_entropy = self._entropy(y)

        #create children
        left_idxs, right_idxs = self._split_children(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:  #information gain is 0 in this case
            return 0

        #calculate the weighted average entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs) #number of samples in each
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs]) #entropy of each
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r #weighted child entropy

        #calculate information gain
        information_gain = parent_entropy - child_entropy 
        return information_gain

    def _split_children(self, X_column, threshold): 
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y): #calculating entropy
        hist = np.bincount(y) #Getting the occurence count for each value
        ps = hist / len(y) #Getting the number of times x has occured divided by the total n
        entropy = -np.sum([p*np.log(p) for p in ps if p>0])
        return entropy

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0] #from counter doc getting the first most value
        return value
    
    def predict(self, X): 
        prediction = np.array([self._traverse_tree(x, self.root) for x in X]) #traversing the whole tree for all values of X starting from the root node
        return prediction 
    
    def _traverse_tree(self, x, node): 
        if node.is_leaf_node(): #if leaf node returning the value
            return node.value
        
        if x[node.feature] <= node.threshold: # if value is less than threshold for that node, going to the left side
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right) # right side