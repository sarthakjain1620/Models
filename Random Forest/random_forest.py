import numpy as np 
from collections import Counter
from decision_trees import DecisionTree

class RandomForest(): 
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None): 
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = [] #keeping all the trees value

    def fit(self,X,y): 
        self.trees = []
        
        for _ in range(self.n_trees): 
            tree = DecisionTree(max_depth=self.max_depth, 
                                min_samples_split=self.min_samples_split, 
                                n_features=self.n_features)
            
            X_sample, y_sample = self._bootstrap_samples(X,y) #subset of dataset
            tree.fit(X_sample, y_sample) #fitting each tree
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y): 
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True) #returns array indices with replacement, n_samples creating array
        return X[idxs], y[idxs]
    
    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0] #from counter doc getting the first most value
        return value
    
    #initially the tree predictions will be like [[1,0,0], [0,1,0]] where each list represents each tree predictions on different samples, 
    # np swapaxes swaps axes and now each list contain predictions from each sample for each tree
    
    def predict(self,X): 
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1) 
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds]) #array for all samples
        return predictions