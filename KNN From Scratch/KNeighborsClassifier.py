class knn:
    
    
    def __init__(self, k):
        self.n_neighbors = k
        self.X_train = None
        self.y_train = None
    
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    
    def predict(self, X_test):
        