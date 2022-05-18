import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


#  Using the modules I sent them
# Resampling RR using scikit learn ridge
class resamplingRR:
    """ This class manages a resampling ridge regression predictor.
    It allows to train and predict on new datapoints
    To estimate the uncertainty, it constructs N_MODELS models, based on
    a resampling from the dataset at a fixed percentage of RESAMPLING_RATIO.
    The train, test and calibration sets must be provided externally"""

    def __init__(self,n_models=64,verbose=False):
        self.alpha = 1.0
        self.n_models = n_models
        self.resampling_ratio = 0.8
        self.models = []
        self.verbose = verbose

    def optimise(self,X,y,lrange):
        if self.verbose : print("1.  Optimizing")
        # Model selection  
        alphas = lrange
        # create and fit a ridge regression model, testing each alpha
        model = Ridge()
        grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas),scoring="neg_root_mean_squared_error")
        grid.fit(X, y)
        # summarize the results of the grid search
        if self.verbose:
            print("Best RMSE: {:.3f}".format(-grid.best_score_))
            print("With regularisation : {:.3E}".format(grid.best_estimator_.alpha))
        self.reg = grid.best_estimator_.alpha

    def train(self, X,y):
        from numpy.random import choice
        if self.verbose : print("2.  Training")
        training_size = int(len(X)*self.resampling_ratio)
        self.indices = np.zeros((self.n_models,training_size),dtype='int')
        for n_r in range(self.n_models):
            self.indices[n_r] = choice(len(X),training_size,replace=False)
            model = Ridge(alpha=self.reg)
            model.fit(X[self.indices[n_r]],y[self.indices[n_r]])
            self.models.append(model)

    def write_weights(self,directory):
        np.save(directory+"/model.npy",self.models,allow_pickle=True)
        np.save(directory+"/indices.npy",self.indices)
        np.save(directory+"/scaling.npy",self.alpha)
        pass
   
    def load_weights(self,directory):
        self.models  = np.load(directory+"/model.npy",allow_pickle=True)
        self.n_models = len(self.models)
        self.indices =  np.load(directory+"/indices.npy")
        self.alpha = np.load(directory+"/scaling.npy")
        pass

    def calibrate(self, X,y):
        if self.verbose : print("3.  Calibrating")
        #Initialise arrays
        predictions = np.zeros((self.n_models,len(X)))
        # Predict on the calibration set
        for n in range(self.n_models):
            predictions[n] = self.models[n].predict(X)
        pred_means = predictions.mean(axis=0)
        pred_var = predictions.var(axis=0,ddof=1)
        # Estimating variance scaling
        # DOI : arXiv:2011.08828
        M = self.n_models
        alpha2 = -1/M + (M-3)/(M-1) * np.mean((y - pred_means)**2 / pred_var)
        self.alpha = np.sqrt(alpha2)
        if self.verbose : print("Variance scaler: {:.3f}".format(self.alpha))

    def predict(self,X):
        if self.verbose : print("4.  Predicting")
        #Initialise arrays
        predictions = np.zeros((self.n_models,len(X)))
        for n in range(self.n_models):
            predictions[n] = self.models[n].predict(X)
        if self.alpha == 0 : print("Careful, you haven't calibrated your models")
        scaled_predictions = predictions * 0.0 + np.mean(predictions,axis=0)
        scaled_predictions += self.alpha * (predictions - np.mean(predictions,axis=0))
        return scaled_predictions