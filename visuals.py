import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")

import matplotlib.pyplot as pl

import cupy as cp
import numpy as np

# import cuml
# from cuml import make_regression, train_test_split
# from cuml.linear_model import LinearRegression as cuLR
# from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit, train_test_split

random_state=0

def ModelLearning(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=random_state)

    train_sizes = cp.rint(cp.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)
    
    fig = pl.figure(figsize=(10,7))

    for k, depth in enumerate([1,3,6,10]):
        regressor = DecisionTreeRegressor(max_depth=depth)

        sizes, train_scores, test_scores = learning_curve(regressor, X, y, cv=cv, train_sizes=train_sizes, scoring='r2')
        
        train_std = cp.std(train_scores, axis=1)
        train_mean = cp.mean(train_scores, axis=1)
        test_std = cp.std(test_scores, axis=1)
        test_mean = cp.mean(test_scores, axis=1)
        
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, train_mean, 'o-', color='r', label='Training Score')
        ax.plot(sizes, test_mean, 'o-', color='g', label='Testing Score')
        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='r')
        ax.fill_between(sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='g')

        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
        ax.set_xlim([0, X.shape[0]*0.8])
        ax.set_ylim([-0.05, 1.05])
        ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad=0.)
        
    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize=16, y=1.03)
    fig.tight_layout()
    fig.show()

def ModuleComplexity(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=random_state)

    max_depth = cp.arange(1,11)

    train_scores, test_scores = validation_curve(DecisionTreeRegressor(), X, y, param_name="max_depth", param_range=max_depth, cv=cv, scoring='r2')

    train_std = cp.std(train_scores, axis=1)
    train_mean = cp.mean(train_scores, axis=1)
    test_std = cp.std(test_scores, axis=1)
    test_mean = cp.mean(test_scores, axis=1)

    pl.figure(figsize=(7, 5))
    pl.title('Decision Tree Regressor Complexity Performance')
    pl.plot(max_depth, train_mean, 'o-', color='r', label='Training Score')
    pl.plot(max_depth, test_mean, 'o-', color='g', label='Validation Score')
    pl.fill_between(max_depth, train_mean - train_std, train_mean + train_std, alpha=0.15, color='r')
    pl.fill_between(max_depth, test_mean - test_std, test_mean + test_std, alpha=0.15, color='g')

    pl.legend(loc = 'lower right')
    pl.xlabel('Maximum Depth')
    pl.ylabel('Score')
    pl.ylim([-0.05,1.05])
    pl.show()

def PredictTrials(X, y, fitter, data):
    prices = []

    for k in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=k)

        reg = fitter(X_train, y_train)

        pred = reg.predict([data[0]])[0]
        prices.append(pred)

        print("Trial {}: ${:,.2f}".format(k+1, pred))
    
    print("\nRange in prices: ${:,.2f}".format(max(prices) - min(prices)))


# def PredictTrials(X, y, fitter, data):
#     prices = []
#
#     for k in range(10):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=k)
#        
#         ols_cuml = cuLR(fit_intercept=True, normalize=True, algorithm='eig')
#        
#         ols_cuml.fit(X_train, y_train)
#        
#         predict_cuml = ols_cuml.predict_cuml([data(0)])(0)
#         prices.append(predict_cuml)
#        
#         print("Trial {}: ${:,.2f}".format(k+1, predict_cuml))