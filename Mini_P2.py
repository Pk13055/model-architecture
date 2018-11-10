#!/usr/bin/env python
# coding: utf-8

# In[7]:


from multiprocessing import Pool
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


# # Data Processing
# 
# First we need to load the data from the pickle objects and perform different preprocessing techniques.
# 
# ## Loading the Data

# In[8]:


def load_cifar():
    
    trn_data, trn_labels, tst_data, tst_labels = [], [], [], []
    unpickle = lambda file: pickle.load(open(file, 'rb'), encoding='latin1')
    for i in range(5):
        batchName = f'./data/cifar-10-batches-py/data_batch_{i + 1}'
        unpickled = unpickle(batchName)
        trn_data.extend(unpickled['data'])
        trn_labels.extend(unpickled['labels'])
    unpickled = unpickle('./data/cifar-10-batches-py/test_batch')
    tst_data.extend(unpickled['data'])
    tst_labels.extend(unpickled['labels'])
    return trn_data, trn_labels, tst_data, tst_labels


# ## Pre-processing
# 
# After loading the data, we can use the function below to apply different preprocessing techniques to allow for better training and feature extraction.

# In[9]:


def image_prep(X, y, onehot=True):
    ''' pre-processes the given image
        performs mean normalization and other such operations'''
    scaler = StandardScaler(copy=False)
    X_ = scaler.fit_transform(X)
    y_ = y
    onehot = None
    if onehot:
        onehot = OneHotEncoder()
        y_ = onehot.fit_transform(y)
    return X_, y_, onehot


# ## Dimensionality Reduction
# 
# There is an option to train on the complete set of **raw** images but this can lead to overfitting. Instead we can use PCA or LDA to extract and keep **only** those dimensions that best represent the data.

# In[10]:


def reduce_dim(X, y, **kwargs):
    ''' performs dimensionality reduction'''
    method = kwargs.pop('method', None)
    if method in ['pca', 'lda']:
        n_components = kwargs.pop('n_components', None)
    if method == 'pca':
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)
    elif method == 'lda':
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X = lda.fit(X, y).transform(X)
    elif method == 'raw':
        pass
    else:
        print("Dimensionality reduction method not supported")
    return X


# # Classification
# 
# For classification, we prepare this model factory that takes in the type of decision maker and returns a model on which you can train. The various classification methods covered under this are:
# - SVM:
#     - Linear Kernel
#     - Gaussian Kernel
#     - Polynomial Kernel
# - Logistic Regression
# - MLP
# - Decision Tree
# - Gradient Boosting

# In[11]:


def classify(X, y, **kwargs):
    ''' trains a classifier by taking input features
        and their respective targets and returns the trained model'''
    method = kwargs.pop('method')
    options = kwargs.pop('options', {})
    if method == 'SVM':
        clf = svm.SVC(kernel='linear', **options)
    elif method == 'RBF':
        clf = svm.SVC(kernel='rbf')
    elif method == 'logistic':
        clf = LogisticRegression(**options)
    elif method == 'MLP':
        clf = MLPClassifier(**options)
    elif method == 'CART':
        clf = DecisionTreeClassifier(**options)
    elif method == 'grad_boost':
        clf = GradientBoostingClassifier(**options)
    else:
        print("Classifier not supported")
        return None, None
    clf.fit(X, y)
    return clf, method


# # Evaluation
# 
# Now that we have all the tools ready for the testing of various methods of classification, we can quantitavely run each classification method with our choice of preprocessing and evaluate the results. Our evaluation metric of choice is the accuracy and F1 score. The F1 score can be represented as follows:
# $$ F_1 = 2 * \frac{ \text{precision} \times \text{recall}}{\text{precision} + \text{recall}} $$

# In[16]:


def test(train_X, train_y, test_X, test_y, meta):
    '''takes test data and trained classifier model,
    performs classification and prints accuracy and f1-score'''
    model_kwargs, processing_kwargs = meta['model'], meta['preprocess']
    train_X = reduce_dim(train_X, train_y, **processing_kwargs)
    model_name, preprocessor = model_kwargs['method'], processing_kwargs['method']
    print(f"Model {model_name} | Preprocessing {preprocessor} started")
    model = classify(train_X, train_y, **model_options)
    predictions = model.predict(test_X)
    accuracy = accuracy_score(test_y, predictions)
    F1 = f1_score(test_y, predictions, average='micro')
    print(f"Model {model_kwargs} | Preprocessing {processing_kwargs} : {(accuracy, F1)}")
    return accuracy, F1


# In[17]:


def evaluate(train_X, train_y, test_X, test_y):
    """Evaluates the various models against params
    :return result: np.array -> The accuracy and F1 score for various training options
    """
    results, metadata = [], []
    classifiers = ["SVM", "RBF", "logistic", "MLP", "CART", 'grad_boost']
    preprocessing = ['raw', 'pca', 'lda']
    for model_name in classifiers:
        model_options = {'method' : model_name }
        for preprocessor in preprocessing:
            preprocessing_options = {'method' : preprocessor }
            metadata.append({'model' : model_options, 
                             'preprocess' : preprocessing_options })
    pool = Pool(8)
    results = pool.starmap_async(test, [(train_X, train_y, test_X, test_y, _) for _ in metadata])
    results.get()
    return results, metadata


# In[18]:


def main():
    train_X, train_y, test_X, test_y = [np.array(_) for _ in load_cifar()]
    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)
    print("Before processing", [_.shape for _ in [train_X, train_y, test_X, test_y]])
    train_X, train_y, onehot = image_prep(train_X, train_y, onehot=False)
#     test_y = onehot.transform(test_y)
    results, meta_ = evaluate(train_X, train_y, test_X, test_y)
    print(results)


# In[ ]:


if __name__ == '__main__':
    main()

