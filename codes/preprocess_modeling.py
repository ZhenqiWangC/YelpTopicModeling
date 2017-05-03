import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2
import numpy as np
import sklearn.tree
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import sklearn.ensemble

review = pd.read_csv("../data/selectedreviews.csv")


#preprocess

star = np.array(review.stars)
text = list(map(lambda x: x[2:-1].replace("\\n","\n"),review.text))
pat = re.compile(r"[^\w\s]")
text_clean = np.array(list(map(lambda x: pat.sub(" ",x).lower(),text)))

#create TF-IDF

vectorizer = TfidfVectorizer(stop_words="english")
text_features = vectorizer.fit_transform(text_clean)
vocab = vectorizer.get_feature_names()

################# Multinomial Regression #################

n_fold=5
n_words=1000
kf=KFold(n_fold,shuffle=True)
parameters=10.0**np.arange(0,8)
acc_mat=np.zeros([n_fold,len(parameters)])
acc_mat_train=np.zeros([n_fold,len(parameters)])
k=0
for train_idx,validate_idx in kf.split(text_features):
    text_features_train=text_features[train_idx]
    star_train=star[train_idx]
    text_features_validate=text_features[validate_idx]
    star_validate=star[validate_idx]
    fselect = SelectKBest(chi2,k=n_words)
    text_features_train = fselect.fit_transform(text_features_train,star_train)
    text_features_validate=text_features_validate[:,fselect.get_support()]
    vocab_current = np.array(vocab)[fselect.get_support()]
    print("Train Feature Space Shape:",text_features_train.shape)
    print("Test Feature Space Shape:",text_features_validate.shape)
    print("Last Three Selected Features:",vocab_current.tolist()[-3:])
    ##########################################
    #    Build Models and Tune Parameters    #
    #         Take Tree as an Example        #
    #             Tuning max_depth           #
    ##########################################
    t=0
    for para in parameters:
        mod_temp=LogisticRegression(C=para)
        mod_temp.fit(X=text_features_train,y=star_train)
        pred=mod_temp.predict(X=text_features_validate)
        acc_mat[k,t]=np.mean(pred==star_validate)
        pred=mod_temp.predict(X=text_features_train)
        acc_mat_train[k,t]=np.mean(pred==star_train)
        t+=1
    k+=1
print(np.mean(acc_mat,axis=0))

################## Random Forest #################
n_fold=10
n_words=500
kf=KFold(n_fold,shuffle=True)
#n_estimators = [10,20,30,40,50,60,70,80,90,100,150,200,250]
n_estimators = [10,50,100,150,200]
max_features = [i for i in range(15,26)] ## best 24
max_depth = [30,40,50,60,70] ## best 40
#acc_mat_forest=np.zeros([n_fold,len(n_estimators)])
mse_mat_forest=np.zeros([n_fold,len(n_estimators)*len(max_features)*len(max_depth)])
k=0
for train_idx,validate_idx in kf.split(text_features):
    text_features_train=text_features[train_idx]
    star_train=star[train_idx]
    text_features_validate=text_features[validate_idx]
    star_validate=star[validate_idx]
    fselect = SelectKBest(chi2 , k=n_words)
    text_features_train = fselect.fit_transform(text_features_train,star_train)
    text_features_validate=text_features_validate[:,fselect.get_support()]
    vocab_current = np.array(vocab)[fselect.get_support()]
    #print("Train Feature Space Shape:",text_features_train.shape)
    #print("Test Feature Space Shape:",text_features_validate.shape)
    print("Last Three Selected Features:",vocab_current.tolist()[-3:])
    ##############################################
    #    Build Models and Tune Parameters        #
    #         Take Tree as an Example            #
    #Tuning n_estimators, max_features，max_depth#
    ##############################################
    t=0
    for est in n_estimators:
        for f in max_features:
            for d in max_depth:
                mod_temp=sklearn.ensemble.RandomForestRegressor(n_estimators = est, max_features = f, max_depth = d)
                mod_temp.fit(X=text_features_train,y=star_train)
                pred=mod_temp.predict(X=text_features_validate)
                #acc_mat_forest[k,t]=np.mean(pred==star_validate)
                mse_mat_forest[k,t]= np.sum((pred-star_validate)**2)
                t+=1
                print("t = ",t)
    k+=1
    print("k = ", k)
# vaverage prediction accuracy
#print(np.mean(acc_mat_forest,axis=0))
print(np.mean(mse_mat_forest/len(pred),axis=0))

##############Multi-Layer Percepton#################
n_fold=3
n_words=500
kf=KFold(n_fold,shuffle=True)
hidden_layer_sizes = [10,100,200,500]
#max_features = []
acc_mat_forest=np.zeros([n_fold,len(hidden_layer_sizes)])
mse_mat_forest=np.zeros([n_fold,len(hidden_layer_sizes)])
k=0
for train_idx,validate_idx in kf.split(text_features):
    text_features_train=text_features[train_idx]
    star_train=star[train_idx]
    text_features_validate=text_features[validate_idx]
    star_validate=star[validate_idx]
    fselect = SelectKBest(chi2 , k=n_words)
    text_features_train = fselect.fit_transform(text_features_train,star_train)
    text_features_validate=text_features_validate[:,fselect.get_support()]
    vocab_current = np.array(vocab)[fselect.get_support()]
    #print("Train Feature Space Shape:",text_features_train.shape)
    #print("Test Feature Space Shape:",text_features_validate.shape)
    print("Last Three Selected Features:",vocab_current.tolist()[-3:])
    ##############################################
    #    Build Models and Tune Parameters        #
    #         Take Tree as an Example            #
    #Tuning n_estimators, max_features，max_depth#
    ##############################################
    t=0
    for par in hidden_layer_sizes:
        mod_temp=MLPClassifier(solver='lbfgs', alpha=1e-5,
                               hidden_layer_sizes=(par,par), random_state=1)
        mod_temp.fit(X=text_features_train,y=star_train)
        pred=mod_temp.predict(X=text_features_validate)
        acc_mat_forest[k,t]=np.mean(pred==star_validate)
        mse_mat_forest[k,t]= np.sum((pred-star_validate)**2)
        t+=1
        print("t = ",t)
    k+=1
    print("k = ", k)
# vaverage prediction accuracy
print(np.mean(acc_mat_forest,axis=0))
print(np.mean(mse_mat_forest/len(pred),axis=0))

################### Multilayer Percepton with logistic activation function #################
n_fold=3
n_words=500
kf=KFold(n_fold,shuffle=True)
hidden_layer_sizes = [10,100,200,500,1000]
#max_features = []
acc_mat_forest=np.zeros([n_fold,len(hidden_layer_sizes)])
mse_mat_forest=np.zeros([n_fold,len(hidden_layer_sizes)])
k=0
for train_idx,validate_idx in kf.split(text_features):
    text_features_train=text_features[train_idx]
    star_train=star[train_idx]
    text_features_validate=text_features[validate_idx]
    star_validate=star[validate_idx]
    fselect = SelectKBest(chi2 , k=n_words)
    text_features_train = fselect.fit_transform(text_features_train,star_train)
    text_features_validate=text_features_validate[:,fselect.get_support()]
    vocab_current = np.array(vocab)[fselect.get_support()]
    #print("Train Feature Space Shape:",text_features_train.shape)
    #print("Test Feature Space Shape:",text_features_validate.shape)
    print("Last Three Selected Features:",vocab_current.tolist()[-3:])
    ##############################################
    #    Build Models and Tune Parameters        #
    #         Take Tree as an Example            #
    #Tuning n_estimators, max_features，max_depth#
    ##############################################
    t=0
    for par in hidden_layer_sizes:
        mod_temp=MLPClassifier(solver='lbfgs', alpha=1e-5,activation="logistic",
                               hidden_layer_sizes=(par,par,par), random_state=1)
        mod_temp.fit(X=text_features_train,y=star_train)
        pred=mod_temp.predict(X=text_features_validate)
        acc_mat_forest[k,t]=np.mean(pred==star_validate)
        mse_mat_forest[k,t]= np.sum((pred-star_validate)**2)
        t+=1
        print("t = ",t)
    k+=1
    print("k = ", k)
# vaverage prediction accuracy
print(np.mean(acc_mat_forest,axis=0))
print(np.mean(mse_mat_forest/len(pred),axis=0))




####REGERESSION###
##############Multi-Layer Percepton#################
n_fold=3
n_words=500
kf=KFold(n_fold,shuffle=True)
hidden_layer_sizes = [10,100,200,500]
#max_features = []
mse_mat_forest=np.zeros([n_fold,len(hidden_layer_sizes)])
k=0
for train_idx,validate_idx in kf.split(text_features):
    text_features_train=text_features[train_idx]
    star_train=star[train_idx]
    text_features_validate=text_features[validate_idx]
    star_validate=star[validate_idx]
    fselect = SelectKBest(chi2 , k=n_words)
    text_features_train = fselect.fit_transform(text_features_train,star_train)
    text_features_validate=text_features_validate[:,fselect.get_support()]
    vocab_current = np.array(vocab)[fselect.get_support()]
    #print("Train Feature Space Shape:",text_features_train.shape)
    #print("Test Feature Space Shape:",text_features_validate.shape)
    print("Last Three Selected Features:",vocab_current.tolist()[-3:])
    ##############################################
    #    Build Models and Tune Parameters        #
    #         Take Tree as an Example            #
    #Tuning n_estimators, max_features，max_depth#
    ##############################################
    t=0
    for par in hidden_layer_sizes:
        mod_temp=MLPRegressor(solver='lbfgs', alpha=1e-5,
                               hidden_layer_sizes=(par,par), random_state=1)
        mod_temp.fit(X=text_features_train,y=star_train)
        pred=mod_temp.predict(X=text_features_validate)
        mse_mat_forest[k,t]= np.mean((pred-star_validate)**2)
        t+=1
        print("t = ",t)
    k+=1
    print("k = ", k)
# vaverage prediction accuracy
print(np.mean(mse_mat_forest,axis=0))

################### Multilayer Percepton with logistic activation function #################

n_fold=3
n_words=500
kf=KFold(n_fold,shuffle=True)
hidden_layer_sizes = [10,100,200,500,1000]
#max_features = []
acc_mat_forest=np.zeros([n_fold,len(hidden_layer_sizes)])
mse_mat_forest=np.zeros([n_fold,len(hidden_layer_sizes)])
k=0
for train_idx,validate_idx in kf.split(text_features):
    text_features_train=text_features[train_idx]
    star_train=star[train_idx]
    text_features_validate=text_features[validate_idx]
    star_validate=star[validate_idx]
    fselect = SelectKBest(chi2 , k=n_words)
    text_features_train = fselect.fit_transform(text_features_train,star_train)
    text_features_validate=text_features_validate[:,fselect.get_support()]
    vocab_current = np.array(vocab)[fselect.get_support()]
    #print("Train Feature Space Shape:",text_features_train.shape)
    #print("Test Feature Space Shape:",text_features_validate.shape)
    print("Last Three Selected Features:",vocab_current.tolist()[-3:])
    ##############################################
    #    Build Models and Tune Parameters        #
    #         Take Tree as an Example            #
    #Tuning n_estimators, max_features，max_depth#
    ##############################################
    t=0
    for par in hidden_layer_sizes:
        mod_temp=MLPRegressor(solver='lbfgs', alpha=1e-5,activation="logistic",
                               hidden_layer_sizes=(par,par,par), random_state=1)
        mod_temp.fit(X=text_features_train,y=star_train)
        pred=mod_temp.predict(X=text_features_validate)
        mse_mat_forest[k,t]= np.mean((pred-star_validate)**2)
        t+=1
        print("t = ",t)
    k+=1
    print("k = ", k)
# vaverage prediction accuracy
print(np.mean(mse_mat_forest,axis=0))
