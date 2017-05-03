import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2
import numpy as np
import sklearn.tree
from sklearn.model_selection import KFold

##put this file into where you files are or change the path below
review = pd.read_csv("yelp_academic_dataset_review.csv")
business = pd.read_csv("yelp_academic_dataset_business.csv")

#preprocess
business_nc = business[business.state == "b'NC'"]
nc_bus_id = business_nc["business_id"]
review_nc = review[review.business_id.isin(nc_bus_id)]
star_nc = np.array(review_nc.stars)
text_nc = list(map(lambda x: x[2:-1].replace("\\n","\n"),review_nc.text))
pat = re.compile(r"[^\w\s]")
text_nc_clean = np.array(list(map(lambda x: pat.sub("",x).lower(),text_nc)))

#create TF-IDF

vectorizer = TfidfVectorizer(stop_words="english")
text_features = vectorizer.fit_transform(text_nc_clean)
vocab = vectorizer.get_feature_names()

#CV
n_fold=3
n_words=100
kf=KFold(n_fold,shuffle=True)
parameters=[1,5,10,50,100]
acc_mat=np.zeros([n_fold,len(parameters)])
k=0
for train_idx,validate_idx in kf.split(text_features):
    text_features_train=text_features[train_idx]
    star_nc_train=star_nc[train_idx]
    text_features_validate=text_features[validate_idx]
    star_nc_validate=star_nc[validate_idx]
    fselect = SelectKBest(chi2 , k=n_words)
    text_features_train = fselect.fit_transform(text_features_train,star_nc_train)
    text_features_validate=text_features_validate[:,fselect.get_support()]
    vocab_current = np.array(vocab)[fselect.get_support()]
    print("Train Feature Space Shape:",text_features_train.shape)
    print("Test Feature Space Shape:",text_features_validate.shape)
    print("Selected Features:",vocab_current.tolist())
    ##########################################
    #    Build Models and Tune Parameters    #
    #         Take Tree as an Example        #
    #             Tuning max_depth           #
    ##########################################
    t=0
    for para in parameters:
        mod_temp=sklearn.tree.DecisionTreeClassifier(max_depth=para)
        mod_temp.fit(X=text_features_train,y=star_nc_train)
        pred=mod_temp.predict(X=text_features_validate)
        acc_mat[k,t]=np.mean(pred==star_nc_validate)
        t+=1
    k+=1
np.mean(acc_mat,axis=0)

import sklearn.ensemble
# tuned hyperparameters: n_estimators, max_features='auto'，max_depth=100
n_fold=3
n_words=1000
kf=KFold(n_fold,shuffle=True)
n_estimators = [5,10,50,100]
#max_features = []
acc_mat_forest=np.zeros([n_fold,len(n_estimators)])
mse_mat_forest=np.zeros([n_fold,len(n_estimators)])
k=0
for train_idx,validate_idx in kf.split(text_features):
    text_features_train=text_features[train_idx]
    star_nc_train=star_nc[train_idx]
    text_features_validate=text_features[validate_idx]
    star_nc_validate=star_nc[validate_idx]
    fselect = SelectKBest(chi2 , k=n_words)
    text_features_train = fselect.fit_transform(text_features_train,star_nc_train)
    text_features_validate=text_features_validate[:,fselect.get_support()]
    vocab_current = np.array(vocab)[fselect.get_support()]
    #print("Train Feature Space Shape:",text_features_train.shape)
    #print("Test Feature Space Shape:",text_features_validate.shape)
    print("Selected Features:",vocab_current.tolist())
    ##############################################
    #    Build Models and Tune Parameters        #
    #         Take Tree as an Example            #
    #Tuning n_estimators, max_features，max_depth#
    ##############################################
    t=0
    for par in n_estimators:
        mod_temp=sklearn.ensemble.RandomForestClassifier(n_estimators = par)
        mod_temp.fit(X=text_features_train,y=star_nc_train)
        pred=mod_temp.predict(X=text_features_validate)
        acc_mat_forest[k,t]=np.mean(pred==star_nc_validate)
        mse_mat_forest[k,t]= np.sum((pred-star_nc_validate)**2)
        t+=1
        print("t = ",t)
    k+=1
    print("k = ", k)
# vaverage prediction accuracy
print(np.mean(acc_mat_forest,axis=0))
print(np.mean(mse_mat_forest/len(pred),axis=0))
