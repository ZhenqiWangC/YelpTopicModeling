import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2
import numpy as np
import sklearn.tree

##put this file into where you files are or change the path below
review = pd.read_csv("yelp_academic_dataset_review.csv")
business = pd.read_csv("yelp_academic_dataset_business.csv")

#preprocess
business_nc = business[business.state == "b'NC'"]
nc_bus_id = business_nc["business_id"]
review_nc = review[review.business_id.isin(nc_bus_id)]
star_nc = review_nc.stars.tolist()
text_nc = list(map(lambda x: x[2:-1].replace("\\n","\n"),review_nc.text))
pat = re.compile(r"[^\w\s]")
text_nc_clean = list(map(lambda x: pat.sub("",x).lower(),text_nc))

#k=1700 can be changed 
vectorizer = TfidfVectorizer(stop_words="english")
text_features = vectorizer.fit_transform(text_nc_clean)
vocab = vectorizer.get_feature_names()
fselect = SelectKBest(chi2 , k=1700)
text_features = fselect.fit_transform(text_features,star_nc)
vocab = np.array(vocab)[fselect.get_support()]

#try tree model
treemod=sklearn.tree.DecisionTreeClassifier()
treemod.fit(X=text_features,y=star_nc)
np.mean(treemod.predict(text_features)==star_nc)