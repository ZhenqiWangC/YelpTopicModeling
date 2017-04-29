import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest,chi2
import re
import string
from nltk.corpus import stopwords
import glob
from sklearn.feature_extraction.text import TfidfVectorizer

def tune_spam(X_train,y_train,alpha_list):
    val_accuracy=[]
    for alpha in alpha_list:
        model = SVC(C=alpha)
        val_accuracy.extend([np.mean(cross_val_score(model, X_train, y_train, cv=2, scoring='accuracy'))])
        print [np.mean(cross_val_score(model, X_train, y_train, cv=2, scoring='accuracy'))]
    max_index =  val_accuracy.index(max( val_accuracy))
    print "CV_val_error:", val_accuracy
    print "Best C:",alpha_list[max_index]
    return alpha_list[max_index]



def load_txt(file):
    file = open(file, 'r')
    text = file.read().lower()
    file.close()
    text = text.replace('\r\n', ' ')
    text= re.sub("[^a-z$!:?</@#%([\*]", " ", text)
    #text.translate(None, string.punctuation)
    words = text.split()
    stop = set(stopwords.words('english'))
    #words = [i for i in words if i not in stop]
    return " ".join(words)

def txt_to_array(file_match):
    text_array = []
    onlyfiles =glob.glob(file_match)
    for i in range(len(onlyfiles)):
        text_array.append(load_txt(onlyfiles[i]))
    return text_array

def test_to_array():
    text_array = []
    for i in range(5857):
        text_array.append(load_txt("hw01_data/spam/test/"+str(i)+".txt"))
    return text_array

if __name__ == "__main__":
    ham_x = txt_to_array("hw01_data/spam/ham/*.txt")
    spam_x = txt_to_array("hw01_data/spam/spam/*.txt")
    test_x = test_to_array()
    label = np.zeros(len(ham_x))
    label = np.concatenate((label,np.ones(len(spam_x))))
    x = np.concatenate((ham_x,spam_x))
    vectorizer = TfidfVectorizer(ngram_range = (1,3),analyzer="word", lowercase=False)
    vectorizer = vectorizer.fit(x.tolist())
    train_data_features = vectorizer.transform(x.tolist())
    fselect = SelectKBest(chi2 , k=1700)
    train_data_features = fselect.fit_transform(train_data_features,label)
    train_data_features = train_data_features.toarray()
    test = vectorizer.transform(test_x)
    test = fselect.transform(test)
    test = test.toarray()

    #parameters= [0.00001,0.001,0.01,10,100,1000]
    #CV_val_error: [0.70997679814385151, 0.70997679814385151, 0.70997679814385151,
    #  0.70997679814385151, 0.71017014694508895, 0.86755607115235889]
    #ngram_range = (1, 4), k=3000, parameters= [1000,2000,3000,4000,5000]
    #CV_val_error: [0.91647331786542918, 0.9532095901005414, 0.9624903325599381, 0.9624903325599381, 0.96287703016241299]
    #ngram_range = (1, 4), k=3000,parameters= [6000,7000,8000,9000,10000]
    #[0.96287703016241299, 0.96307037896365033, 0.96326372776488789, 0.96210363495746321, 0.96210363495746321]
    #ngram_range = (1, 2), k=2000,parameters= [7500,8000,8500]
    #[0.96597061098221193, 0.96558391337973704, 0.9650038669760248]
    #ngram_range = (1, 2), k=2000, parameters= [6500,6800]
    # [0.96558391337973704] [0.9653905645784997]
    #ngram_range = (1, 2), k=2000, parameters= [3000,4000,5000,6000]
    # CV_val_error: [0.96345707656612523, 0.96403712296983757, 0.9650038669760248, 0.96481051817478725]
    #ngram_range = (1, 2), k=2500 parameters= [5000,6000,700,800]
    #[0.96403712296983757, 0.96442382057231246, 0.95726991492652747, 0.9603634957463264]
    #ngram_range = (1, 3), k=2500 parameters= [5000,6000,700,800]
    #[0.96539056457849959, 0.96597061098221193, 0.96539056457849959, 0.96461716937354991]
    #ngram_range = (1, 3), k=2000 parameters= [5000,6000,7000,8000]
    #CV_val_error: [0.96307037896365033, 0.96229698375870076, 0.96384377416860012, 0.96287703016241299]
    #ngram_range = (1, 2), k=1500 parameters= [7000,7200,7500,7700,8000]
    #CV_val_error: [0.9659, 0.9661, 0.9659, 0.9659, 0.9655]
    parameters= [1000,3000,4000,5000,7000,8000,9000,10000]
    best_c = tune_spam(train_data_features,label,parameters)
    model = SVC(C=best_c)
    model.fit(train_data_features,label)
    y_pred = model.predict(test)
    np.savetxt("spam_predict.csv",y_pred,delimiter=",")



