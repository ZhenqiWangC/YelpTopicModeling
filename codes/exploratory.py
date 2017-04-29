from nltk.tokenize import RegexpTokenizer

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora
from gensim import models
import nltk
from nltk.corpus import wordnet as wn


if __name__ == '__main__':
    state = 'IL'
    review = pd.read_csv("./data/" + str(state) + "review.csv")
    raw = review.text.tolist()
    tokenizer = RegexpTokenizer(r'\w+')
    stop = set(stopwords.words('english'))
    food = wn.synsets('food')
    p_stemmer = PorterStemmer()
    text_array = []
    is_noun = lambda pos: pos[:2] == 'NN'
    for i in range(len(raw)):
        text = raw[i].lower()
        text = text.replace('\r\n', ' ')
        text = re.sub("[^a-z0-9]", " ", text)
        # Tokenization segments a document into its atomic elements.
        words = text.split()
        # Stop words
        # Certain parts of English speech, like (for, or) or the word the are meaningless to a topic model.
        # These terms are called stop words and need to be removed from our token list.
        words = [j for j in words if j not in stop]
        tokenized = nltk.word_tokenize(text)
        words = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]

        # Stemming words is another common NLP technique to reduce topically similar words to their root.
        # stemming reduces those terms to stem. This is important for topic modeling, which would otherwise view those terms as separate entities and reduce their importance in the model.
        #words = [p_stemmer.stem(s) for s in words]
        text_array.append(words)

    dictionary = corpora.Dictionary(text_array)
    corpus = [dictionary.doc2bow(text) for text in text_array]
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=20)
    print(ldamodel.print_topics(num_topics=10, num_words=3))
    #vectorizer = TfidfVectorizer(ngram_range=(1,2), analyzer="word", lowercase=False)
    #vectorizer = vectorizer.fit(text_array)
    #train_data_features = vectorizer.transform(text_array)








