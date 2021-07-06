from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from nltk import ngrams
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from  sklearn import metrics

import numpy as np
import textblob
import re, nltk
from nltk.stem import WordNetLemmatizer
import nltk
import collections
from nltk.corpus import stopwords
#directly downloads packages for required things
nltk.download('punkt')
nltk.download('popular')
from nltk.corpus import stopwords
import  pandas as pd
import string
#importing the data
from collections import Counter
from sklearn.model_selection import train_test_split

def rating_prediction(review):
    test_data=pd.read_csv("data2.csv",low_memory=False)
    test_data=test_data.drop_duplicates()
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    def normalizer(tweet):
        aplhabets = re.sub("[^a-zA-Z]", " ",tweet)
        tokens = nltk.word_tokenize(aplhabets)
        lower_case = [l.lower() for l in tokens]
        processed_words = list(filter(lambda l: l not in stop_words, lower_case))
        lemmas = [wordnet_lemmatizer.lemmatize(t) for t in processed_words]
        return lemmas
    test_data["Review Text"]=test_data["Review Text"].astype(str)
    test_data["normalized_Tweets"]=np.nan
    test_data["normalized_Tweets"]=test_data["Review Text"].apply(lambda x: normalizer(x))
    test_data["processed_tweets"]=np.nan
    test_data["sentiment"]=np.nan
    def sentiment(tweet):
        return textblob.TextBlob(tweet).sentiment.polarity
    def feauture_labelling(score):
        if score>=-1 and score<-0.5:
            return 1
        elif score>=-0.5 and score<-0.1:
            return 2
        elif score>=-0.1 and score<0.2:
            return 3
        elif score>=0.2 and score<0.6:
            return 4
        else:
            return 5
    test_data["processed_tweets"]=test_data["normalized_Tweets"].apply(lambda x: " ".join(list(x)))
    test_data["sentiment"]=test_data["processed_tweets"].apply(lambda x: sentiment(x))
    test_data["Target_label"]=np.nan
    test_data["Target_label"]=test_data["sentiment"].apply(lambda x: feauture_labelling(x))
    test_data.to_csv('.\preprocessed_reviews2.csv')
    final_df=test_data[['sentiment','Target_label']]
    train , test = train_test_split(final_df, test_size = 0.3)
    x_train = train.drop('Target_label', axis=1)
    y_train = train['Target_label']
    
    clf2 =  XGBClassifier(n_estimators=5000,max_depth=1)
    clf2.fit(x_train, y_train)
    n_sample=normalizer(review)
    processed_string=' '.join(n_sample)
    if len(review)!= 0:
        if len(review) == 1:
            return "Enter the proper input"
        elif "not" in review:
            score=sentiment(processed_string)*-0.5
        else:
            score=sentiment(processed_string)
    else:
        return "Your input is empty, please Enter the input"
    ypd=clf2.predict(np.asarray([score]))

    x_test=np.array(score)
    x_test=x_test.reshape((1,-1))
    return clf2.predict(x_test)


    
    




