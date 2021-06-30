from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from nltk import ngrams
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
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
#pass the test dataset here
test_data=pd.read_csv(r'data2.csv',engine='python')
print("given size of the data",len(test_data))
#dropping the null values
test_data=test_data.drop_duplicates()
print(" size of data after removing repitations",len(test_data))
#print(test_data['TWEETS'].value_counts())
#DATA PRECPROCESSING STARTS FROM HERRE

wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def normalizer(tweet):
    aplhabets = re.sub("[^a-zA-Z]", " ",tweet)
    tokens = nltk.word_tokenize(aplhabets)[2:]
    lower_case = [l.lower() for l in tokens]
    processed_words = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in processed_words]
    return lemmas

#create a coloum in the dataset to get only normalized words
print("start of pre_processing")
test_data["Review Text"]=test_data["Review Text"].astype(str)
test_data["normalized_Tweets"]=np.nan
test_data["normalized_Tweets"]=test_data["Review Text"].apply(lambda x: normalizer(x))
#now we are converting to ngrams to get the meanining

test_data["processed_tweets"]=np.nan
test_data["sentiment"]=np.nan
def sentiment(tweet):
    return textblob.TextBlob(tweet).sentiment.polarity
def feauture_labelling(score):
    if (score >= 1):
        return 5

    elif (score <= -1):
        return 1

    elif ((score > -0.5) and (score < 0.5)):
        return 3

    elif ((score >= 0.5) and (score < 1)):
        return 4

    elif ((score <= -0.5) and (score > -1)):
        return 2
#note 1=en,2=n,3=nue,4=p,5=exp


test_data["processed_tweets"]=test_data["normalized_Tweets"].apply(lambda x: " ".join(list(x)))
#print(test_data["processed_tweets"].head())

test_data["sentiment"]=test_data["processed_tweets"].apply(lambda x: sentiment(x))
#print(test_data["sentiment"].value_counts())
test_data["Target_label"]=np.nan
test_data["Target_label"]=test_data["sentiment"].apply(lambda x: feauture_labelling(x))
print("displaying the count of data based on caluculated rating")
print(test_data["Target_label"].value_counts())
test_data.to_csv('..\preprocessed_reviews2.csv')

from sklearn.model_selection import train_test_split

final_df=test_data[['sentiment','Target_label']]
print(test_data['sentiment'].value_counts())
print(test_data['Target_label'].value_counts())
train , test = train_test_split(final_df, test_size = 0.1)
x_train = train.drop('Target_label', axis=1)

y_train = train['Target_label']

x_test = test.drop('Target_label', axis = 1)
y_test = test['Target_label']

from sklearn import svm

clf1 =  RandomForestClassifier(n_estimators=3000,max_depth=1)
clf2 = XGBClassifier()

#coding testing and saving with svc    
#Train the model using the training sets
clf1.fit(x_train, y_train)
#Predict the response for test dataset
y_pred1 = clf1.predict(x_test)
#print(y_pred)
#saving predictions
df_test=pd.DataFrame()
df_test = x_test
df_test['predictions']=list(y_pred1)
from sklearn.metrics import f1_score
print("ramdom forest classcification model accuracy(in %):",metrics.accuracy_score(y_test, y_pred1) * 100)
df_test['predictions'].value_counts().sort_index().plot(kind='bar', figsize=(20, 15),
                                                        title='random forest distrubution in test data',
                                                        color='cyan')

df_test.to_csv(r'rating_predictions_svc.csv')
plt.xlabel("PREDICTED CLASSES")
plt.ylabel('COUNT')
plt.show()
   
#coding ,testing and saving with xgboost
final_df=test_data[['sentiment','Target_label']]
print(test_data['sentiment'].value_counts())
print(test_data['Target_label'].value_counts())
train , test = train_test_split(final_df, test_size = 0.3)
x_train = train.drop('Target_label', axis=1)

y_train = train['Target_label']

x_test = test.drop('Target_label', axis = 1)
y_test = test['Target_label']



#Train the model using the training sets
clf2.fit(x_train, y_train)
#Predict the response for test dataset
y_pred2 = clf2.predict(x_test)
#print(y_pred)
#saving predictions
df_test2=pd.DataFrame()
df_test2 = x_test
df_test2['predictions']=list(y_pred2)
print("xgboost classcification model accuracy(in %):", f1_score(y_test, y_pred2,average='micro') * 100)
df_test['predictions'].value_counts().sort_index().plot(kind='bar', figsize=(20, 15),
                                                        title='xgboost distrubution in test data',
                                                        color='cyan')

df_test.to_csv(r'rating_predictions_xgboost.csv')
plt.xlabel("PREDICTED CLASSES")
plt.ylabel('COUNT')
plt.show()





import numpy as np
sample_Input=input("enter a sentence")
while sample_Input!='q':
 #sample_Input=input("enter a sentence")
 n_sample=normalizer(sample_Input)
 #print(n_sample)
 processed_string=' '.join(n_sample)
 score=sentiment(processed_string)
 print(np.asarray([score]))
 ypd=clf2.predict(np.asarray([score]))
 print(ypd)
 sample_Input = input("enter a sentence,q to exit")

