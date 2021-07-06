from flask import Flask, render_template, request
import review as r

app = Flask(__name__)
@app.route('/',methods=["GET","POST"])
def review():
    if request.method == 'POST':
        review = request.form["rev"]
        rating_pred=r.rating_prediction(review)
        return render_template('index.html',my_rating=rating_pred)
    return render_template('index.html')
    
"""
@app.route('/predict',methods=['POST'])

def lemmatizer(t1):
    lemma=WordNetLemmatizer()
    return lemma.lemmatize(t1)


def normalizer(tweet):
    aplhabets = re.sub("[^a-zA-Z]", " ",tweet)
    tokens = nltk.word_tokenize(aplhabets)[2:]
    lower_case = [l.lower() for l in tokens]
    processed_words = list(filter(lambda l: l not in stopwords, lower_case))
    lemmas = [lemmatizer(t) for t in processed_words]
    return lemmas

def sentiment(tweet):
    return textblob.TextBlob(tweet).sentiment.polarity
def predict(review):
 
    if request.method == 'POST':
 
        review = request.form['rev']
        data =[[str(review)]]
        l=normalizer(data)
       
        
        l2=l.apply(lambda x: " ".join(list(x)))
        s=sentiment(l2)

 
        lr = pickle.load(open('review.pkl', 'rb'))
        predict= lr.predict(s)
 
    return render_template('predict.html', prediction=predict)
 
 
 
 

 
 """
if __name__ == '__main__':
    app.run()