
from flask import Flask, jsonify,render_template,request,redirect,url_for

from flask_login import login_required
import urllib.request

import numpy as np
import pickle
from keras.models import load_model
import time
from keras.preprocessing.sequence import pad_sequences
import re
from nltk.tokenize import word_tokenize
import pandas as pd
from keras.preprocessing.text import Tokenizer
from tensorflow.compat.v1 import Graph, Session
import tensorflow as tf
import os

#TO DISTRIBUTE GPU SPACE BETWEEN TWO SESSIONS

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True


tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))


SENTIMENT_THRESHOLDS = (0.4, 0.7)

SEQUENCE_LENGTH = 200
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"

def decode_sentiment(score, include_neutral=True):
     if include_neutral:        
          label = NEUTRAL
          if score <= SENTIMENT_THRESHOLDS[0]:
               label = NEGATIVE
          elif score >= SENTIMENT_THRESHOLDS[1]:
               label = POSITIVE

          return label
     else:
          return NEGATIVE if score < 0.5 else POSITIVE

def predict_sentiment(text, include_neutral=True):
     start_at = time.time()
     # Tokenize text
     x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
     # Predict
     graph1 = Graph()
     with graph1.as_default():
          session1 = Session(config=config)
          with session1.as_default():
               # load model
               model1 = load_model("model.h5")
               score = model1.predict([x_test])[0]               
     # Decode sentiment
     label = decode_sentiment(score, include_neutral=include_neutral)
     """print("_____________+++________________")
     print (label)
     print("_____________+++________________")"""
     return label

#FOR INTENT CLASSIFICATION
def cleaning(sentences):
     words = []
     for s in sentences:
          #removing irrelevant symbols from sentence and replacing them spaces
          clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
          w = word_tokenize(clean)
    #stemming
          words.append([i.lower() for i in w])
    
     return words


def max_length(word_max_length):
     return(len(max(word_max_length, key = len)))

#loading dataset for intents
def load_dataset(filename):
     df = pd.read_csv(filename, encoding = "latin1", names = ["Sentence", "Intent"])
     sentences = list(df["Sentence"])
     intent = df["Intent"]
     unique_intent = list(set(intent))
  
  
     return (unique_intent,sentences)

unique_intent,sentences= load_dataset("Dataset_intent.csv")

cleaned_words = cleaning(sentences)
max_length = max_length(cleaned_words)



def create_tokenizer(words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
     token = Tokenizer(filters = filters)
     token.fit_on_texts(words)
     return token

word_tokenizer = create_tokenizer(cleaned_words)
output_tokenizer = create_tokenizer(unique_intent, filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')



def encoding_doc(token, words):
     return(token.texts_to_sequences(words))

encoded_doc = encoding_doc(word_tokenizer, cleaned_words)



def padding_doc(encoded_doc, max_length):
     return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))

#predicting intent
def predict_intent(text):
     clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
     test_word = word_tokenize(clean)
     test_word = [w.lower() for w in test_word]
     test_ls = word_tokenizer.texts_to_sequences(test_word)
     print(test_word)
     #Check for unknown words
     if [] in test_ls:
          test_ls = list(filter(None, test_ls))
          test_ls = np.array(test_ls).reshape(1, len(test_ls))
 
     x = padding_doc(test_ls, max_length)
     graph2 = Graph()
     with graph2.as_default():
          session2 = Session(config=config)
          with session2.as_default():
          # load model
               model2 = load_model("model_intent.h5")
               pred = model2.predict_proba(x)

     return pred

def get_final_output_sentiment(pred, classes):
     predictions = pred[0]
     classes = np.array(classes)
     ids = np.argsort(-predictions)
     classes = classes[ids]
     predictions =-np.sort(-predictions)
     pmax=predictions.max()
     for i in range(pred.shape[1]):
          if predictions[i]==pmax:
               return(classes[i])

          
#initializing app
app = Flask(__name__)

# Route for handling the login page logic
@app.route('/', methods=['GET', 'POST'])

def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('home'))
    return render_template('login.html', error=error)
    
@app.route("/chatbot")
def home():    
    return render_template("home.html") 
@app.route("/get")
def get_bot_response():    
     userText = request.args.get('msg')   
     
     
     c=predict_sentiment(userText)
     pred=predict_intent(userText)
     intent=get_final_output_sentiment(pred,unique_intent)
     
     if c=='POSITIVE' or c=='NEUTRAL':
          
          
          if intent== 'commonQ.bot':
               #enter custom story for intent
               return str("commonQ.bot")

          elif intent=='commonQ.assist':
               #enter custom story for intent
               return str("commonQ.assist")

          elif intent=='faq.biz_new':
               #enter custom story for intent
               return str(" faq.biz_new")

          elif intent=='contact.contact':
               #enter custom story for intent
               return str(" contact.contact")

          elif intent=='faq.borrow_use':
               #enter custom story for intent
               return str("faq.borrow_use")

          elif intent=='faq.aadhaar_missing':
               #enter custom story for intent
               return str(" faq.aadhaar_missing")

          elif intent=='faq.approval_time':
               #enter custom story for intent
               return str(" faq.approval_time")

          elif intent=='faq.address_proof':
               #enter custom story for intent
               return str("faq.address_proof")

          elif intent=='faq.bad_service':
               #enter custom story for intent
               return str(" faq.bad_service")

          elif intent=='commonQ.just_details':
               #enter custom story for intent
               return str("commonQ.just_details")

          elif intent=='commonQ.not_giving':
               #enter custom story for intent
               return str(" commonQ.not_giving")
          
          elif intent=='commonQ.wait':
               #enter custom story for intent
               return str("commonQ.wait")
          
          elif intent=='faq.banking_option_missing':
               #enter custom story for intent
               return str(" faq.banking_option_missing")
          
          elif intent=='commonQ.query':
               #enter custom story for intent
               return str("commonQ.query")
          
          elif intent=='faq.apply_register':
               #enter custom story for intent
               return str(" faq.apply_register")
          
          elif intent=='commonQ.name':
               #enter custom story for intent
               return str(" commonQ.name")
          
          elif intent=='faq.biz_simpler':
               #enter custom story for intent
               return str("faq.biz_simpler")
          
          elif intent=='commonQ.how':
               #enter custom story for intent
               return str(" commonQ.how")

          elif intent=='faq.biz_category_missing':
               #enter custom story for intent
               return str(" faq.biz_category_missing")
     
          elif intent=='faq.borrow_limit':
               #enter custom story for intent
               return str(" faq.borrow_limit")
     
         
          else:
               return str("i cant understand what you want to say") 
     elif c=="NEGATIVE":
          if intent== 'commonQ.bot':
               #enter custom story for intent
               return str("commonQ.bot")

          elif intent=='commonQ.assist':
               #enter custom story for intent
               return str("commonQ.assist")

          elif intent=='faq.biz_new':
               #enter custom story for intent
               return str(" faq.biz_new")

          elif intent=='contact.contact':
               #enter custom story for intent
               return str(" contact.contact")

          elif intent=='faq.borrow_use':
               #enter custom story for intent
               return str("faq.borrow_use")

          elif intent=='faq.aadhaar_missing':
               #enter custom story for intent
               return str(" faq.aadhaar_missing")

          elif intent=='faq.approval_time':
               #enter custom story for intent
               return str(" faq.approval_time")

          elif intent=='faq.address_proof':
               #enter custom story for intent
               return str("faq.address_proof")

          elif intent=='faq.bad_service':
               #enter custom story for intent
               return str(" faq.bad_service")

          elif intent=='commonQ.just_details':
               #enter custom story for intent
               return str("commonQ.just_details")

          elif intent=='commonQ.not_giving':
               #enter custom story for intent
               return str(" commonQ.not_giving")
          
          elif intent=='commonQ.wait':
               #enter custom story for intent
               return str("commonQ.wait")
          
          elif intent=='faq.banking_option_missing':
               #enter custom story for intent
               return str(" faq.banking_option_missing")
          
          elif intent=='commonQ.query':
               #enter custom story for intent
               return str("commonQ.query")
          
          elif intent=='faq.apply_register':
               #enter custom story for intent
               return str(" faq.apply_register")
          
          elif intent=='commonQ.name':
               #enter custom story for intent
               return str(" commonQ.name")
          
          elif intent=='faq.biz_simpler':
               #enter custom story for intent
               return str("faq.biz_simpler")
          
          elif intent=='commonQ.how':
               #enter custom story for intent
               return str(" commonQ.how")

          elif intent=='faq.biz_category_missing':
               #enter custom story for intent
               return str(" faq.biz_category_missing")
     
          elif intent=='faq.borrow_limit':
               #enter custom story for intent
               return str(" faq.borrow_limit")
          return str("plz enter your email so that we can contact you")
if __name__ == '__main__':
     
     app.run(port=5000,debug=True)


