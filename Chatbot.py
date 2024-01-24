# -*- coding: utf-8 -*-

Import nltk
import random
import string
import warnings
warnings.filterwarnings('ignore')

file= open('C:\\Users\\User\\Downloads\\Ai.txt', 'r', errors='ignore')
rawdata = file.read()
rawdata= rawdata.lower()

sent_token = nltk.sent_tokenize(rawdata) #convert into list of sentences
word_token = nltk.word_tokenize(rawdata) #convert into list of words

sentTokens = sent_tokens[:3]
#print(sentTokens)
wordTokens= word_tokens[:3]
#print(wordTokens)

#preprocessing 
sentences=nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [sentences.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#Greetings
GREETING_INPUTS = ("hello", "hi", "greetings",  "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! you are talking to me"]

def greeting(sentence):
    
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        
#Vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    chatbot_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf == 0):
        chatbot_response = chatbot_response + "I am sorry! I don't understand "
        return chatbot_response
    
    else:
        chatbot_response=chatbot_response+sent_tokens[idx]
        return chatbot_response
    

if __name__ == "__main__":
    flag = True
    print("Hello, there my name is Siri. I will answer your queries. If you want to quit, type Bye!")
    while(flag==True):
        user_response = input()
        user_response = user_response.lower()
        if(user_response!='bye'):
            if user_response == 'thanks' or user_response == 'thank you':
                flag = False
                print("Siri: You're welcome!")
            else:
                if(greeting(user_response)!=None):
                        print("Siri:"+greeting(user_response))
                else:
                    print("Siri:", end='')
                    print(response(user_response))
                    sent_tokens.remove(user_response)
        else:
            flag = False
            print("Siri: Bye! Have a great time!" )
