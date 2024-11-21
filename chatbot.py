# import random 
# import json
# import pickle
# import numpy as np
# import tensorflow 
# import nltk
# from nltk.stem import WordNetLemmatizer

# from tensorflow.keras.models import load_model 

# lemmatizer = WordNetLemmatizer()

# intents = json.loads(open('audispot.json').read())

# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))
# model = load_model('chatbot_model.keras')

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
#     return sentence_words

# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for w in sentence_words:
#         for i, word in enumerate(words):
#             if word == w:
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

#     results.sort(key=lambda x: x[1], reverse = True)
#     return_list = []
#     for r in results:
#         return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

#     return return_list

# def get_response(intents_list, audispot_json):
#     tag = intents_list[0]['intent']
#     list_of_intents = audispot_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             result = random.choice(i['responses'])
#             break
#     return result

# print("Bot is running")

# while True:
#     message = input("")
#     ints = predict_class(message)
#     res = get_response(ints, intents)
#     print(res)


import random 
import json
import pickle
import numpy as np
import tensorflow
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model 
import sys

# Set default encoding to UTF-8 for print statements
sys.stdout.reconfigure(encoding='utf-8')

lemmatizer = WordNetLemmatizer()

# Load the intents file and pre-trained model
intents = json.loads(open('audispot.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

# Function to clean up the sentence by tokenizing and lemmatizing
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Function to create a bag of words from the sentence
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to predict the class of the sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list

# Function to get the response based on the predicted intent
def get_response(intents_list, audispot_json):
    tag = intents_list[0]['intent']
    list_of_intents = audispot_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Print a message to indicate that the bot is running
print("Bot is running")

# Open a log file to store output
with open('chatbot_output.txt', 'a', encoding='utf-8') as log_file:
    while True:
        message = input("")
        ints = predict_class(message)
        res = get_response(ints, intents)

        # Write response to log file
        log_file.write(f"User: {message}\nBot: {res}\n\n")
        
        # Print response to console with UTF-8 encoding
        print(res)
