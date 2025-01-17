# import random
# import json
# # import sys
# # sys.stdout = open('training_output.log', 'w', encoding='utf-8')

# import pickle
# import numpy as np
# import tensorflow as tf

# import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# from nltk.stem import WordNetLemmatizer

# lemmatizer = WordNetLemmatizer()

# intents = json.loads(open('audispot.json').read())

# words = []
# classes = []
# documents = []
# ignoreLetters = ['?', '!', '.', ',']

# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         wordList = nltk.word_tokenize(pattern)
#         words.extend(wordList)
#         documents.append((wordList, intent['tag']))
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
# words = sorted(set(words))

# classes = sorted(set(classes))

# pickle.dump(words, open('words.pkl', 'wb'))
# pickle.dump(classes, open('classes.pkl', 'wb'))

# training = []
# outputEmpty = [0] * len(classes)

# for document in documents:
#     bag = []
#     wordPatterns = document[0]
#     wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
#     for word in words:
#         bag.append(1) if word in wordPatterns else bag.append(0)

#     outputRow = list(outputEmpty)
#     outputRow[classes.index(document[1])] = 1
#     training.append(bag + outputRow)

# random.shuffle(training)
# training = np.array(training)

# trainX = training[:, :len(words)]
# trainY = training[:, len(words):]

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(64, activation = 'relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)
# model.save('chatbot_model.keras', hist)
# print('Done')

import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
import os

# Set TensorFlow log level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Redirect output to log file
import sys
sys.stdout = open('training_output.log', 'w', encoding='utf-8')

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load the intents JSON file with proper encoding
intents = json.loads(open('audispot.json', 'r', encoding='utf-8').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.keras')

print('Done')
