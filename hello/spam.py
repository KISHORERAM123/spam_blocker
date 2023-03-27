import os
import io
import numpy
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

#Function to read the messages leaving the header from each of the files and adding it
#to the list for classification
def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message

#Function to append the message and their particular classification
def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

#Main to call functions
data = DataFrame({'message': [], 'class': []})
data = pd.concat([data, dataFrameFromDirectory('G:\police hackathon\spam-classifier-master\spam-classifier-master\emails\spam', 'spam')])
data = pd.concat([data, dataFrameFromDirectory('G:\police hackathon\spam-classifier-master\spam-classifier-master\emails\ham', 'ham')])

#Reading Data Frame
data.head()

#Training data using MultinomialNB classifier
# vectorizer = CountVectorizer(stop_words='english', min_df=1)
# counts = vectorizer.fit_transform(data['message'].values)
# remove any empty or null values
data = data.dropna()
data = data[data.message != '']

# train the vectorizer
vectorizer = CountVectorizer(stop_words='english', min_df=1)
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)

#Predicting the Spam Emails
examples = ['Free Viagra now', "Hi Bob, how about a game of golf tomorrow","hello sanjay"]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)
