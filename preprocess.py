# IFT 3335 - TP 2
# Code de Philippe Schoeb et Nathan Bussière
# 19 avril 2024

import numpy as np
import csv
from tqdm import tqdm
from sklearn.utils import shuffle
from nltk import word_tokenize
import nltk
# You can put the second two lines as comments if punkt and stopwords are already downloaded
nltk.download('punkt')
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class DataReader:

    def __init__(self, file_path, sub_task=None):
        self.file_path = file_path
        self.sub_task = sub_task

    def get_labelled_data(self):
        data = []
        labels = []
        with open(self.file_path, encoding='utf8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for i, line in enumerate(tqdm(reader, 'Reading Data')):
                if i == 0:
                    continue
                label = self.str_to_label(line[-3:])
                if self.sub_task:
                    self.filter_subtask(data, labels, line[1], label)
                else:
                    labels.append(label)
                    data.append(line[1])
        return data, labels


    def shuffle(self, data, labels, state=None):
        if not state:
            if not self.sub_task or self.sub_task == 'A':
                off_data, off_labels = [], []
                not_data, not_labels = [], []
                for i, tweet in tqdm(enumerate(data), 'Shuffling Data'):
                    if labels[i] == 0:
                        not_data.append(tweet)
                        not_labels.append(labels[i])
                    else:
                        off_data.append(tweet)
                        off_labels.append(labels[i])
                shuffled_data = off_data[:len(off_data) // 4] + not_data[:len(not_data) // 4] + off_data[
                                len(off_data) // 4:len(off_data) // 2] + not_data[len(not_data) // 4:len(
                                not_data) // 2] + off_data[len(off_data) // 2:3 * len(off_data) // 4] + not_data[
                                len(not_data) // 2:3 * len(not_data) // 4] + off_data[3 * len(
                                off_data) // 4:] + not_data[3 * len(not_data) // 4:]
                shuffled_labels = off_labels[:len(off_labels) // 4] + not_labels[:len(not_labels) // 4] + off_labels[
                                  len(off_labels) // 4:len(off_labels) // 2] + not_labels[len(not_labels) // 4:len(
                                  not_labels) // 2] + off_labels[len(off_labels) // 2:3 * len(off_labels) // 4] + \
                                  not_labels[len(not_labels) // 2:3 * len(not_labels) // 4] + off_labels[3 * len(
                                  off_labels) // 4:] + not_labels[3 * len(not_labels) // 4:]
                return shuffled_data, shuffled_labels
            elif self.sub_task in ['B', 'C']:
                pass
        elif state == 'random':
            shuffled_data, shuffled_labels = shuffle(data, labels, random_state=16)
            return shuffled_data, shuffled_labels
        else:
            return data, labels

    def upsample(self, data, labels, label=0):
        new_data = []
        new_labels = []
        count = 0
        for i, tweet in enumerate(data):
            new_labels.append(labels[i])
            new_data.append(data[i])
            if labels[i] == label:
                new_labels.append(labels[i])
                new_data.append(data[i])
                count += 1
        return new_data, new_labels

    def str_to_label(self, all_labels):
        label = 0
        if all_labels[0] == 'OFF':
            label = 1
        return label



# Labels
# 0 - Not offensive
# 1 - Offensive untargeted
# 2 - Offensive targeted indiviualds
# 3 - Offensive targeted groups
# 4 - Offensive targeted others


# Function for lemmanizer
def get_pos(word):
    tag = pos_tag([word])[0][1]
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess_data(data, minuscule=True, steming='default', vecto='count', stopword='default'):
    # Tokenize
    if minuscule:
        for i, tweet in tqdm(enumerate(data), 'Tokenization'):
            data[i] = word_tokenize(tweet.lower())
    else:
        for i, tweet in tqdm(enumerate(data), 'Tokenization'):
            data[i] = word_tokenize(tweet)

    # Stopwords
    if type(stopword) == type(""):
        stop = set(stopwords.words("english"))
        noise = ['user']
        for i, tweet in tqdm(enumerate(data), 'Stopwords Removal (default)'):
            data[i] = [w for w in tweet if w not in stop and not re.match(r"[^a-zA-Z\d\s]+", w) and w not in noise]
    elif len(stopword) > 0:
        stop = set(stopword)
        for i, tweet in tqdm(enumerate(data), 'Stopwords Removal'):
            ###############################################  I don't fully understand re.match yet. to be looked at.
            data[i] = [w for w in tweet if w not in stop and not re.match(r"[^a-zA-Z\d\s]+", w)]

    # Steming or lemmanizer
    if steming == 'default':
        stemmer = PorterStemmer()
        for i, tweet in tqdm(enumerate(data), 'Stemming (PorterStemmer)'):
            for j, word in enumerate(tweet):
                data[i][j] = stemmer.stem(word)
    elif steming == 'lemmatizeWNL':
        wnl = WordNetLemmatizer()
        for i, tweet in tqdm(enumerate(data), 'Lemmatization'):
            for j, word in enumerate(tweet):
                data[i][j] = wnl.lemmatize(word, pos=get_pos(word))

    # Vectorisation
    untokenized_data = [' '.join(tweet) for tweet in data]
    if vecto == 'binary_count':
        vectorizer = CountVectorizer(binary=True)
        vectors = vectorizer.fit_transform(untokenized_data)
    elif vecto == 'count':
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform(untokenized_data)
    elif vecto == 'binary_tfidf':
        vectorizer = TfidfVectorizer(binary=True)
        vectors = vectorizer.fit_transform(untokenized_data)
    elif vecto == 'tfidf':
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(untokenized_data)

    return vectors