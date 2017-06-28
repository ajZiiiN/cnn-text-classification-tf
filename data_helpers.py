import numpy as np
import re
import itertools
from collections import Counter
import pandas as pd
import numpy as np


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def loadData(file):
    tpg_data = pd.read_csv(file, sep='|')
    print(tpg_data.dtypes)

    x_text = np.array(tpg_data.text, dtype = pd.Series)

    y_text = np.array(tpg_data.label , dtype = pd.Series)
    y_text_onehot = []
    countBad = 0
    totalCount = 0
    for i, j in enumerate(y_text):
        totalCount = totalCount + 1
        # if j == 1:
        #     y_text_onehot.append([0,1])
        #     countBad = countBad + 1
        #print(str(x_text[i:i+1][0]))
        if isBad(str(x_text[i:i+1][0])):
            y_text_onehot.append([0,1])
            countBad = countBad + 1
        else:
            y_text_onehot.append([1,0])
    print("bad: {0}, total: {1}, ratio: {2}".format(countBad, totalCount, countBad/totalCount))
    return (x_text, np.array(y_text_onehot))

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def loadBad(filepath):
    badWords = set()
    with open(filepath, 'r') as fb:
        for line in fb:
            word = line.strip()
            if len(word) > 1:
                badWords.add(word)
    return badWords

def checkIfBad(msg, badWords):
    isbad = False
    for word in badWords:
        if word in str(msg):
            isbad = True
            break
    return isbad

def isBad(msg):
    badWords = loadBad("../../data/abusive_content/bad_words.txt")
    return checkIfBad(msg.lower(), badWords)

'''
"Hey motherfucker. I fuck ur mom. And I fuck ur sister too. Sisterfuckers. O machikna randiko chora. Khelchas ain't. Oea randiko chora kt lai j sukai vanchas Teri. Tero ama lai mailay dui khep chekako. Tero AMA Ko kalo puti chiya. Randiko chora valu tero didi. Sabailay chikako. Hamilay tero didi lai 12 jana lay chikako. Randiko chora mero kalo lado chha. Khelchas muji tero thau van muji tero gau ma kutchu ta randiko chora l. Ta randi Ko ban tero AMA Ko haneko. Tero AMA Ko chikako. Ter. Tero didi Ko ra sano baini Ko puti ma falam chirako. Randiko chora. Oea tero address van randiko chora darauchas. Daraysis. Muji tero AMA lai kati chiknu. Boksi Ko chora. Khelchas vani tero exect address van. Katikay baun muji darauxas randiko ban. Taro AMA randi valu. Boksi Ko ban darauxas bahun muji. Address van talai ra tero bau lai kutchu. Daraixa bahun muji bau chora lai kutarY. AMA ra baini lai chikchu. Fuck ur mom saroj muji. Suck my dick bitches. Motherfucker. Darako puri. Randi Ko baan. Valu Ko chora. Buini chikuwa. Didi lai besyA laya send gar. Daraixas bahun randiko chora. Randikoban address send gar vaneko. Muji tero santan lai siduxu"

'''