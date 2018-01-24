#############################################################
## ASSIGNMENT 2 CODE SKELETON
## RELEASED: 1/17/2018
## DUE: 1/24/2018
## DESCRIPTION: In this assignment, you will explore the
## text classification problem of identifying complex words.
## We have provided the following skeleton for your code,
## with several helper functions, and all the required
## functions you need to write.
#############################################################

from collections import defaultdict
import gzip
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import re
from sklearn.naive_bayes import GaussianNB
import syllables

#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels
y_prediction = [1, 1, 1, 1, 1, 0, 0, 0, 0]
y_truth = [1, 1, 0, 0, 0, 1, 1, 0, 0]

def get_true_comp(y_pred, y_true):
    val = 0
    for i in range(len(y_pred)):
        if (y_pred[i] and y_true[i]):
            val += 1
    return val

def get_false_comp(y_pred, y_true):
    val = 0
    for i in range(len(y_pred)):
        if (y_pred[i] and not y_true[i]):
            val += 1
    return val


def get_false_simp(y_pred, y_true):
    val = 0
    for i in range(len(y_pred)):
        if (not y_pred[i] and y_true[i]):
            val += 1
    return val

## Calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
    tc = get_true_comp(y_pred, y_true)
    fc = get_false_comp(y_pred, y_true)

    if tc + fc != 0:
        return tc / (tc + fc)
    else:
        return 1
    
## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    tc = get_true_comp(y_pred, y_true)
    fs = get_false_simp(y_pred, y_true)

    if tc + fs != 0:
        return tc / (tc + fs)
    else:
        return 1

## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    p = get_precision(y_pred, y_true)
    r = get_recall(y_pred, y_true)

    if p + r != 0:
        return 2*(p * r) / (p + r)
    else:
        return 

def get_predictions(y_pred, y_true):
    return get_precision(y_pred, y_true), get_recall(y_pred, y_true), get_fscore(y_pred, y_true)

def test_predictions(y_pred, y_true):
    print ("Precision: %0.3f\nRecall: %0.3f\nf-score: %0.3f" 
        %(get_precision(y_pred, y_true), get_recall(y_pred, y_true), get_fscore(y_pred, y_true)))

# test_predictions(y_prediction, y_truth)
#### 2. Complex Word Identification ####

## Loads in the words and labels of one of the datasets
def load_file_upper(data_file):
    words = []
    labels = []   
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0])
                labels.append(int(line_split[1]))
            i += 1
    return words, labels

def load_file(data_file):
    words = []
    labels = []   
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels

### 2.1: A very simple baseline

## Labels every word complex
def all_complex(data_file):
    ## YOUR CODE HERE...
    words, labels = load_file(data_file)
    y_pred = []
    y_true = []
    for i in range(len(words)):
        y_pred.append(1)
        y_true.append(labels[i])
    performance = [get_precision(y_pred, y_true), 
        get_recall(y_pred, y_true), get_fscore(y_pred, y_true)]
    # print(performance)
    return performance

# print(all_complex("complex_words_training.txt"))

### 2.2: Word length thresholding

## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set

def word_length_dicts(words, labels, threshold):
    pred = []
    true = []
    for i in range(len(words)):
        if (len(words[i]) >= threshold):
            pred.append (1)
        else:
            pred.append(0)
        true.append(labels[i])

    return pred, true

def word_length_threshold(training_file, development_file):
    tp = np.zeros(28)
    tr = np.zeros(28)
    tf = np.zeros(28)
    best_thresh = 1
    precisions = []
    recalls = []
    best_f = 0 
    best_r = 0 
    best_p = 0
    t_words, t_labels = load_file(training_file)
    d_words, d_labels = load_file(development_file)

    for threshold in range(2, 30):
        i = threshold - 2
        train_pred, train_true = word_length_dicts(t_words, t_labels, threshold)
        tfs = get_fscore(train_pred, train_true)
        tps = get_precision(train_pred, train_true)
        trs = get_recall(train_pred, train_true)

        precisions.append(tps)
        recalls.append(trs)
        tp[i] = tps 
        tr[i] = trs
        tf[i] = tfs 

        if tfs > best_f:
            best_thresh = threshold
            best_f = tfs 
            best_p = tps 
            best_r = trs

    print("Length Training Performance Stats ")
    print("Best Recall: " + str(best_r))
    print("Best F-Score: " + str(best_f))
    print("Best Precision: " + str(best_p))
    print("Best Length Threshold: " + str(best_thresh))

    dev_pred, dev_true = word_length_dicts(d_words, d_labels, best_thresh)
    dps = get_precision(dev_pred, dev_true)
    dfs = get_fscore(dev_pred, dev_true)
    drs = get_recall(dev_pred, dev_true)

    # plt = matplotlib.pyplot
    plt.plot(recalls, precisions, '-')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.draw()
    plt.savefig("Precision-Recall Length Curve")

    training_performance = [best_p, best_r, best_f]
    development_performance = [dps, drs, dfs]
    # print (training_performance)
    # print (development_performance)
    return training_performance, development_performance

word_length_threshold("complex_words_training.txt", "complex_words_development.txt")
### 2.3: Word frequency thresholding

## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file): 
   counts = defaultdict(int) 
   with gzip.open(ngram_counts_file, 'rt') as f: 
       for line in f:
           token, count = line.strip().split('\t') 
           if token[0].islower(): 
               counts[token] = int(count) 
   return counts

def word_frequ_dicts(words, labels, counts,threshold):
    pred = []
    true = []
    for i in range(len(words)):
        count = counts[words[i]]
        if count == 0:
            word = re.sub(pattern="-", repl="", string = words[i])
            count = counts[word]
        if (count <= threshold):
            pred.append(1)
        else:
            pred.append(0)
        true.append(labels[i])

    return pred, true

# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set
def word_frequency_threshold(training_file, development_file, counts):
    # tp = np.zeros(28)
    # tr = np.zeros(28)
    # tf = np.zeros(28)
    best_thresh = 1
    precisions = []
    recalls = []
    best_f = 0 
    best_r = 0 
    best_p = 0
    t_words, t_labels = load_file_upper(training_file)
    d_words, d_labels = load_file_upper(development_file)

    i = 0
    for threshold in range(0, 60000000, 100000):
        i = threshold
        train_pred, train_true = word_frequ_dicts(t_words, t_labels, counts, threshold)
        tfs = get_fscore(train_pred, train_true)
        tps = get_precision(train_pred, train_true)
        trs = get_recall(train_pred, train_true)

        precisions.append(tps)
        recalls.append(trs)
        # tp[i] = tps 
        # tr[i] = trs
        # tf[i] = tfs 
        if tfs > best_f:
            best_thresh = threshold
            best_f = tfs 
            best_p = tps 
            best_r = trs
        i += 1

    print("Frequency Training Performance Stats ")
    print("Best Recall: " + str(best_r))
    print("Best F-Score: " + str(best_f))
    print("Best Precision: " + str(best_p))
    print("Best Frequency Threshold: " + str(best_thresh))

    dev_pred, dev_true = word_frequ_dicts(d_words, d_labels, counts, best_thresh)
    dps = get_precision(dev_pred, dev_true)
    dfs = get_fscore(dev_pred, dev_true)
    drs = get_recall(dev_pred, dev_true)

    # plt = matplotlib.pyplot
    plt.plot(recalls, precisions, '-')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for Frequency")
    plt.show()

    # TODO IMPLEMENT ERROR ERROR ERROR
    # training_performance = [tprecision, trecall, tfscore]
    # training_performance = [get_per]
    # development_performance = [dprecision, drecall, dfscore]
    # return training_performance, development_performance

### 2.4: Naive Bayes
def norm(vec):
    mean = np.mean(vec)
    sd = np.std(vec)
    for i in range(len(vec)):
        vec[i] = (vec[i] - mean) / sd
    return vec

## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    ## YOUR CODE HERE
    t_words, t_labels = load_file_upper(training_file)
    feat_mat = np.zeros((len(t_words), 2))
    labels_vec = np.zeros(len(t_words))

    for i in range(0,len(t_words)):
        feat_mat[i, 0] = len(t_words[i])
        count = counts[t_words[i]]
        if count == 0:
            fixed = re.sub(pattern = '-', repl="", string = t_words[i])
            count = counts[fixed]
        feat_mat[i, 1] = count
        labels_vec[i] = t_labels[i]

    feat_mat[ :, 0] = norm(feat_mat[ :, 0])
    feat_mat[ :, 1] = norm(feat_mat[ :, 1])

    clf = GaussianNB()
    clf.fit(feat_mat, labels_vec)

    d_words, d_labels = load_file_upper(development_file)
    dev_mat = np.zeros((len(d_words), 2))
    # dev_vec = np.zeros(len(d_words))

    for i in range(0, len(d_words)):
        dev_mat[i, 0] = len(d_words[i])
        count = counts[d_words[i]]
        if count == 0:
            re.sub(pattern="-", repl="", string = d_words[i])
            count = counts[fixed]
        dev_mat[i, 1] = count
        # dev_fec[i] = labels[i]

    dev_mat[ :, 0] = norm(dev_mat[ :, 0])
    dev_mat[ :, 1] = norm(dev_mat[ :, 1])

    train_pred = clf.predict(feat_mat)
    dev_pred = clf.predict(dev_mat)

    print("Naive Bayes Performance Test Statistics")
    test_predictions(train_pred, t_labels)
    print()

    print("Naive Bayes Performance Dev Statistics")
    test_predictions(dev_pred, d_labels)
    print()

   
    # training_performance = [tprecision, trecall, tfscore]
    training_performance = [get_predictions(train_pred, t_labels)]
    # development_performance = [dprecision, drecall, dfscore]
    development_performance = [get_predictions(dev_pred, d_labels)]
    return training_performance, development_performance

### 2.5: Logistic Regression

## Trains a Naive Bayes classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):
    ## YOUR CODE HERE    
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

def show_both():
    plt.plot(recalls, precisions, '-')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for Frequency")
    plt.show()
### 2.7: Build your own classifier

## Trains a classifier of your choosing, predicts labels for the test dataset
## and writes the predicted labels to the text file 'test_labels.txt',
## with ONE LABEL PER LINE

def sentence_length(f):
    sen_len = dict()
    i = 0
    for line in f:
        if i > 0:
            line_split = line[:-1].split("\t")
            word = line_split[0]
            sen_len[word] = len(line_split[3].split(r"\s"))
        i += 1
    return sen_len

def syllable_length(f):
    syl_len = {}
    for line in f:
        if i > 0:
            line_split = line[:-1].split("\t")
            word = line_split[0]
            syl_len[word] = syllables.count_syllables(word)


def classifier(file_name):
    file = open(file_name, 'rt', encoding = "utf8")

    sen_len = sentence_length(file)
    syl_len = syllabe_length(file)

    file.close()


if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    train_data = load_file(training_file)
    
    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)
    print ()
    # word_frequency_threshold("complex_words_training.txt", "complex_words_development.txt", counts)
    naive_bayes("complex_words_training.txt", "complex_words_development.txt", counts)


