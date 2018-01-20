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

#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels


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

    return tc / (tc + fc)
    
## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    tc = get_true_comp(y_pred, y_true)
    fs = get_false_simp(y_pred, y_true)

    return tc / (tc + fs)

## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    p = get_precision(y_pred, y_true)
    r = get_recall(y_pred, y_true)
    return 2(p * r) / (p + r)

def test_predictions(y_pred, y_true):
    print ("Precision: %i\nRecall: %i\nf-score: %i\n" 
        %(get_precision(y_pred, y_true), get_recall(y_pred, y_true), get_fscore(y_pred, y_true)))

#### 2. Complex Word Identification ####

## Loads in the words and labels of one of the datasets
def load_file(data_file):
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

### 2.1: A very simple baseline

## Labels every word complex
def all_complex(data_file):
    ## YOUR CODE HERE...
    words, labels = load_file(data_file)
    y_pred = {}
    y_true = {}
    for i in range(len(words)):
        y_pred[words[i]] = 1
        y_true[words[i]] = labels[i]
    performance = [get_precision(y_pred, y_true), 
        get_recall(y_pred, y_true), get_fscore(y_pred, y_true)]
    return performance


### 2.2: Word length thresholding

## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file):
    ## YOUR CODE HERE
    threshold = 9

    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

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

# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set
def word_frequency_threshold(training_file, development_file, counts):
    ## YOUR CODE HERE
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

### 2.4: Naive Bayes
        
## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    ## YOUR CODE HERE
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

### 2.5: Logistic Regression

## Trains a Naive Bayes classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):
    ## YOUR CODE HERE    
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

### 2.7: Build your own classifier

## Trains a classifier of your choosing, predicts labels for the test dataset
## and writes the predicted labels to the text file 'test_labels.txt',
## with ONE LABEL PER LINE

def hello():
    print ("Hello World")

# def load_file(data_file):
#     print ("INSIDE METHOD?")
#     file = open(data_file)
#     print (file)
#     results = {}
#     skip = True
#     for line in file:
#         if skip:
#             skip = False 
#             continue
#         words = line.split("\t")
#         # print (words[0] + " %s" %words[1])
#         results[words[0]] = words[1]
#     file.close()
#     return results

if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    train_data = load_file(training_file)
    
    # ngram_counts_file = "ngram_counts.txt.gz"
    # counts = load_ngram_counts(ngram_counts_file)



