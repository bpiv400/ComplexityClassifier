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
import matplotlib 
import numpy as np
plt = matplotlib.pyplot
#pylab.savefig
def sum_el(same, val, pred_list, true_list):
    sum = 0
    for i in range(len(pred_list)):
        if true_list[i] == val:
            if (pred_list[i] == true_list[i]) == same:
                sum += 1 
    return sum


#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

## Calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
    denom =  (sum_el(same = True, val = 1, pred_list = y_pred, true_list = y_true) + 
        sum_el(same = False, val = 0, pred_list = y_pred, true_list = y_true))
    if denom == 0: 
        return 1
    else:
        precision = sum_el(same = True, val = 1, pred_list = y_pred, true_list = y_true) / denom
        return precision
    
## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    denom = (sum_el(same = True, val = 1, pred_list = y_pred, true_list = y_true) +
        sum_el(same = False, val = 1, pred_list = y_pred, true_list = y_true))
    if denom == 0: 
        return 1
    else:
        recall = sum_el(same = True, val = 1, pred_list = y_pred, true_list = y_true) / denom
        return recall

## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    recall = get_recall(y_pred, y_true)
    precision = get_precision(y_pred, y_true)
    fscore = 2 * recall * precision/(recall + precision)
    return fscore

#### 2. Complex Word Identification ####
y_prediction = [1, 1, 1, 1, 1, 0, 0, 0, 0]
y_truth = [1, 1, 0, 0, 0, 1, 1, 0, 0]

def test_predictions(y_pred, y_true):
    print("Recall: " + str(get_recall(y_pred, y_true)))
    print("Precision: " + str(get_precision(y_pred, y_true)))
    print("Fscore: " + str(get_fscore(y_pred, y_true)))

test_predictions(y_prediction, y_truth)

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
    training_dic = load_file(data_file)
    pred_list = list()
    training_list = list()
    for key in training_dic.keys():
        training_list.append(training_dic[key])
        pred_list.append(1)
    precision = get_precision(pred_list, training_list)
    recall = get_recall(pred_list, training_list)
    fscore = get_fscore(pred_list, training_list)
    performance = [precision, recall, fscore]
    return performance

print(str(range(2,20)))
### 2.2: Word length thresholding

## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file):
    # Tests word_length thresholds from 2 to 29
    # For a given threshold i, classifies a word as simple if len(word) < i
    # and complex if len(word) > i
    # Additionally creates & saves a precision-recall curve
    # Plots precision on the y axis and recall on the x axis
    training_dic = load_file(training_file)
    training_precision = numpy.zeroes(1, 28)
    training_recall = numpy.zeroes(1, 28)
    training_fscore = numpy.zeroes(1, 28)
    development_dic = load_file(development_file)
    best_thresh = 1
    best_fscore, best_recall, best_precision = 0
    tprecision, trecall, tfscore = 0
    for thresh in range(2, 30):
        i = thresh - 2
        training_vec = list()
        pred_vec = list()
        for key in training_dic.keys():
            length = len(key)
            if length < thresh:
                pred_vec.append(0)
            else:
                pred_vec.append(1)
            training_vec.append(training_dic[key])
        tfscore = get_fscore(pred_vec, training_vec)
        tprecision = get_precision(pred_vec, training_vec)
        trecall = get_recall(pred_vec, training_vec)
        training_precision[i] = tprecision
        training_recall[i] = trecall
        training_fscore[i] = tfscore
        if tfscore > best_fscore:
            best_thresh = thresh
            best_fscore = tfscore
            best_precision = tprecision
            best_recall = trecall
    dev_dic = load_file(development_file)
    dev_vec = list()
    pred_vec = list()

    for key in dev_dic.keys():
        dev_vec.append(dev_dic[key])
        length = len(key)
        if length < key:
            pred_vec.append(0)
        else: 
            pred_vec.append(1)
    dprecision = get_precision(pred_vec, dev_vec)
    drecall = get_recall(pred_vec, dev_vec)
    dfscore = get_fscore(pred_vec, dev_vec)
    threshold_performance = [trainining_precision, training_recall, training_fscore]
    training_performance = [best_precision, best_recall, best_fscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance, threshold_performance

def precision_recall_plot(recall_vec, precision_vec, name):
    plt.plot(recall_vec, precision_vec, '-')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for " + name + " Baseline")
    file_name = name + ".png"
    plt.show()
    plt.savefig(file_name)
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


## if __name__ == "__main__":
 #   training_file = "data/complex_words_training.txt"
 #   development_file = "data/complex_words_development.txt"
 #   test_file = "data/complex_words_test_unlabeled.txt"
#
 #   train_data = load_file(training_file)
 #   
 #   ngram_counts_file = "ngram_counts.txt.gz"
 #   counts = load_ngram_counts(ngram_counts_file)
