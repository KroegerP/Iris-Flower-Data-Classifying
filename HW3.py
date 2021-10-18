# Authored by: Yahya Emara, Peter Kroeger, Ryan Kunkel, Griffin Ramsey, Ryan Rubadue
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


BIN_SIZES = [5,10,15,20]
SAMPLE_NUMBER = 50

def FunctionScore(test_data, predictions): # Weighted average of the precision and recall - Recall =TP / (TP+FN) - Precision = TP / (TP + FP)
    false_positive = 0
    true_positive = 0
    false_negative = 0
    true_negative = 0
    precision = 0
    recall = 0
    function_score = 0
    i = 0
    for temp, row in test_data.iterrows():
        if predictions[i]:
            # predict setosa
            if(row['species'] == 'setosa'):
                true_positive += 1
            else:
                false_positive += 1
        else:
            # predict not setosa
            if(row['species'] != 'setosa'):
                true_negative += 1
            else:
                false_negative += 1
        i += 1
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    function_score = 2 / ((1/recall) + (1/precision))
    fp_rateList, tp_rateList = ROCCurve(false_negative, false_positive, true_negative, true_positive, predictions)

    return function_score, fp_rateList, tp_rateList     # For a given attribute (col)

def ROCCurve(false_negative, false_positive, true_negative, true_positive, predictions): #Instead of test_data, should have a confidence value
    fp_rateList = [] # False Positive Rate = FP / F, where F is the toal number of false responses (Flower is not Setosa)
    tp_rateList = [] # True Positive Rate = TP / T, where T is the total number of true responses
    points = []
    fpr = 0
    tpr = 0
    T = predictions.count(1)
    F = predictions.count(0)
    for i in predictions: # Checking our predictions to create True Positive and False Positive Rates
        if i == 1:
            tpr += 1
        else:
            fpr += 1

        tp_rateList.append(tpr / T)
        fp_rateList.append(fpr / F)

    return fp_rateList, tp_rateList    # For a given attribute (col)

def get_test_sample(data):
    # Create random test sample of 50 entries
    return random.sample(population=data.index.tolist(), k=SAMPLE_NUMBER)


def get_bin_frequencies(data, num_bins):
    counts, bins, bars = plt.hist(data, bins=num_bins)
    plt.close()
    return counts, bins

def find_best_split(train_data, col, bin_centers, counts): # Searching for the best split for a given attribute (col)
    max_gain = 0
    optimal_bin_val = 0
    expect_larger_setosa = None
    for val in bin_centers:
        greaterSetosa = len(train_data[(train_data[col] > val) & (train_data['species'] == 'setosa')])
        greaterNonSetosa = len(train_data[(train_data[col] > val) & (train_data['species'] != 'setosa')])
        lesserSetosa = len(train_data[(train_data[col] < val) & (train_data['species'] == 'setosa')])
        lesserNonSetosa = len(train_data[(train_data[col] < val) & (train_data['species'] != 'setosa')])

        total_positive = greaterSetosa + greaterNonSetosa
        total_negative = lesserSetosa + lesserNonSetosa

        vals = [greaterSetosa, greaterNonSetosa, lesserSetosa, lesserNonSetosa]
        i = 0
        num_rows = len(train_data)
        e_total = (((vals[i] + vals[i+2]) / num_rows) * (-math.log2(((vals[i] + vals[i+2]) / num_rows)))) + \
                              (((vals[i + 1] + vals[i+3]) / num_rows) * (-math.log2(((vals[i + 1] + vals[i+3]) / num_rows))))

        total_positive = vals[i] + vals[i + 1]
        try:
            first_half = ((vals[i] / total_positive) * (-math.log2((vals[i] / total_positive))))
        except ValueError:
            first_half = 0
        try:
            second_half = ((vals[i + 1] / total_positive) * (-math.log2((vals[i + 1] / total_positive))))
        except ValueError:
            second_half = 0
        e_plus = first_half + second_half

        total_negative = vals[i + 2] + vals[i + 3]
        try:
            first_half = ((vals[i + 2] / total_negative) * (-math.log2((vals[i + 2] / total_negative))))
        except ValueError:
            first_half = 0
        try:
            second_half = ((vals[i + 3] / total_negative) * (-math.log2((vals[i + 3] / total_negative))))
        except ValueError:
            second_half = 0
        e_minus = first_half + second_half

        gain = e_total - total_positive/num_rows * e_plus - total_negative/num_rows * e_minus
        if gain > max_gain:
            max_gain = gain
            optimal_bin_val = val
            # If the first element of vals is greater than the third, there are more setosa entries above the optimal_bin_val
            expect_larger_setosa = (vals[0] > vals[2])

    return max_gain, optimal_bin_val, expect_larger_setosa

def get_predictions(test_data, optimal_bin_val, expect_larger_setosa, col):
    predictions = []
    for i, row in test_data.iterrows():
        if (row[col] > optimal_bin_val) == expect_larger_setosa:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

def GetAccuracy(test_data, optimal_bin_val, expect_larger_setosa, col): # If the Setosa attribute expected to be greater than the Versicolor/Virginica, expect_larger_setosa = 1
    correct_predictions = 0

    for i, row in test_data.iterrows():
        if (row[col] > optimal_bin_val) == expect_larger_setosa:
            # predict setosa
            if(row['species'] == 'setosa'):
                correct_predictions += 1
        else:
            # predict not setosa
            if(row['species'] != 'setosa'):
                correct_predictions += 1

    accuracy = 100 * (correct_predictions / len(test_data))

    return accuracy

def getBinCenters(bins):
    bin_centers = [0] * (len(bins) -1)
    for k in range(len(bins) -1):
        bin_centers[k] = (bins[k] + bins[k+1]) / 2
    return bin_centers

def printAccuracyTable(j, accuracyTable):
    print(f"Accuracies for {j} bins: ", accuracyTable)
    print(f"Max Accuracy: ", max(accuracyTable))
    print(f"Min Accuracy: ", min(accuracyTable))
    print(f"Avg Accuracy: ", (sum(accuracyTable)/len(accuracyTable)), "\n")
    return

def mean(data):
  return sum(data)/len(data)

def stdev(data):
  average = mean(data)
  var = sum([(i-average)**2 for i in data]) / (len(data) - 1)
  return math.sqrt(var)

def GetPriorProbability(data):
    setosa_count = 0
    for index, row in data.iterrows():
        if(row['species'] == 'setosa'):
            setosa_count += 1
    prior_probability = setosa_count / len(data)
    return prior_probability

def GetEvidence(val, data):         # Get evidence for a particular attribute/feature being less than our discretized value
    evidence_count = 0
    for row in data:
        if(row < val):
            evidence_count += 1
    evidence = evidence_count / len(data)
    return evidence

def GetLikelihood(val, data, col, expect_larger_setosa): # Likelihood of an instance being a setosa given that it has similar characteristics to a setosa
    likelihood_count = 0
    positiveCount = 0
    tempList = []
    for i, row in data.iterrows():
        if((row[col] < val) == expect_larger_setosa):
            positiveCount += 1
            if(row['species'] == 'setosa'):
                likelihood_count += 1
    likelihood = likelihood_count / positiveCount
    return likelihood

def plotAccuracyScores(accuracyScores):
    attributeList = ["sepal_length", "sepal_width", "petal_length", "pedal_width"]
    i = 0
    for score in accuracyScores:
        plt.plot(BIN_SIZES, score, label = attributeList[i])
        i += 1
        if i % 4 == 0:
            i = 0
    plt.xlabel("Bin Size")
    plt.ylabel("Accuracy Score")
    plt.title("Accuracy Scores Versus Bin Sizes")
    plt.legend(loc="lower right")
    plt.show()
    plt.close()

def PlotFScore(function_scoreList):
    plt.plot(BIN_SIZES, function_scoreList, label = "Function Score")
    plt.xlabel("Bin Size")
    plt.ylabel("Function Score")
    plt.title("Function Scores Versus Bin Sizes")
    plt.legend(loc="lower right")
    plt.show()
    plt.close()

def PlotROCPairs(roc_fpr, roc_tpr):
    i = 0
    for x, y in zip(roc_fpr, roc_tpr):
        plt.plot(x,y, label = BIN_SIZES[i])
        i += 1
        if i % 4 == 0:
            i = 0

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for each bin size")
    plt.legend(loc="lower right")
    plt.show()
    plt.close()

def id3(testing_data, indicies):
    print("Decision Trees\n")
    data = pd.read_csv('iris.csv')
    function_scoreList = []
    roc_tpr = []
    roc_fpr = []
    accuracyScores = []
    for j in range(5, 21, 5):

        attributeList = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        sample = get_test_sample(data)
        if (len(testing_data) == 0):
            test_data = data.loc[sample]
            train_data = data.drop(sample)
        else:
            test_data = testing_data
            train_data = data.drop(indicies)
        largest_gain = 0
        largest_gain_index = 0
        accuracyTable = []
        gainVals = []
        discretizedValues = []

        # For each pair of sets, create histograms of 5, 10, 15, 20 bins
        for col in data.columns:
            predictions = []
            if col == 'species':
                accuracyScores.append(accuracyTable)
                printAccuracyTable(j, accuracyTable)
                continue
            test_data = test_data.sort_values(by=col)

            counts, bins = get_bin_frequencies(train_data[col], j)

            bin_centers = getBinCenters(bins)

            max_gain, optimal_bin_val, expect_larger_setosa = find_best_split(train_data, col, bin_centers, counts)

            gainVals.append(max_gain)
            discretizedValues.append(optimal_bin_val)

            accuracyTable.append(GetAccuracy(test_data, optimal_bin_val, expect_larger_setosa, col))

        largest_gain = max(gainVals) # Because the data set is binary (it is a setosa or not), we only need to make a single decision in the tree
        largest_gain_index = gainVals.index(largest_gain) # Attribute for the decision is based on what attribute has the highest information gain

        predictions = get_predictions(test_data, discretizedValues[largest_gain_index], expect_larger_setosa, attributeList[largest_gain_index])

        function_score, fp_rateList, tp_rateList = FunctionScore(test_data, predictions)

        function_scoreList.append(function_score)
        roc_tpr.append(tp_rateList)
        roc_fpr.append(fp_rateList)

    return function_scoreList, roc_fpr, roc_tpr, accuracyScores, predictions, test_data, sample

def naive_bayes(testing_data, indicies):
    print("\nNaive Bayes\n")
    data = pd.read_csv('iris.csv')
    roc_tpr = []
    roc_fpr = []
    function_scoreList = []
    accuracyScores
    for j in range(5, 21, 5):

        sample = get_test_sample(data)
        if (len(testing_data) == 0):
            test_data = data.loc[sample]
            train_data = data.drop(sample)
        else:
            test_data = testing_data
            train_data = data.drop(indicies)
        evidenceProbabilities = []
        discretizedValues = []
        likelihoods = []
        predictions = []
        accuracyTable = []

        setosa_probability = GetPriorProbability(train_data)
        non_setosa_probability = 1 - setosa_probability

        arr = []
        for col in data.columns:
            if col == 'species':
                accuracyScores.append(accuracyTable)
                printAccuracyTable(j, accuracyTable)
                continue
            test_data = test_data.sort_values(by=col)
            counts, bins = get_bin_frequencies(train_data[col], j)
            bin_centers = getBinCenters(bins)
            max_gain, optimal_bin_val, expect_larger_setosa = find_best_split(train_data, col, bin_centers, counts)
            accuracyTable.append(GetAccuracy(test_data, optimal_bin_val, expect_larger_setosa, col))

            arr.append([optimal_bin_val, GetEvidence(optimal_bin_val, train_data[col]), GetLikelihood(optimal_bin_val, train_data, col, expect_larger_setosa), expect_larger_setosa])

        for i, row in test_data.iterrows():
            rowSpace = 0
            _likelihood = 1
            _evidence = 1
            inv_likelihood = 1
            inv_evidence = 1
            probability = 1
            non_probability = 1
            for i in range(len(row)):
                if rowSpace > 3:
                    continue
                if((row[i] < arr[rowSpace][0]) == arr[rowSpace][3]):  # Sepal length less than discretized value, use the evidence and likelihoods that
                    _likelihood *= arr[rowSpace][2]
                    _evidence *= arr[rowSpace][1]
                    inv_likelihood *= 1- arr[rowSpace][2]
                    inv_evidence *= 1- arr[rowSpace][1]
                else:                              # Sepal length greater than discretized value, use the evidence and likelihoods that
                    _likelihood *= (1 - arr[rowSpace][2])
                    _evidence *= (1- arr[rowSpace][1])
                    inv_likelihood *= arr[rowSpace][2]
                    inv_evidence *= arr[rowSpace][1]
                rowSpace += 1

            probability = (_likelihood * setosa_probability) / _evidence
            non_probability = (inv_likelihood * non_setosa_probability) / inv_evidence

            if (probability > non_probability):
                predictions.append(1)
            else:
                predictions.append(0)

        function_score, fp_rateList, tp_rateList = FunctionScore(test_data, predictions)
        function_scoreList.append(function_score)
        roc_tpr.append(tp_rateList)
        roc_fpr.append(fp_rateList)

    return function_scoreList, roc_fpr, roc_tpr, accuracyScores, predictions, test_data, sample

if __name__ == "__main__":
    id3_testing = []
    id3_indicies = []
    naive_testing = []
    naive_indicies = []
    function_scoreList, roc_fpr, roc_tpr, accuracyScores, id3_predictions, id3_testing, id3_indicies = id3(id3_testing, id3_indicies) # Getting a single function score and ROC curve data per bin

    plotAccuracyScores(accuracyScores)
    PlotFScore(function_scoreList)
    PlotROCPairs(roc_fpr, roc_tpr)

    function_scoreList = [] # Reset the variables for reuse
    accuracyScores = []
    roc_fpr = []
    roc_tpr = []

    function_scoreList, roc_fpr, roc_tpr, accuracyScores, naive_predictions, naive_testing, naive_indicies = naive_bayes(naive_testing, naive_indicies) # Getting a single function score and
    plotAccuracyScores(accuracyScores)
    PlotFScore(function_scoreList)
    PlotROCPairs(roc_fpr, roc_tpr)

    function_scoreList = [] # Reset the variables for reuse
    accuracyScores = []
    roc_fpr = []
    roc_tpr = []

    print("\nUsing decision trees as ground truth\n")
    function_scoreList, roc_fpr, roc_tpr, accuracyScores, naive_predictions, naive_testing, naive_indicies = naive_bayes(id3_testing, id3_indicies)
    plotAccuracyScores(accuracyScores)
    PlotFScore(function_scoreList)
    PlotROCPairs(roc_fpr, roc_tpr)

    function_scoreList = [] # Reset the variables for reuse
    accuracyScores = []
    roc_fpr = []
    roc_tpr = []

    print("\nUsing naive bayes as ground truth\n")
    function_scoreList, roc_fpr, roc_tpr, accuracyScores, id3_predictions, id3_testing, id3_indicies = id3(naive_testing, naive_indicies) # Getting a single function score and ROC curve data per bin
    plotAccuracyScores(accuracyScores)
    PlotFScore(function_scoreList)
    PlotROCPairs(roc_fpr, roc_tpr)
