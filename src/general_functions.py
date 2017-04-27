# Created by JJW Apr 22 2016
# This contains some general functions (like I/O functions)
# that are used by many of the codes
# for COS 424 final project


ignore_after_value = 98  # Ignore columns past this value
columns_to_remove = [53, 47, 46, 33, 1, 0]  # Ignore these columns (mostly just the name of the object listed several times)

line_to_ignore_unselected = 19750  # This particular line of data is bad


def read_in_data(path="../data/"):
    """Reads in the data and returns a feature set with classification

    Returns: tuple, first element is the feature set (2-d list) and
             second element is a list of the categories of the data

    Keyword Arguments:
        path {str} -- path to the two data files (default: {"../data/"})
    """

    # First read in the data for the candidates that were selected (category 1)
    with open(path + "selectedcandidates.txt", "r") as f:
        lines = f.readlines()
        data_selected = []  # Empty list to collect the data

        for line in lines:
            temp = line.split()
            temp = temp[:ignore_after_value]  # Cut off unused values at end
            for i in columns_to_remove:
                _ = temp.pop(i)  # Remove the columns we need to ignore
            data_selected.append(temp)

    # Convert all the data values from string to float
    for i in range(len(data_selected)):
        for j in range(len(data_selected[i])):
            data_selected[i][j] = float(data_selected[i][j])

    # Now read in the data for the unselected candidates (category 0)
    with open(path + "unselectedcandidates.txt", "r") as f:
        lines = f.readlines()
        data_unselected = []  # Empty list to collect the data

        for j in range(len(lines)):
            if j == line_to_ignore_unselected:  # Ignore the bad line
                continue
            temp = lines[j].split()
            temp = temp[:ignore_after_value]  # Cut off unused values at the end
            for i in columns_to_remove:
                _ = temp.pop(i)  # Remove the columns we need to ignore
            data_unselected.append(temp)

    # Convert all data values to float
    for i in range(len(data_unselected)):
        for j in range(len(data_unselected[i])):
            data_unselected[i][j] = float(data_unselected[i][j])

    # Create a list that quantifies the classification
    was_selected = [1] * len(data_selected)
    was_selected += [0] * len(data_unselected)

    # Return the data and the classification
    return (data_selected + data_unselected, was_selected)


"""
a, b = read_in_data()

import numpy as np

i = 0
i2 = 0
for j in range(len(a)):
    if any([np.isnan(item) for item in a[j] ]):
        if j < 1000:
            i+= 1
        else:
            i2+=1
print i
print i2
"""


def precision_recall_etc(classified_sentiment, actual_sentiment):
    if len(classified_sentiment) != len(actual_sentiment):  # if lengths don't match
        raise RuntimeError("Lengths of arguments to accuracy_percentage not the same")
    tp = fp = tn = fn = 0  # t=true, f=false, p=postive, n=negative
    for i in range(len(classified_sentiment)):
        if actual_sentiment[i] == 1:  # actual sentiment is positive
            if classified_sentiment[i] == actual_sentiment[i]:  # if matches
                tp += 1
            else:  # if doesn't match
                fn += 1
        else:  # actual sentiment is negative
            if classified_sentiment[i] == actual_sentiment[i]:  # if matches
                tn += 1
            else:  #if doesn't match
                fp += 1

    # calculate the various performance metrics
    precision = float(tp)/float(tp + fp)
    recall = float(tp)/float(tp + fn)
    specificity = float(tn)/float(fp + tn)
    NPV = float(tn)/float(tn + fn)
    f1 = 2.*float(precision*recall)/float(precision + recall)

    return {'precision': precision, 'recall': recall,
            'specificity': specificity, 'NPV': NPV,
            'f1': f1, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'accuracy': float(tp + tn)/float(tp + fp + tn + fn)}
