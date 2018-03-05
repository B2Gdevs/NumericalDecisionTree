import pandas as pd
import math
import numpy as np
from sklearn.cross_validation import KFold
import io
from contextlib import redirect_stdout
import sys

class Node:
    def __init__(self):
        self.parent = None
        self.leftNode = None#Must be a node
        self.rightNode = None #Must be a node
        self.threshold = None #best Threshold
        self.feature = None #name of feature
        self.isLeaf = False
        self.dataFrame = None
        self.classification = None
        self.visited = False
        self.height = 0


class Tree:
    def __init__(self):
        self.rootNode = None
        self.height = 0
        
def visualize_tree(prefix, node):
    if node.classification is not None:
        print(prefix + "Classification: " + str(node.classification))
    else:
        print(prefix + "Feature: " +node.feature+  " Threshold: " + str(node.threshold))
    
    if node.rightNode is not None:
        visualize_tree(prefix + " ", node.rightNode)
    if  node.leftNode is not None:
        visualize_tree(prefix + " ", node.leftNode)

     
def get_counts_with_threshold(dataframe, featureIndex, threshold, className):
    column_values = dataframe.iloc[:, featureIndex]
    yValues = get_yValue_column(dataframe)
    newYValues = []
    
    #needed to reindex the yValues
    for i in yValues:
        newYValues.append(i)
        
    lessThan_TrueCount = 0
    greaterThan_TrueCount = 0
    
    lessThan_FalseCount = 0
    greaterThan_FalseCount = 0
    

    for counter, i in enumerate(column_values):
        if i <= threshold:
            if newYValues[counter] == className:
                lessThan_TrueCount += 1
            else:
                lessThan_FalseCount += 1
        elif i > threshold:
            if newYValues[counter] == className:
                greaterThan_TrueCount += 1
            else:
                greaterThan_FalseCount += 1
    
    return lessThan_TrueCount, greaterThan_TrueCount, lessThan_FalseCount, greaterThan_FalseCount
    
def generate_tree(dataframe, parentNode, tree):
    
    if parentNode == None:
        parentNode = Node()
        tree.rootNode = parentNode
        parentNode.dataFrame = dataframe
        
    if dataset_entropy(parentNode.dataFrame) == 0:
        parentNode.isLeaf = True
        parentNode.classification = parentNode.dataFrame.iloc[:, -1].unique()
        return tree
        
    features = dataframe.columns
    featureIndicesMaxRange = len(features) - 1
    highestGain = 0
    highestGainFeatureIndex = 0
    bestThreshold = 0

    for i in range(0,featureIndicesMaxRange):
        threshold, gain = find_optimal_threshold_and_highest_gain(dataframe, i)
        if gain == 0:
            parentNode.feature = features[i]
            parentNode.threshold = threshold
            return tree
        if gain > highestGain:
            highestGain = gain
            highestGainFeatureIndex = i
            bestThreshold = threshold

    parentNode.feature = features[highestGainFeatureIndex]
    parentNode.threshold = bestThreshold
    
#==============================================================================
#     if parentNode != None:
#         print(parentNode.threshold)
#==============================================================================
    leftFrame = dataframe[dataframe[features[highestGainFeatureIndex]] <= bestThreshold]
    rightFrame = dataframe[dataframe[features[highestGainFeatureIndex]] > bestThreshold]
    
    leftNode = Node()
    rightNode = Node()
    
    leftNode.dataFrame = leftFrame
    rightNode.dataFrame = rightFrame
    #print(rightNode.dataFrame)
    
    leftNode.parent = parentNode
    rightNode.parent = parentNode
    
    parentNode.leftNode = leftNode
    parentNode.rightNode = rightNode
    
    leftNode.height = parentNode.height + 1
    rightNode.height = parentNode.height + 1

    #print(leftNode.dataFrame)
    

#==============================================================================
#     if dataset_entropy(leftFrame) == 0:
#         leftNode.isLeaf = True
#         leftNode.classification = leftFrame.iloc[:, -1].unique()
#         #print(leftNode.classification)
#     if dataset_entropy(rightFrame) == 0:
#         rightNode.isLeaf = True
#         rightNode.classification = rightFrame.iloc[:, -1].unique()
#==============================================================================
        #print(rightNode.classification)
    
    #print(rightNode.dataFrame.iloc[:, -1].unique())
    generate_tree(leftNode.dataFrame, leftNode, tree)
    generate_tree(rightNode.dataFrame, rightNode, tree)
    
    tree.height += 1

    #print(leftNode.classification)
    #print(rightFrame)
            
    #print(featureIndicesMaxRange)
    #print(parentNode.feature)
    return tree

def true_count_probability_of_threshold_counts(lessThan_TrueCount, greaterThan_TrueCount, lessThan_FalseCount, greaterThan_FalseCount):
    probabilityOfLessThanTrueCount = lessThan_TrueCount / (lessThan_TrueCount + lessThan_FalseCount + greaterThan_TrueCount + greaterThan_FalseCount)
    probabilityOfGreaterThanTrueCount = greaterThan_TrueCount / (lessThan_TrueCount + lessThan_FalseCount + greaterThan_TrueCount + greaterThan_FalseCount)
  
    return probabilityOfLessThanTrueCount, probabilityOfGreaterThanTrueCount
    
def generate_thresholds(dataframe, featureIndex, numberofthresholdstoCheck = 10):
    maxValue = dataframe.iloc[:, featureIndex].max()
    minValue = dataframe.iloc[:, featureIndex].min()
    step = (maxValue - minValue) / numberofthresholdstoCheck
    
    listOfThresholds = [n for n in np.arange(minValue, maxValue, step)]
    return listOfThresholds
    
def entropy(truecount, totalcount):
    
    entropy = 0
    probability = truecount / totalcount
    
    ######DEBUGGER#########################
    #print("\n\n\n probability in entropy")
    #print(probability)
    ######DEBUGGER#########################

    if(probability != 0):
                   # - p(x) * log2(p(x)) still needs - (1- p(x)) *log2((1-p(x))
        entropy -= probability * math.log(probability, 2)

        ######DEBUGGER#########################
        #print("\n\n\nfirst round of Entropy")
        #print(entropy)
        ######DEBUGGER#########################

    probability = (1 - probability)

    ######DEBUGGER#########################
    #print(("\n\n\n the next probability"))
    #print(probability)
    ######DEBUGGER#########################

    if(probability != 0):
        entropy -= probability * math.log(probability, 2)

        ######DEBUGGER#########################
        #print("\n\n\n the next part of entropy")
        #print(entropy)
        ######DEBUGGER#########################


    return entropy
    
def entropy_for_threshold_counts(truecount, falsecount):
    
    totalcount = truecount + falsecount
    entropy = 0
    probability = truecount / totalcount
    ######DEBUGGER#########################
    #print("\n\n\n probability in entropy")
    #print(probability)
    ######DEBUGGER#########################

    if(probability != 0):
                   # - p(x) * log2(p(x)) still needs - (1- p(x)) *log2((1-p(x))
        entropy -= probability * math.log(probability, 2)

        ######DEBUGGER#########################
        #print("\n\n\nfirst round of Entropy")
        #print(entropy)
        ######DEBUGGER#########################

    probability = (1 - probability)

    ######DEBUGGER#########################
    #print(("\n\n\n the next probability"))
    #print(probability)
    ######DEBUGGER#########################

    if(probability != 0):
        entropy -= probability * math.log(probability, 2)

        ######DEBUGGER#########################
        #print("\n\n\n the next part of entropy")
        #print(entropy)
        ######DEBUGGER#########################


    return entropy

def find_optimal_threshold_and_highest_gain(dataframe, featureIndex, numberOfThresholdstoCheck = 10):
    classNames = get_yValue_column(dataframe).unique()
    
    thresholds = generate_thresholds(dataframe, featureIndex, numberOfThresholdstoCheck)
    thresholdentropies = []
    
    for j in thresholds:
        for i in classNames:
            lessThan_TrueCount, greaterThan_TrueCount, lessThan_FalseCount, greaterThan_FalseCount = get_counts_with_threshold(dataframe, featureIndex, j,  i)
            lessThanT_prob, greaterThanT_prob = true_count_probability_of_threshold_counts(lessThan_TrueCount, greaterThan_TrueCount, lessThan_FalseCount, greaterThan_FalseCount)
            lessEntropThreshold = entropy_for_threshold_counts(lessThan_TrueCount, lessThan_FalseCount)
            greatEntropThreshold = entropy_for_threshold_counts(greaterThan_TrueCount, greaterThan_FalseCount)
            thresholdentropies.append((lessThanT_prob * lessEntropThreshold) + (greaterThanT_prob * greatEntropThreshold))
    portionedListOfThresholdEntropies = []
    thresholdEntropy = 0    
    

    j = 1
    for i in thresholdentropies:
        if j != len(classNames):
            thresholdEntropy += i
            j += 1     
        else:
            portionedListOfThresholdEntropies.append(thresholdEntropy)
            thresholdEntropy = 0
            j = 1
    datasetEntropy = dataset_entropy(dataframe)
    gains = []
    
    for i in portionedListOfThresholdEntropies:
        gains.append(information_gain(datasetEntropy, i))
    
    if gains == []:
        return 0, 0
        
    return thresholds[gains.index(max(gains))], max(gains)

def information_gain(datasetEntropy, thresholdEntropy):
    gain = datasetEntropy - thresholdEntropy
    return gain

def best_gain(datasetEntropy, featureEntropy):
    pass

def dataset_entropy(dataframe):
    entropyOfData = 0
    totalCount = get_total_count_of_rows(dataframe)
    countOfTrue = get_counts_of_classes(dataframe)
    
    for i in countOfTrue:
        entropyOfData += entropy(i[0], totalCount)
        
    return(entropyOfData / (len(dataframe.columns) - 1))
    

def get_total_count_of_rows(dataframe):
    return dataframe.shape[0]

def get_counts_of_classes(dataframe):
    #Grabbing the last column index INTEGER
    classifyingColumnIndex = get_yValue_column_index(dataframe)
    
    #creates 2-tuples of the total count of the classes and the name of the classes
    zipOfCountAndClass = zip(dataframe.iloc[:, classifyingColumnIndex].value_counts(), dataframe.iloc[:, classifyingColumnIndex].unique())
    
    #transforms it to a list of tuples rather than a zip object
    listOf2Tuples_Count_Class = list(zipOfCountAndClass)
    
    return listOf2Tuples_Count_Class
    
    #shortcut method rather than figuring out indexes again...
def get_yValue_column_index(dataframe):
    classifyingColumnIndex = len(dataframe.columns) - 1
    return classifyingColumnIndex

    #shortcut method rather than getting figuring out the rest of the data without classes
def get_all_data_other_than_last_column(dataframe):
    classColumnIndex = get_yValue_column_index(dataframe)
    dataframeWithoutClasses = dataframe.iloc[:, 0:(classColumnIndex - 1)]
    return dataframeWithoutClasses

    #shortcut method to get last column data
def get_yValue_column(dataframe):
    columnIntegerIndex = get_yValue_column_index(dataframe)
    yValues = dataframe.iloc[:, columnIntegerIndex]
    return yValues
    
    #normalize a dataframe...Do Not Have Classification Columns When Using
def normalizeData(dataframe):
    dataframeWithoutClasses = get_all_data_other_than_last_column(dataframe)
    yValuesColumn = get_yValue_column(dataframe)
    normalizedDataframe = (dataframeWithoutClasses - dataframeWithoutClasses.mean()) / (dataframeWithoutClasses.max() - dataframeWithoutClasses.min())
    normalizedDataframe["classification"] = yValuesColumn
    return(normalizedDataframe)
    
def predict(node, series):
    
    if node.feature is not None:
        if float(series[node.feature]) <= node.threshold:
            if node.leftNode is not None:
                return predict(node.leftNode, series)
        elif series[node.feature] > node.threshold:
            if node.rightNode is not None:
                return predict(node.rightNode, series)
    else:
        return node.classification

def cross_validation(dataframe, K):
    
    folds = KFold(dataframe.shape[0], n_folds = K)
    
    testSets = []
    trainSets = []
    for train, test in folds:
        trainSets.append(dataframe.iloc[train])
        testSets.append(dataframe.iloc[test])
        
    trees = []
    for i in trainSets:
        frame = pd.DataFrame(i)
        tree = generate_tree(frame, None, Tree())
        trees.append(tree)

    
    i = 0
    accuracyList = []
    for tree in trees:
        for frame in testSets:
            trueCount = 0
            for index, series in frame.iterrows():
                prediction = predict(tree.rootNode, series)
                if prediction == series[-1]:
                    trueCount += 1
        i += 1
        accuracyList.append(trueCount / frame.shape[0])
        
    meanAccuracy = sum(accuracyList) / len(accuracyList)
    return meanAccuracy

def main(argv):

    #====================Get Features==================

    indicesOfFeaturesWanted = [int(i) for i in argv[2:]]
    print(indicesOfFeaturesWanted)
    indices = []
    for i in indicesOfFeaturesWanted:
        indices.append(i - 1)
        
    print(indices)
    #Grabs file name and reads the file
    Data = pd.read_csv(argv[1])
    #print(Data)
    rows, cols = Data.shape
    indices.append(cols - 1)
    featuresWanted = Data.iloc[:, indices]
    tree = Tree()
    
    generate_tree(featuresWanted, None, tree)
    
    f = io.StringIO()
    with redirect_stdout(f):
        visualize_tree("", tree.rootNode)
    out = f.getvalue()
    
    file = open("decisionTree.txt", 'w')
    file.write(out)
    file.close()
    
    accuracy = cross_validation(featuresWanted, 10)
    
    print("The average accuracy of the model is " + str(accuracy) + " based on 10 " +
          "fold cross-validation.  An output of the initial tree is saved in a text file "+
          "called decisionTree.txt")
    
if __name__ == "__main__":
    main(sys.argv)
    
