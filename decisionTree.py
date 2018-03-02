import pandas as pd
import math
import numpy as np
from ete3 import Tree

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



class Treee:
    def __init__(self):
        self.rootNode = None
        self.height = 0
        
def visualize_tree(node):
     result = ''
     hasRightNode = False;
     hasLeftNode = False;
     if node.rightNode != None:
         hasRightNode = True
         result = result + "("
         result = result + visualize_tree(node.rightNode)
     if node.leftNode != None:
        hasLeftNode = True
        if hasRightNode:
            result = result + ","
        else:
            result = result + "("
        result = result + visualize_tree(node.leftNode)
     if hasLeftNode or hasRightNode:
        result = result + ")"
     if node.feature is not None:
         result = result + node.feature
     return result
     

        

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
    
    print(leftNode.height)
    print(rightNode.height)
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
    
    
iris_data = pd.read_csv("iris.csv")
tree = Treee()

tree = generate_tree(iris_data, None, tree)

print(tree.height)
print(Tree(visualize_tree(tree.rootNode)+';'))



