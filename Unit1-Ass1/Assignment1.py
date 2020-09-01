"""
Assume df is a pandas dataframe object of the dataset given
"""
import numpy as np
import pandas as pd
import random

"""Calculate the entropy of the enitre dataset"""
# input:pandas_dataframe
# output:int/float/double/large
smallest = np.finfo(float).eps


def get_entropy_of_dataset(df):
    entropy = 0
    lastCol = df.keys()[-1]
    uniques = df[lastCol].unique()
    uniques.pop("NULL", "NA", "")
    for i in uniques:
        con = df[lastCol].value_counts()[i] / len(df[lastCol])
        entropy += -con * (np.log2(con))
    # print(entropy)
    return entropy


"""Return entropy of the attribute provided as parameter"""
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float/double/large
def get_entropy_of_attribute(df, attribute):
    entropy_of_attribute = 0
    lastCol = df.keys()[-1]
    uniques = df[lastCol].unique()
    attr_uniques = df[attribute].unique()
    for i in attr_uniques:
        entropy = 0
        for j in uniques:
            n = len(df[attribute][df[attribute] == i][df[lastCol] == j])
            d = len(df[attribute][df[attribute] == i])
            f = n / (d + smallest)
            entropy += -f * np.log2(f + smallest)
        f2 = d / len(df)
        entropy_of_attribute += -f2 * entropy
    return abs(entropy_of_attribute)


"""Return Information Gain of the attribute provided as parameter"""
# input:int/float/double/large,int/float/double/large
# output:int/float/double/large
def get_information_gain(df, attribute):
    information_gain = 0
    information_gain = get_entropy_of_dataset(df) - get_entropy_of_attribute(
        df, attribute
    )
    # print(information_gain)
    return information_gain


""" Returns Attribute with highest info gain"""
# input: pandas_dataframe
# output: ({dict},'str')
def get_selected_attribute(df):
    information_gains = {i: get_information_gain(df, i) for i in df.keys()[:-1]}
    selected_column = max(information_gains, key=information_gains.get)
    """
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	"""
    return (information_gains, selected_column)
