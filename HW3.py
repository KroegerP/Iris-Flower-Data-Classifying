# Machine Learning HW 3
# Yayha Emara, Peter Kroeger, Ryan Rubadue, Ryan Kunkel, Griffin Ramsey

import csv
import math
import random
import numpy

data = []
with open('iris.csv', newline='') as csvFile:
    dataReader = csv.reader(csvFile, delimiter=',')
    firstRow = True
    for row in dataReader:
        dataRow = {}
        if not firstRow:    # Don't want to translate the name row into the dataset
            dataRow["sepal_length"] = float(row[0])
            dataRow["sepal_width"] = float(row[1])
            dataRow["petal_length"] = float(row[2])
            dataRow["petal_width"] = float(row[3])
            dataRow["species"] = row[4]
            data.append(dataRow)
        firstRow = False
print(data)

attributeList = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

testSet = random.sample