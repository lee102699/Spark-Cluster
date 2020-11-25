from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkConf, SparkContext
import numpy as np

def getSparkContext():
    """
    Gets the Spark Context
    """
    conf = (SparkConf()
         .setMaster("local") # run on local
         .setAppName("Logistic Regression") # Name of App
         .set("spark.executor.memory", "1g")) # Set 1 gig of memory
    sc = SparkContext(conf = conf) 
    return sc

def mapper(line):
    """
    Mapper that converts an input line to a feature vector
    """    
    feats = line.strip().split(",") 
    # labels must be at the beginning for LRSGD, it's in the end in our data, so 
    # putting it in the right place
    label = feats[len(feats) - 1] 
    feats = feats[: len(feats) - 1]
    feats.insert(0,label)
    features = [ float(feature) for feature in feats ] # need floats
    return np.array(features)
    #return LabeledPoint(label, feats)

def gradient(row, w):
    Y = row[0]
    X = row[1:]
    return ((1.0 / (1.0 + np.exp(-Y * X.dot(w))) - 1.0) * Y * X.T).sum()

def add(x, y):
    x += y
    return x

# Load and parse the data
sc = getSparkContext()
data = sc.textFile("./opt/spark-data/data_banknote_authentication.txt")
parsedData = data.map(mapper).cache()

w = 2 * np.random.ranf(size=4) -1
for i in range(100):
    w -= parsedData.map(lambda r: gradient(r,w)).reduce(add)

labelsAndPreds = parsedData.map(lambda point: (int(point[0]), 1.0 / (1.0 + np.exp(-1 * w.dot(point[1:])))))
# Evaluating the model on training data
trainErr = labelsAndPreds.filter(lambda vp: vp[0] != vp[1]).count() / float(parsedData.count())

# Print some stuff
print("Training Error = " + str(trainErr))
