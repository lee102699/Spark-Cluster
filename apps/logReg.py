
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkConf, SparkContext

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
    #feats.insert(0,label)
    #features = [ float(feature) for feature in feats ] # need floats
    return LabeledPoint(label, feats)

# Load and parse the data
sc = getSparkContext()
data = sc.textFile("./opt/spark-data/data_banknote_authentication.txt")
parsedData = data.map(mapper)

# Train model
iterations = int(10000)
model = LogisticRegressionWithSGD.train(parsedData,iterations)

# Predict the first elem will be actual data and the second
# item will be the prediction of the model
labelsAndPreds = parsedData.map(lambda point: (int(point.label),
        model.predict(point.features)))

# Evaluating the model on training data
trainErr = labelsAndPreds.filter(lambda vp: vp[0] != vp[1]).count() / float(parsedData.count())

# Print some stuff
print("Training Error = " + str(trainErr))
print("Final weights: " + str(model.weights))
print("Final intercept: " + str(model.intercept))