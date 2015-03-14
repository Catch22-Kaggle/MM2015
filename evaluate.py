import numpy as np

def evaluate(model):
    
    print "Results -"
    print "LogLoss: %f" % logloss(model.result,model.prediction)
    numCorrectPred = np.count_nonzero(np.sign(model.prediction-0.5) == np.sign(model.result - 0.5))
    numPred = (model.prediction==0.5).value_counts()[False]
    print "%.4f Correct: %d out of %d" % (numCorrectPred / float(numPred), numCorrectPred, numPred)

    print "2011-2014"
    print "LogLoss: %f" % logloss(model.result[model.season>=2011],model.prediction[model.season>=2011])
    numCorrectPred = np.count_nonzero(np.sign(model.prediction[model.season>=2011]-0.5) == np.sign(model.result[model.season>=2011] - 0.5))
    numPred = (model.prediction[model.season>=2011]==0.5).value_counts()[False]
    print "%.4f Correct: %d out of %d" % (numCorrectPred / float(numPred), numCorrectPred, numPred)
    
    print "Logloss per year..."
    for season in model.season.unique():
        seasonMatches = model.loc[model.season == season]
        print "%d: %f" % (season, logloss(seasonMatches.result, seasonMatches.prediction))
        numCorrectPred = np.count_nonzero(np.sign(seasonMatches.prediction-0.5) == np.sign(seasonMatches.result - 0.5))
        numPred = (seasonMatches.prediction==0.5).value_counts()[False]
        print "%.4f Correct: %d out of %d" % (numCorrectPred / float(numPred), numCorrectPred, numPred)
    
                                    

def logloss(actual, prediction):
    epsilon = 1e-15
    prediction = np.maximum(prediction, epsilon)
    prediction = np.minimum(prediction, 1-epsilon)

    logl = actual * np.log(prediction) + (1-actual) * np.log(1-prediction)  
    return -np.mean(logl)
        
