import numpy as np

def evaluate(model):
    
    print "Results -"
    print "LogLoss: %f" % logloss(model.result,model.prediction)
    numCorrectPred = np.count_nonzero(np.sign(model.prediction-0.5) == np.sign(model.result - 0.5))
    numPred = (model.prediction==0.5).value_counts()[False]
    print "%.4f Correct: %d out of %d" % (numCorrectPred / float(numPred), numCorrectPred, numPred)
#    from IPython.core.debugger import Tracer
#    Tracer()()
    for limit in [0.7,0.8,0.9]:
        highconviction = model.loc[(model.prediction >= limit) | (model.prediction <= (1-limit))]
        numCorrectPred = np.count_nonzero(np.sign(highconviction.prediction-0.5) == np.sign(highconviction.result - 0.5))
        numPred = highconviction.shape[0]
        print "high conviction (>%.2f pct): %.4f Correct: %d out of %d" % (limit,numCorrectPred / float(1 if numPred == 0 else numPred), numCorrectPred, numPred)

    print "2011-2014"
    stage1model = model.loc[(model.season>=2011) & (model.season<2015)]
    print "LogLoss: %f" % logloss(stage1model.result,stage1model.prediction)
    numCorrectPred = np.count_nonzero(np.sign(stage1model.prediction-0.5) == np.sign(stage1model.result - 0.5))
    numPred = (stage1model.prediction==0.5).value_counts()[False]
    print "%.4f Correct: %d out of %d" % (numCorrectPred / float(numPred), numCorrectPred, numPred)
    highconviction = stage1model.loc[(stage1model.prediction >= 0.8) | (stage1model.prediction <= 0.2)]
    numCorrectPred = np.count_nonzero(np.sign(highconviction.prediction-0.5) == np.sign(highconviction.result - 0.5))
    numPred = highconviction.shape[0]
    print "high conviction (>80pct): %.4f Correct: %d out of %d" % (numCorrectPred / float(1 if numPred == 0 else numPred), numCorrectPred, numPred)

    
    print "Logloss per year..."
    for season in model.season.unique():
        seasonMatches = model.loc[model.season == season]
        print "%d: %f" % (season, logloss(seasonMatches.result, seasonMatches.prediction))
        numCorrectPred = np.count_nonzero(np.sign(seasonMatches.prediction-0.5) == np.sign(seasonMatches.result - 0.5))
        numPred = (seasonMatches.prediction==0.5).value_counts()[False]
        print "%.4f Correct: %d out of %d" % (numCorrectPred / float(numPred), numCorrectPred, numPred)
        highconviction = seasonMatches.loc[(seasonMatches.prediction >= 0.8) | (seasonMatches.prediction <= 0.2)]
        numCorrectPred = np.count_nonzero(np.sign(highconviction.prediction-0.5) == np.sign(highconviction.result - 0.5))
        numPred = highconviction.shape[0]
        print "high conviction (>80pct): %.4f Correct: %d out of %d" % (numCorrectPred / float(1 if numPred == 0 else numPred), numCorrectPred, numPred)

    
                                    

def logloss(actual, prediction):
    epsilon = 1e-15
    prediction = np.maximum(prediction, epsilon)
    prediction = np.minimum(prediction, 1-epsilon)

    logl = actual * np.log(prediction) + (1-actual) * np.log(1-prediction)  
    return -np.mean(logl)
        
