import numpy as np

def evaluate(model):
    
    print "Results -"
    print "LogLoss: %f" % logloss(model.Result,model.Prediction)
    numCorrectPred = np.count_nonzero(np.sign(model.Prediction-0.5) == np.sign(model.Result - 0.5))
    numPred = model.loc[model.Prediction!=0.5].shape[0]
    print "%.4f Correct: %d out of %d" % (numCorrectPred / float(1 if numPred == 0 else numPred), numCorrectPred, numPred)
#    from IPython.core.debugger import Tracer
#    Tracer()()
    for limit in [0.7,0.8,0.9]:
        highconviction = model.loc[(model.Prediction >= limit) | (model.Prediction <= (1-limit))]
        numCorrectPred = np.count_nonzero(np.sign(highconviction.Prediction-0.5) == np.sign(highconviction.Result - 0.5))
        numPred = highconviction.shape[0]
        print "high conviction (>%.2f pct): %.4f Correct: %d out of %d" % (limit,numCorrectPred / float(1 if numPred == 0 else numPred), numCorrectPred, numPred)

    print "2013-2016"
    stage1model = model.loc[(model.Season>=2013) & (model.Season<2017)]
    print "LogLoss: %f" % logloss(stage1model.Result,stage1model.Prediction)
    numCorrectPred = np.count_nonzero(np.sign(stage1model.Prediction-0.5) == np.sign(stage1model.Result - 0.5))
    numPred = stage1model.loc[stage1model.Prediction!=0.5].shape[0]
    print "%.4f Correct: %d out of %d" % (numCorrectPred / float(1 if numPred == 0 else numPred), numCorrectPred, numPred)
    highconviction = stage1model.loc[(stage1model.Prediction >= 0.8) | (stage1model.Prediction <= 0.2)]
    numCorrectPred = np.count_nonzero(np.sign(highconviction.Prediction-0.5) == np.sign(highconviction.Result - 0.5))
    numPred = highconviction.shape[0]
    print "high conviction (>80pct): %.4f Correct: %d out of %d" % (numCorrectPred / float(1 if numPred == 0 else numPred), numCorrectPred, numPred)

    
    print "Logloss per year..."
    for season in model.Season.unique():
        seasonMatches = model.loc[model.Season == season]
        print "%d: %f" % (season, logloss(seasonMatches.Result, seasonMatches.Prediction))
        numCorrectPred = np.count_nonzero(np.sign(seasonMatches.Prediction-0.5) == np.sign(seasonMatches.Result - 0.5))
        numPred = seasonMatches.loc[seasonMatches.Prediction!=0.5].shape[0]
        print "%.4f Correct: %d out of %d" % (numCorrectPred / float(1 if numPred == 0 else numPred), numCorrectPred, numPred)
        highconviction = seasonMatches.loc[(seasonMatches.Prediction >= 0.8) | (seasonMatches.Prediction <= 0.2)]
        numCorrectPred = np.count_nonzero(np.sign(highconviction.Prediction-0.5) == np.sign(highconviction.Result - 0.5))
        numPred = highconviction.shape[0]
        print "high conviction (>80pct): %.4f Correct: %d out of %d" % (numCorrectPred / float(1 if numPred == 0 else numPred), numCorrectPred, numPred)
        

def logloss(actual, prediction):
    epsilon = 1e-15
    prediction = np.maximum(prediction, epsilon)
    prediction = np.minimum(prediction, 1-epsilon)

    logl = actual * np.log(prediction) + (1-actual) * np.log(1-prediction)  
    return -np.mean(logl)
        
