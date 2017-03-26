from scipy.optimize._minimize import minimize

import pandas as pd
import numpy as np
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.linear_model.logistic import LogisticRegression,\
    LogisticRegressionCV
from sklearn.neighbors.classification import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier

from evaluate import logloss
import statsmodels.api as sm
import multiprocessing
import itertools
from sklearn.metrics.classification import classification_report,\
    confusion_matrix, log_loss, accuracy_score
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from numpy.matlib import rand


def doF(args):
    (match_id, row), matchupSlots = args
    roundPlayed = matchupSlots.loc[(matchupSlots.T1t2 == (row["T1_id"]+row["T2_id"])) | (matchupSlots.T1t2 == (row["T2_id"]+row["T1_id"]))]
    
    return roundPlayed.Round.values[0]

def doG(args):
    (match_id, row), groupedTeamStats, xCols, X = args
    diffs = groupedTeamStats.loc[row["T1_id"]] - groupedTeamStats.loc[row["T2_id"]]
    X.loc[match_id,xCols] = diffs.values[1:]
    
def expWeight(n, a=None):
    '''
    function to generate an exponential weight series of length n
    '''
    alpha = 1- (a if a else 1.5/(1+n))
#     alpha = 1- (a if a else 4.0/(1+n))
    ixs = range(n-1,-1,-1)
    wts = [alpha**ix for ix in ixs]
    return wts / np.sum(wts)

# def expWeight(n, a=None):
#     '''
#     function to generate an exponential weight series of length n
#     '''
#     alpha = 0.25
# #     alpha = 1- (a if a else 4.0/(1+n))
#     ixs = range(n-1,-1,-1)
#     wts = [np.exp(-alpha*ix) for ix in ixs]
#     
#     return wts / np.sum(wts)    
     
    

def model1(tourn, reg, matchups, priorMatchupMeanMargin, matchupSlots):
    # logistic regression
    
    
    #daynum
    #wloc
    #numot
    
    #wfgm - field goals made
    #wfga - field goals attempted
    #wfgm3 - three pointers made
    #wfga3 - three pointers attempted
    #wftm - free throws made
    #wfta - free throws attempted
    #wor - offensive rebounds
    #wdr - defensive rebounds
    #wast - assists
    #wto - turnovers
    #wstl - steals
    #wblk - blocks
    #wpf - personal fouls
    
    tourn["T1_id"] = tourn.Season.map(str) + "_" + tourn.T1.map(str) 
    tourn["T2_id"] = tourn.Season.map(str) + "_" + tourn.T2.map(str)
    
    
    #re-organise regualr season details by team rather than match
    reg["Wteam_id"] = reg.Season.map(str) + "_" + reg.Wteam.map(str) 
    reg["Lteam_id"] = reg.Season.map(str) + "_" + reg.Lteam.map(str)
    reg["WinningMargin"] = reg.Wscore - reg.Lscore
    
    priorMatchupMeanMargin.fillna(0.0, inplace=True)
    
    
    teams = pd.concat([reg.Wteam_id, reg.Lteam_id]).unique()
    teams.sort()
    
    teamStats = pd.DataFrame(index=reg.index)
    statCols = ["Team_id","Daynum","WinningMargin","Fgm","Fga","Fgm3","Fga3","Ftm","Fta","Or","Dr","Ast","To","Stl","Blk","Pf"]
    winCols = ["Wteam_id","Daynum","WinningMargin","Wfgm","Wfga","Wfgm3","Wfga3","Wftm","Wfta","Wor","Wdr","Wast","Wto","Wstl","Wblk","Wpf"]
    loseCols = ["Lteam_id","Daynum","WinningMargin","Lfgm","Lfga","Lfgm3","Lfga3","Lftm","Lfta","Lor","Ldr","Last","Lto","Lstl","Lblk","Lpf"]
    teamStats = reg[winCols]
    loseStats = reg[loseCols]
    loseStats["WinningMargin"] = loseStats.WinningMargin.multiply(-1)
    teamStats.columns = statCols
    loseStats.columns = statCols
    teamStats = pd.concat([teamStats, loseStats])
    
    
    # set up regression
    # Y = actual results (1=t1Wins, 0=t2Wins)
    # X = :
    # difference in avg winningMargin (t1-t2)
    # difference in avg field goals made
    # difference in avg field goals attempted
    # difference in avg 3pt goals made
    # difference in avg 3pt goals attempted
    # difference in avg freethrows made
    # difference in avg freethrows attempted
    # difference in avg offensive rebounds
    # difference in avg defensive rebounds
    # difference in avg steals
    # difference in avg blocks
    # difference in avg personal fouls
    
    # each row is a tournament match
    
    # use an out of sample - 2013-14
    tourn_in = tourn.loc[tourn.Season < 2013]
    tourn_out = tourn.loc[(tourn.Season >= 2013) & (tourn.Season <= 2016)]
    matchups_in = matchups#.loc[matchups.Season<2013]
    
    y_insample = tourn_in.Result
    y_out = tourn_out.Result
    y_fullsample = tourn.Result

    X = pd.DataFrame(index=matchups_in.index)
    xCols = ["AvgWinMargDiff",
             "AvgFGMDiff",
             "AvgFGADiff",
             "AvgFGPctDiff",
             "AvgFGM3Diff",
             "AvgFGA3Diff",
             "AvgFG3PctDiff",
             "AvgFTMDiff",
             "AvgFTADiff",
             "AvgFTPctDiff",
             "AvgORDiff",
             "AvgDRDiff",
             "AvgAstDiff",
             "AvgToDiff",
             "AvgTOAstDiff",
             "AvgStlDiff",
             "AvgBlDiff",
             "AvgPFDiff",
             ]
    for col in xCols:
        X[col] = np.nan
    
    X["SeedDiff"] = np.nan

    teamStats.insert(5, "Fgpct", teamStats.Fgm/teamStats.Fga)
    teamStats.insert(8, "Fg3pct", teamStats.Fgm3/teamStats.Fga3)
    teamStats.insert(11, "Ftpct", teamStats.Ftm/teamStats.Fta)
    teamStats.insert(15, "ToAst", teamStats.Ast / teamStats.To)
    
    # handle any infinit percentages
    teamStats.replace(np.inf,np.nan, inplace=True)
    
    
#     ewma =  pd.stats.moments.ewma
    
    def expwMean(series):
        
        expp = expWeight(series.shape[0]).reshape((-1,1))
        series.sort("Daynum",inplace=True)
        res = pd.Series(np.sum(series.values[:,1:]*expp, axis=0),index=series.columns[1:])
        if np.isnan(res.Ftpct):
            res.Ftpct = series.Ftpct.mean()
        if np.isnan(res.Fgpct):
            res.Fgpct = series.Fgpct.mean()
        if np.isnan(res.Fg3pct):
            res.Fg3pct = series.Fg3pct.mean()
        #series.values[-1,0]
        return res
        #return np.sum(series.values[:,1:]*expp, axis=0)
    
    
#     grouped = teamStats.groupby(teamStats.team_id)
#     groupedTeamStats = grouped.agg(expwMean) # skips nans by default
    
    # equally weighted average
    groupedTeamStats = teamStats.groupby(teamStats.Team_id).mean() # skips nans by default
    
    matchupSlots["T1t2"] = matchupSlots.Strongteam+matchupSlots.Weakteam
    
    matchups_in["T1t2"] = matchups_in.T1_id+matchups_in.T2_id
    
    
            
    
    for match_id, row in matchups_in.iterrows():
         
#         t1Matches = teamStats.loc[teamStats.team_id == row["t1_id"]]
#         t2Matches = teamStats.loc[teamStats.team_id == row["t2_id"]]
#         diffs = t1Matches.mean() - t2Matches.mean()
        diffs = groupedTeamStats.loc[row["T1_id"]] - groupedTeamStats.loc[row["T2_id"]]
        X.loc[match_id,xCols] = diffs.values[1:]
#         roundPlayed = matchupSlots.loc[((matchupSlots.strongteam == row["t1_id"]) & (matchupSlots.weakteam == row["t2_id"])) | ((matchupSlots.weakteam == row["t1_id"]) & (matchupSlots.strongteam == row["t2_id"]))  ]
#         roundPlayed = matchupSlots.loc[(matchupSlots.t1t2 == (row["t1_id"]+row["t2_id"])) | (matchupSlots.t1t2 == (row["t2_id"]+row["t1_id"]))]
 
#         if roundPlayed.shape[0]!=1:
#             raise Exception ("oops")
#         X.loc[match_id,"seedDiff"] = roundPlayed.round.values[0]
            
        
    # use raw seed difference
    X["SeedDiff"] = matchups_in.SeedDiff


    # adjust seed difference by the round the matchup occurs in
    
    pool = multiprocessing.Pool(4)
    mapResults = pool.map(doF, itertools.izip(matchups_in.iterrows(), itertools.repeat(matchupSlots)),100)
#     results = [pool.apply_async(doF, (match_id, row)) for match_id, row in matchups_in.iterrows()]
     
    X["RoundPlayed"] = mapResults
    # ignore seed diff
    # X["SeedDiff"] = X.SeedDiff / np.sqrt(X.RoundPlayed)
    X.drop("RoundPlayed", axis=1, inplace=True)
    
    X["PriorMatchupMargin"] = priorMatchupMeanMargin
    
    # turnovers not significant?
    X = X.drop("AvgToDiff",1)
#    X = X.drop("seedDiff",1)
#    X["reboundDiff"] = X.avgORDiff - X.avgDRDiff
    
    # need to scale X??
    x_insample = X.loc[tourn_in.index]
    x_outsample = X.loc[tourn_out.index]
    x_fullsample = X.loc[tourn.index]

    # statsmodels
    

    if False:
    
    
        logitModel = sm.Logit(y, x_insample)
        res = logitModel.fit()
        print res.summary()
        
        yPred = logitModel.predict(res.params, x_insample)
        print "logloss = " + str(logloss(y, yPred))
        
        actualTournPred = pd.Series(yPred, index=x_insample.index)
        fullMatchupPred = pd.DataFrame(logitModel.predict(res.params, X), index=X.index, columns=["prob"])
        
        fullMatchupPred["Season"] = matchups.Season.map(int)
        
        
        
        return (fullMatchupPred, 
                actualTournPred) 
        
    
    
    
    #sklearn
    if False:
     
        penalty = "l2" # l1 or l2
        dual=False
        #tol=0.0001
        #C=1.0 # strength of regularization (smaller = stronger)
        fit_intercept=True # false = no bias?
        regress = LogisticRegressionCV(penalty=penalty,
                                     dual=dual,
                                     fit_intercept=fit_intercept)
         
        regress.fit(X_actualTourn, y)
        
        yPred = regress.predict_proba(X_actualTourn)[:,1]
        actualTournPred  = pd.Series(yPred, index=X_actualTourn.index)
        fullMatchupPred = pd.DataFrame(regress.predict_proba(X)[:,1], index=X.index, columns=["prob"])
        fullMatchupPred["Season"] = matchups.Season.map(int)
    
        return(fullMatchupPred, actualTournPred)
    
    if False:
        
        param_grid={"n_estimators" : [10,40,100,200],
                    "min_samples_leaf" : [1,3,5,10],
                    "max_features" : ["auto",3,7,15],
                    }
        est = RandomForestClassifier(oob_score=False)
        grid = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs=-1, cv=5, verbose=10)
        grid.fit(x_insample, y_insample)

        print grid.best_params_
        print grid.best_score_
        from IPython.core.debugger import Tracer;Tracer()()
        # best max_feat = 7, n_estimators = 200, min_samp_leaf = 10


    #sklearn RFC
    if False:
        n_estimators=200
        n_jobs=-1
        oob_score = True
        min_samples_leaf = 10
        max_features = 7
        regress = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs, oob_score=oob_score, min_samples_leaf=min_samples_leaf)
    
    # in actual stage 2, fit using the whole dataset    
#         fit_X = X_actualTourn.loc[tourn_in.Season < 2013]
#         fit_Y = tourn_in.loc[tourn_in.Season<2013, "Result"]
#         regress.fit(fit_X, fit_Y)
        regress.fit(x_insample, y_insample)
#         from IPython.core.debugger import Tracer;Tracer()()

        yPred = regress.predict_proba(x_fullsample)[:, 1]
        actualTournPred  = pd.Series(yPred, index=x_fullsample.index)
        fullMatchupPred = pd.DataFrame(regress.predict_proba(X)[:, 1], index=X.index, columns=["prob"])
        fullMatchupPred["Season"] = matchups.Season.map(int)
    
        return fullMatchupPred, actualTournPred
    # return the

    if True:
        # ensenble
        rf = RandomForestClassifier(n_estimators=200, max_features=7, min_samples_leaf=10, oob_score=False, n_jobs=-1)
        lr = LogisticRegressionCV()
        ada = AdaBoostClassifier(n_estimators=200)
        knn = KNeighborsClassifier(n_neighbors=20, weights="distance")
        xgb = XGBClassifier(n_estimators=200)

        classifiers = [["rf", rf],
                        ["lr", lr],
                        ["ada", ada],
                       ["knn", knn],
                       ["xgb", xgb]
                       ]

        ensemble = VotingClassifier(estimators=classifiers, voting="soft", weights = [1.0]*len(classifiers), n_jobs=-1)

        ensemble.fit(x_insample, y_insample)

        votedPredProb = ensemble.predict_proba(x_outsample)
        votedPred = votedPredProb[:, 1] > votedPredProb[:, 0]

        for clasname, clas in zip(classifiers, ensemble.estimators_):
            print "{0} logloss: {1}".format(clasname[0], log_loss(y_out, clas.predict_proba(x_outsample)))
            print "{0} accuracy: {1}".format(clasname[0], accuracy_score(y_out, clas.predict(x_outsample)))

        print "{0} logloss: {1}".format("ensemble", log_loss(y_out, votedPredProb))
        print "{0} accuracy: {1}".format("ensemble", accuracy_score(y_out, votedPred))

        collectProbas = ensemble._collect_probas(x_outsample)

        def loglossfunc(weights):
            return log_loss(y_out, np.average(collectProbas, axis=0, weights=weights))

        x0 = [0.5] * len(classifiers)
        constraints = ({"type": "eq",
                        "fun": lambda w: 1 - sum(w)})
        bounds = [(0, 1)] * len(classifiers)

        res = minimize(loglossfunc, x0, method="SLSQP", bounds=bounds, constraints=constraints)

        print "Ensemble score: {0}".format(res['fun'])
        print "Best weights: {0}".format(res['x'])


        # ensemble.set_params(weights=x0) # equal weight
        ensemble.set_params(weights=res['x'])

        votedPredProbNew = ensemble.predict_proba(x_outsample)
        votedPredNew = votedPredProbNew[:, 1] > votedPredProbNew[:, 0]

        print "optimised ensemble logloss: {0}".format(log_loss(y_out, votedPredProbNew))
        print "optimised ensemble accuracy: {0}".format(accuracy_score(y_out, votedPredNew))

        yPred = ensemble.predict_proba(x_fullsample)[:, 1]
        actualTournPred = pd.Series(yPred, index=x_fullsample.index)
        fullMatchupPred = pd.DataFrame(ensemble.predict_proba(X)[:, 1], index=X.index, columns=["prob"])
        fullMatchupPred["Season"] = matchups.Season.map(int)

        return fullMatchupPred, actualTournPred

        #
#     # predict_proba - probability prediction
#     # predict - classifier prediction (win/loss)
#     # classification_report -
#     # transform - sparsify the X matrix
#     
#     # coefficient SE
#     se = np.sqrt(X_actualTourn.cov().values.diagonal())
#     zVals = regress.coef_ / se
#     waldScores = np.square(zVals)
#     
#     
#     # sumsquarederror
#     sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
#      
#     
#     print classification_report(y, regress.predict(X_actualTourn))#, ["t1 win","t2 win"])#, target_names)
#     
#     print confusion_matrix(y, regress.predict(X_actualTourn))
#     print regress.score(X_actualTourn, y)
#     
#     full_probab = pd.DataFrame(regress.predict_proba(X)[:,1], index=X.index, columns=["prob"])
#     full_probab["season"] = matchups.season
#     
    
    return (full_probab, 
            pd.Series(regress.predict_proba(X_actualTourn)[:,1], index=X_actualTourn.index)) 
    
    
