import pandas as pd
import numpy as np
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics.metrics import classification_report, confusion_matrix
from evaluate import logloss
import statsmodels.api as sm

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
     
    

def model1(tourn, reg, matchups, priorMatchupMeanMargin):
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
    
    tourn["t1_id"] = tourn.season.map(str) + "_" + tourn.t1.map(str) 
    tourn["t2_id"] = tourn.season.map(str) + "_" + tourn.t2.map(str)
    
    
    #re-organise regualr season details by team rather than match
    reg["wteam_id"] = reg.season.map(str) + "_" + reg.wteam.map(str) 
    reg["lteam_id"] = reg.season.map(str) + "_" + reg.lteam.map(str)
    reg["winningMargin"] = reg.wscore - reg.lscore
    
    priorMatchupMeanMargin.fillna(0.0, inplace=True)
    
    
    teams = pd.concat([reg.wteam_id, reg.lteam_id]).unique()
    teams.sort()
    
    teamStats = pd.DataFrame(index=reg.index)
    statCols = ["team_id","daynum","winningMargin","fgm","fga","fgm3","fga3","ftm","fta","or","dr","ast","to","stl","blk","pf"]
    winCols = ["wteam_id","daynum","winningMargin","wfgm","wfga","wfgm3","wfga3","wftm","wfta","wor","wdr","wast","wto","wstl","wblk","wpf"]
    loseCols = ["lteam_id","daynum","winningMargin","lfgm","lfga","lfgm3","lfga3","lftm","lfta","lor","ldr","last","lto","lstl","lblk","lpf"]
    teamStats = reg[winCols]
    loseStats = reg[loseCols]
    loseStats.winningMargin *= -1
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
    tourn_in = tourn#.loc[tourn.season<2013]
    tourn_out = tourn#.loc[tourn.season>=2013]
    matchups_in = matchups#.loc[matchups.season<2013]
    
    y = tourn_in.result
    X = pd.DataFrame(index=matchups_in.index)
    xCols = ["avgWinMargDiff",
             "avgFGMDiff",
             "avgFGADiff",
             "avgFGPctDiff",
             "avgFGM3Diff",
             "avgFGA3Diff",
             "avgFG3PctDiff",
             "avgFTMDiff",
             "avgFTADiff",
             "avgFTPctDiff",
             "avgORDiff",
             "avgDRDiff",
             "avgAstDiff",
             "avgToDiff",
             "avgStlDiff",
             "avgBlDiff",
             "avgPFDiff",
             ]
    for col in xCols:
        X[col] = np.nan
    
    X["seedDiff"] = np.nan

    teamStats.insert(5, "fgpct", teamStats.fgm/teamStats.fga)
    teamStats.insert(8, "fg3pct", teamStats.fgm3/teamStats.fga3)
    teamStats.insert(11, "ftpct", teamStats.ftm/teamStats.fta)
    
    # handle any infinit percentages
    teamStats.replace(np.inf,np.nan, inplace=True)
    
    
    ewma = pd.stats.moments.ewma
    
    def expwMean(series):
        
        expp = expWeight(series.shape[0]).reshape((-1,1))
        series.sort("daynum",inplace=True)
        res = pd.Series(np.sum(series.values[:,1:]*expp, axis=0),index=series.columns[1:])
        if np.isnan(res.ftpct):
            res.ftpct = series.ftpct.mean()
        if np.isnan(res.fgpct):
            res.fgpct = series.fgpct.mean()
        if np.isnan(res.fg3pct):
            res.fg3pct = series.fg3pct.mean()
        #series.values[-1,0]
        return res
        #return np.sum(series.values[:,1:]*expp, axis=0)
    
    
    grouped = teamStats.groupby(teamStats.team_id)
    groupedTeamStats = grouped.agg(expwMean) # skips nans by default
    
    # equally weighted average
#     groupedTeamStats = teamStats.groupby(teamStats.team_id).mean() # skips nans by default
    
    
    for match_id, row in matchups_in.iterrows():
        
#         t1Matches = teamStats.loc[teamStats.team_id == row["t1_id"]]
#         t2Matches = teamStats.loc[teamStats.team_id == row["t2_id"]]
#         diffs = t1Matches.mean() - t2Matches.mean()
        diffs = groupedTeamStats.loc[row["t1_id"]] - groupedTeamStats.loc[row["t2_id"]]
        X.loc[match_id,xCols] = diffs.values[2:]
        
    X["seedDiff"] = matchups_in.seedDiff
    
#     from IPython.core.debugger import Tracer
#     Tracer()()
    X["priorMatchupMargin"] = priorMatchupMeanMargin
    
    # turnovers not significant?
    X = X.drop("avgToDiff",1)
    
    # need to scale X??
    X_actualTourn = X.loc[tourn_in.index]
    
    # statsmodels
    
    from IPython.core.debugger import Tracer
    Tracer()()     
    
    logitModel = sm.Logit(y, X_actualTourn)
    res = logitModel.fit()
    print res.summary()
    
    yPred = logitModel.predict(res.params, X_actualTourn)
    print "logloss = " + str(logloss(y, yPred))
    
    actualTournPred = pd.Series(yPred, index=X_actualTourn.index)
    fullMatchupPred = pd.DataFrame(logitModel.predict(res.params, X), index=X.index, columns=["prob"])
    
    fullMatchupPred["season"] = matchups.season.map(int)
    
    
    
    return (fullMatchupPred, 
            actualTournPred) 
    
    
    
    
    #sklearn
     
     
    penalty = "l2" # l1 or l2
    dual=False
    tol=0.0001
    C=1.0 # strength of regularization (smaller = stronger)
    fit_intercept=False # no bias?
    intercept_scaling=1
    class_weight=None
    random_state=None
    regress = LogisticRegression(penalty,
                                 dual,
                                 tol,
                                 C,
                                 fit_intercept,
                                 intercept_scaling,
                                 class_weight,
                                 random_state)
     
     
    from IPython.core.debugger import Tracer
    Tracer()()
     
    regress.fit(X_actualTourn, y)
    
    
    
    
    # return the 
    
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
    
    