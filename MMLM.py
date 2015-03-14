import pandas as pd
import numpy as np
from matplotlib import pyplot
from itertools import combinations
from evaluate import evaluate
from model1 import model1

reg_det = pd.read_csv("regular_season_detailed_results.csv")
tourn_det = pd.read_csv("tourney_detailed_results.csv")

seeds = pd.read_csv("tourney_seeds.csv")
# parse this a bit more
seeds["region"] = seeds.seed.apply(lambda x: x[0])
seeds["regionSeed"] = seeds.seed.apply(lambda x: int(x[1:3]))
seeds.index = seeds.season.map(str)+"_"+seeds.team.map(str)

slots = pd.read_csv("tourney_slots.csv")
slots["round"] = slots.slot.str[1].map(int)


slots["seed"] = np.nan

######################## matchup slots #################
# here we want to figure out which round every matchup would be played in
# dataframe with one seed column, repeatedly joined on itself?


all_slots = pd.DataFrame(columns=["season","slot","strongseed","weakseed","round","seed"])
for year in range(2003,2015):
    print "calculating slots for " + str(year)
    
    yearSlots = slots.loc[slots.season==year]
    
    yearSlotOut = pd.DataFrame(columns=["season","slot","strongseed","weakseed","round","seed"])
    currRoundSlots = yearSlots[yearSlots.round==1]
    
    
    slotData = slots
    slotData["seed"] = slots.strongseed
    slots2 = slots.copy(deep=True)
    slots2["seed"] = slots2.weakseed
    slotData = slotData.append(slots2)
    
    # process pre-round possibilities
    regions = ['X','W','Y','Z']
    #regions = '|'.join(['X','W','Y','Z'])
    preRound = currRoundSlots.loc[currRoundSlots.slot.map(len) == 3]
    preRound["seed"] = preRound.strongseed
    preRound2 = preRound.copy(deep=True)
    preRound2["seed"] = preRound2.weakseed
    preRound = preRound.append(preRound2)
    preRound = preRound[["season","slot","seed"]]
    preRound.drop_duplicates(cols=["season","slot","seed"], take_last=True, inplace=True)
    
    # strong seed combinations
    strongSeedMerge = pd.merge(currRoundSlots, preRound, how='inner', left_on="strongseed", right_on="slot")
    strongSeedMerge = strongSeedMerge[["season_x","slot_x","seed_y","weakseed","round","seed_x"]]
    strongSeedMerge.columns = ["season","slot","strongseed","weakseed","round","seed"]
    
    # and weak seed combinations
    weakSeedMerge = pd.merge(currRoundSlots, preRound, how='inner', left_on="weakseed", right_on="slot")
    weakSeedMerge = weakSeedMerge[["season_x","slot_x","strongseed","seed_y","round","seed_x"]]
    weakSeedMerge.columns = ["season","slot","strongseed","weakseed","round","seed"]
    
    currRoundSlots = currRoundSlots.append(strongSeedMerge)
    currRoundSlots = currRoundSlots.append(weakSeedMerge)
    currRoundSlots = currRoundSlots.loc[(~currRoundSlots.strongseed.isin(preRound.slot.values)) & (~currRoundSlots.weakseed.isin(preRound.slot.values))]
    # add first round matchups
    yearSlotOut = yearSlotOut.append(currRoundSlots)
    
    
    for roundNum in range(2,7): # 6 rounds

        prevRoundTeams = currRoundSlots
        currRoundSlots = yearSlots[yearSlots.round==roundNum]
        
        prevRoundTeams["seed"] = prevRoundTeams.strongseed
        prevRoundTeams2 = prevRoundTeams.copy(deep=True)
        prevRoundTeams2["seed"] = prevRoundTeams2.weakseed
        prevRoundTeams = prevRoundTeams.append(prevRoundTeams2)
        prevRoundTeams = prevRoundTeams[["season","slot","seed"]]
        prevRoundTeams.drop_duplicates(cols=["season","slot","seed"], take_last=True, inplace=True)
        
        
        # strong seed combinations
        strongSeedMerge = pd.merge(currRoundSlots, prevRoundTeams, how='left', left_on="strongseed", right_on="slot")
        strongSeedMerge = strongSeedMerge[["season_x","slot_x","seed_y","weakseed","round","seed_x"]]
        strongSeedMerge.columns = ["season","slot","strongseed","weakseed","round","seed"]
        
        # and weak seed combinations
        bothSeedMerge = pd.merge(strongSeedMerge, prevRoundTeams, how='left', left_on="weakseed", right_on="slot")
        bothSeedMerge = bothSeedMerge[["season_x","slot_x","strongseed","seed_y","round","seed_x"]]
        bothSeedMerge.columns = ["season","slot","strongseed","weakseed","round","seed"]
        
        # append to yearly matchups
        currRoundSlots=bothSeedMerge
        yearSlotOut = yearSlotOut.append(currRoundSlots)

    all_slots = all_slots.append(yearSlotOut)

# now replace with team ids
all_slots["strongseed"] = all_slots.season.map(str) + "_" + all_slots.strongseed.map(str)
all_slots["weakseed"] = all_slots.season.map(str) + "_" + all_slots.weakseed.map(str)

seeds["seasonseed"] = seeds.season.map(str) + "_" + seeds.seed.map(str)


#####################################

# create indices that are t1_t2, where t1 id < t2 id
reg_ix = reg_det.season.astype(str) + "_" + reg_det[["wteam","lteam"]].min(axis=1).astype(str) + "_" + reg_det[["wteam","lteam"]].max(axis=1).astype(str)
reg_det.index = reg_ix
tourn_ix = tourn_det.season.astype(str) + "_" + tourn_det[["wteam","lteam"]].min(axis=1).astype(str) + "_" + tourn_det[["wteam","lteam"]].max(axis=1).astype(str)
tourn_det.index = tourn_ix



# create a result column in tourn, representing the 'actual' result (ie. did the team 
# with the lower ID win?
reg_det["result"] = 1.0 * (reg_det["wteam"] < reg_det["lteam"])
reg_det["wmargin"] = (reg_det["result"] * 2 - 1) # 1 if winner is t1, -1 otherwise
reg_det["wmargin"] = reg_det["wmargin"] * (reg_det["wscore"] - reg_det["lscore"]) # winscore - losescore * 1 if winner is t1 - ie. t1score - t2score
tourn_det["result"] = 1.0 * (tourn_det["wteam"] < tourn_det["lteam"])
tourn_det["t1"] = tourn_det[["wteam","lteam"]].min(axis=1)
tourn_det["t2"] = tourn_det[["wteam","lteam"]].max(axis=1)

predictions = pd.DataFrame(columns=["t1","t2"])
print "generating all possible matchups for each year"
all_matchups = None
for year in range(2003,2015):
#for year in range(2011,2012):
    year_tourn = tourn_det[tourn_det.season == year]
    year_reg = reg_det[reg_det.season == year]
    
    # we're only interested in teams that made the tourney for this year...
    year_teams = pd.Series(year_tourn.loc[:,["wteam","lteam"]].values.ravel()).unique()
    year_teams.sort()

    # all 2-team combinations of the team list...
    matchups = combinations(year_teams, 2)
    
    matchup_combos = np.array([[str(year) + "_" + str(t1) + "_" + str(t2), year, t1, t2] for t1,t2 in matchups])
    if all_matchups == None:
        all_matchups = matchup_combos
    else:
        all_matchups = np.vstack((all_matchups, matchup_combos))

allMatchups = pd.DataFrame(all_matchups[:,1:], index=all_matchups[:,0], columns=["season","t1","t2"])
allMatchups["t1_id"] = allMatchups["season"].map(str) + "_" + allMatchups["t1"]
allMatchups["t2_id"] = allMatchups["season"].map(str) + "_" + allMatchups["t2"]



print "generating seeded benchmark"
seededBenchmark = pd.Series(index=allMatchups.index)
seedDiff = pd.Series(index=allMatchups.index)
for ix, row in allMatchups.iterrows():
    year=ix.split("_")[0]
    t1seed = seeds.loc[year+"_"+row["t1"]]["regionSeed"]
    t2seed = seeds.loc[year+"_"+row["t2"]]["regionSeed"]
    seedDiff[ix] = t1seed - t2seed
    predict = 0.5 + (t2seed - t1seed)*0.03
    #if t1seed > t2seed:
    #    predict = 0.0
    #elif t1seed < t2seed:
    #    predict = 1.0
    seededBenchmark[ix] = predict

allMatchups["seedDiff"] = seedDiff

priorMatches = reg_det.ix[allMatchups.index]
priorMatchesGrouped = priorMatches.groupby(priorMatches.index) #, axis, level, as_index, sort, group_keys, squeeze)
# take the mean of each
priorMatchesMeanWins = priorMatchesGrouped.result.mean() 
priorMatchesMeanMargin = priorMatchesGrouped.wmargin.mean()
# priorMatches = pd.Series([reg_det.loc[reg_det.index == matchIx].result.mean() for matchIx in allMatchups.index], index=allMatchups.index)
priorMatchesMeanWins.fillna(0.5, inplace=True)

# regression model!
print "generating logistic regression model"
model1_fullPred, model1_pred = model1(tourn_det, reg_det, allMatchups, priorMatchesMeanMargin)

#priorMatches =






#         
#         #from IPython.core.debugger import Tracer
#         #Tracer()()
#     
#     
    
        
    # we have predictions now.
    # so line them up with actual results and evaluate them

print "evaluating model"
correct = 0
guessed = 0
prediction = year_tourn
tourney_predictions = pd.DataFrame(tourn_det[["result","season"]],index=tourn_det.index)

tourney_predictions["season"] = tourn_det.season
tourney_predictions["prediction"] = 0.5
print "no prediction"
evaluate(tourney_predictions)

print "seeded prediction"
tourney_predictions["prediction"] = seededBenchmark

evaluate(tourney_predictions)

print "priorMatchup prediction"
tourney_predictions["prediction"] = priorMatchesMeanWins

evaluate(tourney_predictions)

print "model1 prediction"
tourney_predictions["prediction"] = model1_pred

evaluate(tourney_predictions)

# output 2011-2014 predictions for stage 1 results

from IPython.core.debugger import Tracer
Tracer()()
stage1_predictions = model1_fullPred.loc[model1_fullPred.season >= 2011]["prob"]

stage1_predictions.to_csv("stage1_1.csv", header=["pred"], index_label="id")

    
#         
#         
#         
#             if not np.isnan(matchPred["pred"] or matchPred["pred"] == 0):
#                 guessed += 1
#                 if (matchPred["pred"] > 0 and row["wteam"] == matchPred["t1"] ) or (matchPred["pred"] < 0 and row["lteam"] == matchPred["t1"] ):
#                     correct += 1
#         except KeyError, e:
#             pass
#     
#     
#     print "Year: %s" % (year)
#     print "guessed %d out of %d correct" % (correct, guessed)
#     if guessed > 0:
#         print "win pct = %f" % (float(correct)/guessed)
#                 
# 
#         
#     


