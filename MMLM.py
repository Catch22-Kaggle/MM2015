import pandas as pd
import numpy as np
from matplotlib import pyplot
from itertools import combinations
from evaluate import evaluate
from model1 import model1

reg_det = pd.read_csv("RegularSeasonDetailedResults.csv")
# reg_det_2017 = pd.read_csv("RegularSeasonDetailedResults_2017.csv")
# reg_det = reg_det.append(reg_det_2017)
tourn_det = pd.read_csv("TourneyDetailedResults.csv")

seeds = pd.read_csv("TourneySeeds.csv")
# seeds2017 = pd.read_csv("TourneySeeds_2017.csv")
# seeds = seeds.append(seeds2017)
# parse this a bit more
seeds["Region"] = seeds.Seed.apply(lambda x: x[0])
seeds["RegionSeed"] = seeds.Seed.apply(lambda x: int(x[1:3]))
seeds.index = seeds.Season.map(str)+"_"+seeds.Team.map(str)

slots = pd.read_csv("TourneySlots.csv")
# slots2017 = pd.read_csv("TourneySlots_2017.csv")
# slots = slots.append(slots2017)
slots["Round"] = slots.Slot.str[1].map(int)


slots["Seed"] = np.nan

######################## matchup slots #################
# here we want to figure out which round every matchup would be played in
# dataframe with one seed column, repeatedly joined on itself?


all_slots = pd.DataFrame(columns=["Season","Slot","Strongseed","Weakseed","Round","Seed"])
for year in range(2003,2018):
    print "calculating slots for " + str(year)
    
    yearSlots = slots.loc[slots.Season==year]
    
    yearSlotOut = pd.DataFrame(columns=["Season","Slot","Strongseed","Weakseed","Round","Seed"])
    currRoundSlots = yearSlots[yearSlots.Round==1]
    
    
    slotData = slots
    slotData["Seed"] = slots.Strongseed
    slots2 = slots.copy(deep=True)
    slots2["Seed"] = slots2.Weakseed
    slotData = slotData.append(slots2)
    
    # process pre-round possibilities
    regions = ['X','W','Y','Z']
    #regions = '|'.join(['X','W','Y','Z'])
    preRound = currRoundSlots.loc[currRoundSlots.Slot.map(len) == 3].copy()
    preRound["Seed"] = preRound.Strongseed
    preRound2 = preRound.copy(deep=True)
    preRound2["Seed"] = preRound2.Weakseed
    preRound = preRound.append(preRound2)
    preRound = preRound[["Season","Slot","Seed"]]
    preRound.drop_duplicates(subset=["Season","Slot","Seed"], keep="last", inplace=True)
    
    # strong seed combinations
    strongSeedMerge = pd.merge(currRoundSlots, preRound, how='inner', left_on="Strongseed", right_on="Slot")
    strongSeedMerge = strongSeedMerge[["Season_x","Slot_x","Seed_y","Weakseed","Round","Seed_x"]]
    strongSeedMerge.columns = ["Season","Slot","Strongseed","Weakseed","Round","Seed"]
    
    # and weak seed combinations
    weakSeedMerge = pd.merge(currRoundSlots, preRound, how='inner', left_on="Weakseed", right_on="Slot")
    weakSeedMerge = weakSeedMerge[["Season_x","Slot_x","Strongseed","Seed_y","Round","Seed_x"]]
    weakSeedMerge.columns = ["Season","Slot","Strongseed","Weakseed","Round","Seed"]
    
    currRoundSlots = currRoundSlots.append(strongSeedMerge)
    currRoundSlots = currRoundSlots.append(weakSeedMerge)
    currRoundSlots = currRoundSlots.loc[(~currRoundSlots.Strongseed.isin(preRound.Slot.values)) & (~currRoundSlots.Weakseed.isin(preRound.Slot.values))]
    # add first round matchups
    yearSlotOut = yearSlotOut.append(currRoundSlots)
#     from IPython.core.debugger import Tracer
#     Tracer()()
    
    
    for roundNum in range(2,7): # 6 rounds

        prevRoundTeams = currRoundSlots
        currRoundSlots = yearSlots[yearSlots.Round==roundNum]
        
        prevRoundTeams["Seed"] = prevRoundTeams.Strongseed
        prevRoundTeams2 = prevRoundTeams.copy(deep=True)
        prevRoundTeams2["Seed"] = prevRoundTeams2.Weakseed
        prevRoundTeams = prevRoundTeams.append(prevRoundTeams2)
        prevRoundTeams = prevRoundTeams[["Season","Slot","Seed"]]
        prevRoundTeams.drop_duplicates(subset=["Season","Slot","Seed"], keep="last", inplace=True)
        
        
        # strong seed combinations
        strongSeedMerge = pd.merge(currRoundSlots, prevRoundTeams, how='left', left_on="Strongseed", right_on="Slot")
        strongSeedMerge = strongSeedMerge[["Season_x","Slot_x","Seed_y","Weakseed","Round","Seed_x"]]
        strongSeedMerge.columns = ["Season","Slot","Strongseed","Weakseed","Round","Seed"]
        
        # and weak seed combinations
        bothSeedMerge = pd.merge(strongSeedMerge, prevRoundTeams, how='left', left_on="Weakseed", right_on="Slot")
        bothSeedMerge = bothSeedMerge[["Season_x","Slot_x","Strongseed","Seed_y","Round","Seed_x"]]
        bothSeedMerge.columns = ["Season","Slot","Strongseed","Weakseed","Round","Seed"]
        
        # append to yearly matchups
        currRoundSlots=bothSeedMerge
        yearSlotOut = yearSlotOut.append(currRoundSlots)
        
#     from IPython.core.debugger import Tracer
#     Tracer()()
    all_slots = all_slots.append(yearSlotOut)

all_slots = all_slots.drop("Seed")
# now replace with team ids
all_slots["Strongseed"] = all_slots.Season.map(int).map(str) + "_" + all_slots.Strongseed.map(str)
all_slots["Weakseed"] = all_slots.Season.map(int).map(str) + "_" + all_slots.Weakseed.map(str)

seeds["Seasonseed"] = seeds.Season.map(int).map(str) + "_" + seeds.Seed.map(str)


strongs = pd.merge(all_slots,seeds, "left",left_on="Strongseed",right_on="Seasonseed")
strongs = strongs[["Season_x","Slot","Strongseed","Weakseed","Round","Team"]]
strongs.columns = ["Season","Slot","Strongseed","Weakseed","Round","Strongteam"]
matchupSlots = pd.merge(strongs,seeds, "left",left_on="Weakseed",right_on="Seasonseed")
matchupSlots = matchupSlots[["Season_x","Slot","Strongseed","Weakseed","Round","Strongteam","Team"]]
matchupSlots.columns = ["Season","Slot","Strongseed","Weakseed","Round","Strongteam", "Weakteam"]

matchupSlots["Strongteam"] = matchupSlots.Season.map(int).map(str) + "_" + matchupSlots.Strongteam.map(str)
matchupSlots["Weakteam"] = matchupSlots.Season.map(int).map(str) + "_" + matchupSlots.Weakteam.map(str)


#####################################

# create indices that are t1_t2, where t1 id < t2 id
reg_ix = reg_det.Season.astype(str) + "_" + reg_det[["Wteam","Lteam"]].min(axis=1).astype(str) + "_" + reg_det[["Wteam","Lteam"]].max(axis=1).astype(str)
reg_det.index = reg_ix
tourn_ix = tourn_det.Season.astype(str) + "_" + tourn_det[["Wteam","Lteam"]].min(axis=1).astype(str) + "_" + tourn_det[["Wteam","Lteam"]].max(axis=1).astype(str)
tourn_det.index = tourn_ix



# create a result column in tourn, representing the 'actual' result (ie. did the team 
# with the lower ID win?
reg_det["Result"] = 1.0 * (reg_det["Wteam"] < reg_det["Lteam"])
reg_det["Wmargin"] = (reg_det["Result"] * 2 - 1) # 1 if winner is t1, -1 otherwise
reg_det["Wmargin"] = reg_det["Wmargin"] * (reg_det["Wscore"] - reg_det["Lscore"]) # winscore - losescore * 1 if winner is t1 - ie. t1score - t2score
tourn_det["Result"] = 1.0 * (tourn_det["Wteam"] < tourn_det["Lteam"])
tourn_det["T1"] = tourn_det[["Wteam","Lteam"]].min(axis=1)
tourn_det["T2"] = tourn_det[["Wteam","Lteam"]].max(axis=1)

predictions = pd.DataFrame(columns=["t1","t2"])
print "generating all possible matchups for each year"
all_matchups = None
for year in range(2003,2018):
#for year in range(2011,2012):
#     year_tourn = tourn_det[tourn_det.season == year]
#     year_reg = reg_det[reg_det.season == year]
    year_seeds = seeds.loc[seeds.Season==year]
    
    # we're only interested in teams that made the tourney for this year...
    year_teams = year_seeds["Team"] #pd.Series(year_seeds.loc[:,["wteam","lteam"]].values.ravel()).unique()
    year_teams = year_teams.copy()
    year_teams.sort_values(inplace=True)

    # all 2-team combinations of the team list...
    matchups = combinations(year_teams, 2)
    
    matchup_combos = np.array([[str(year) + "_" + str(t1) + "_" + str(t2), year, t1, t2] for t1,t2 in matchups])
    if all_matchups is None:
        all_matchups = matchup_combos
    else:
        all_matchups = np.vstack((all_matchups, matchup_combos))

allMatchups = pd.DataFrame(all_matchups[:,1:], index=all_matchups[:,0], columns=["Season","T1","T2"])
allMatchups["T1_id"] = allMatchups["Season"].map(str) + "_" + allMatchups["T1"]
allMatchups["T2_id"] = allMatchups["Season"].map(str) + "_" + allMatchups["T2"]



print "generating seeded benchmark"
seededBenchmark = pd.Series(index=allMatchups.index)
seedDiff = pd.Series(index=allMatchups.index)
for ix, row in allMatchups.iterrows():
    year=ix.split("_")[0]
    t1seed = seeds.loc[year+"_"+row["T1"]]["RegionSeed"]
    t2seed = seeds.loc[year+"_"+row["T2"]]["RegionSeed"]
    seedDiff[ix] = t1seed - t2seed
    predict = 0.5 + (t2seed - t1seed)*0.03
    #if t1seed > t2seed:
    #    predict = 0.0
    #elif t1seed < t2seed:
    #    predict = 1.0
    seededBenchmark[ix] = predict


allMatchups["SeedDiff"] = seedDiff

priorMatches = reg_det.ix[allMatchups.index]
priorMatchesGrouped = priorMatches.groupby(priorMatches.index) #, axis, level, as_index, sort, group_keys, squeeze)
# take the mean of each
priorMatchesMeanWins = priorMatchesGrouped.Result.mean() 
priorMatchesMeanMargin = priorMatchesGrouped.Wmargin.mean()
# priorMatches = pd.Series([reg_det.loc[reg_det.index == matchIx].result.mean() for matchIx in allMatchups.index], index=allMatchups.index)
priorMatchesMeanWins.fillna(0.5, inplace=True)



# regression model!
print "generating logistic regression model"
model1_fullPred, model1_pred = model1(tourn_det, reg_det, allMatchups, priorMatchesMeanMargin, matchupSlots)
        
    # we have predictions now.
    # so line them up with actual results and evaluate them

print "evaluating model"
correct = 0
guessed = 0
tourney_predictions = pd.DataFrame(tourn_det[["Result","Season"]],index=tourn_det.index)

tourney_predictions["Season"] = tourn_det.Season
tourney_predictions["Prediction"] = 0.5
print "no prediction"
evaluate(tourney_predictions)

print "seeded prediction"
tourney_predictions["Prediction"] = seededBenchmark

evaluate(tourney_predictions)

print "priorMatchup prediction"
tourney_predictions["Prediction"] = priorMatchesMeanWins

evaluate(tourney_predictions)

print "model1 prediction"
tourney_predictions["Prediction"] = model1_pred

evaluate(tourney_predictions)

# output 2011-2014 predictions for stage 1 results

from IPython.core.debugger import Tracer
Tracer()()
stage1_predictions = model1_fullPred.loc[model1_fullPred.Season >= 2013]["prob"]

stage1_predictions.to_csv("stage1_1.csv", header=["pred"], index_label="id")

stage2_predictions = model1_fullPred.loc[model1_fullPred.Season == 2017]["prob"]

stage2_predictions.to_csv("stage2_1.csv", header=["pred"], index_label="id")


    
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


