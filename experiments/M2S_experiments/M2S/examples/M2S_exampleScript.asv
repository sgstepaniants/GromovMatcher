%% M2S_exampleScript: M2S example for matching two untargeted LCMS datasets
%
% This script contains brief explanations on each step of the M2S toolbox
% to match metabolomics features of two untargeted datasets using RT/MZ/FI.
% It contains data to run two examples, all possibilities of inputs, and 
% lines of code to run the method on default or user-defined settings.
% *** To run code and figures select the desired line(s) and click F9 ***
% Save this file with another name and modify it to run your own projects.
%
% M2S toolbox to match LCMS metabolomics features of untargeted datasets.
% *** Rui Climaco Pinto ***
% *** Imperial College London, 2021 ***


%% Load reference and target features of sets to match 
% These are two data files containing 3 columns without headers, only values for each feature, in this order: RT, MZ, FI. 
% NOTE: RT = retention time median;  MZ = mass-to-charge (m/z) ratio median; FI = Feature Intensity median
% RT should be in minutes. If it is in seconds, divide it by 60;
% If there is no feature intensity use a column with ones or random values.

%% Load the data

addpath("..")

datasetName = "LPOS"; % {"LNEG","LPOS"}

% NOTE: there are two Plasma Lipid examples (ESI- and ESI+), each with two
% datasets that can be matched


if datasetName == "LNEG"
    refFilename = 'test_refFeatures_LNEG_csv.csv';
    targetFilename = 'test_targetFeatures_LNEG_csv.csv';
elseif datasetName == "LPOS"
    refFilename = 'test_refFeatures_LPOS_csv.csv';
    targetFilename = 'test_targetFeatures_LPOS_csv.csv';
end

% Function 'importdata.m' loads data from .csv, .txt, or .xlsx 
% (You may need to change directory to the file location)
[refFeatures] = importdata(refFilename);
[targetFeatures] = importdata(targetFilename);

% Create labels for all features

[refMZRT_str] = M2S_createLabelMZRT('ref',refFeatures(:,2),refFeatures(:,1));
[targetMZRT_str] = M2S_createLabelMZRT('target',targetFeatures(:,2),targetFeatures(:,1));

% Visualize the two feature sets
M2S_figureH(0.8,0.5)
subplot(1,2,1), 
M2S_plotMZRT_featureSet(refFeatures,1,8,1); title('Reference featureSet')
subplot(1,2,2), 
M2S_plotMZRT_featureSet(targetFeatures,1,8,1); title('Target featureSet')

%%*************************************************************************
%% Procedure part 1: find all possible matches
%%*************************************************************************

% create a structure to keep the options chosen at each step
opt = struct;

%% Set thresholds for matching all features 
% Try running it with default settings as below, to see the major trends.
% It may become very crowded.
% Default settings: no RTthresh ; MZthresh < 0.02 (m/z); log10FIthresh < 1000
% Later each setting can be set as desired.
[refSet,targetSet,Xr_connIdx,Xt_connIdx, opt]=M2S_matchAll(refFeatures,targetFeatures);

% Additional plot of results:
Xsymbol = '.k';
plotRow = 2; % y-axis is e.g. RTdist (1) or both RTdist and RT_target (2)
M2S_plotDelta_matchedSets(refSet,targetSet,Xsymbol,plotRow)


% NOTES on setting thresholds:
% Thresholds define the maximum acceptable inter-dataset shift for each dimension (RT, MZ, log10FI). 
% You can define "horizontal line" fixed thresholds by using only "intercept" values, and setting "slope" to zero.
% These are equivalent to defining a fixed m/z.
% You can define "diagonal line" relative thresholds by using also "slope" values.
% These are equivalent to defining a relative m/z (kind of a ppm threshold value)
% HINT: the function below helps to calculate an intercept and slope in one of the dimensions:
%[RTMZFI_intercept,RTMZFI_slope] = M2S_calculateInterceptSlope() % to define it with the mouse on a figure
%[RTMZFI_intercept,RTMZFI_slope] = M2S_calculateInterceptSlope(startPoint,endPoint) % to insert two points manually

%% M2S_matchAll SETTINGS for the two main examples:

if datasetName == "LNEG"
    % Example settings for LNEG test data
    opt.FIadjustMethod = 'regression'; % {'none','median','regression'}
    opt.multThresh.RT_intercept = [-0.55,0.15];
    opt.multThresh.RT_slope = [0 0];
    opt.multThresh.MZ_intercept = [-0.01 0.01]; % m/z units
    opt.multThresh.MZ_slope = [-5e-6 5e-6]; % ppm
    opt.multThresh.log10FI_intercept = [-1 1.5];
    opt.multThresh.log10FI_slope = [0 1];
    
elseif datasetName == "LPOS"
    % Example settings for LPOS test data
    opt.FIadjustMethod = 'median'; % {'median','regression'}
    opt.multThresh.RT_intercept = [-0.1,0.1];
    opt.multThresh.RT_slope = [0 0];
    opt.multThresh.MZ_intercept = [-0.0075 0.015];
    opt.multThresh.MZ_slope = [-0.01/2000, -0.01/2000];
    opt.multThresh.log10FI_intercept = [-1.4 0.75];
    opt.multThresh.log10FI_slope = [0 0];    
    
end


% Match all features within defined thresholds

% Define the plot type as:
% No plot: plotType = 0
% Scatter plots: plotType = 1
% Multiple match plots: plotType = 2 (with lines connecting clusters of multiple
% matches containing the same feature). Also plots a network of all matches.

plotType = 2; 
[refSet,targetSet,Xr_connIdx,Xt_connIdx,opt]=M2S_matchAll(refFeatures,targetFeatures,opt.multThresh,opt.FIadjustMethod,plotType);

% OPTIONAL: Obtain properties of current network of matches, with focus
% in one of the dimensions
focusDim = 1 % 1=RT; 2=MZ; 
[G1,CC] = M2S_infoClusters(refSet,targetSet,focusDim,':');

%%*************************************************************************
%% Procedure part 2: Calculate penalisation scores for each match
%%*************************************************************************

%% Find neighbours, the RT/MZ/log10FI trends, and the residuals - SETTINGS OPTIONAL
% - Find the inter-dataset shifts in RT/MZ/log10FI, and  use it to 
% calculate the residuals (distance to those shift curves).
% - The method to find RT/MZ/FI shift trends uses subsets of features closest to each feature (the "neighbors").
% For a reference feature, its shift is the median shift of its neighbours.
% - The number of neighbors is not critical and can be defined by a percentage of the total of
% features in the refSet (e.g. 0.05) or by a specific number (e.g. 25). The
% less neighbours the more detailed is the modeling (less linear).
% - The neighbor method to calculate trends can be:
%       - "cross", finds neighbours of a feature in RT, MZ, FI individually.
%       Different neighbours are found in each of these dimensions.
%       -(default) "circle", finds neighbours by Euclidean distance using
%       simultaneously RT and MZ (but not FI). "Circle" does not subtract the 
%       trend from FI, only yelds the log10FI difference between target and ref.
% - The shifts can be individual points for each ref feature, or additionally
% smoothed using a robust loess 
%
% SETTINGS POSSIBILITIES
% opt.neighbours.nrNeighbors = 0.01; %as a percentage of the number of features in refSet  
% opt.neighbours.nrNeighbors = 21; % as a specific number of neighbours
% opt.calculateResiduals.neighMethod = 'circle'; %{'cross','circle'}
% opt.pctPointsLoess = 0.1; options are 0 (no loess) or ]0,1], use for the 
% loess a percentage of the total number of points (size(refSet,1)).
% plotTypeResiduals = 1 % Options are 0 (no plot), 1 (default), 2 (extra)

if datasetName == "LNEG"
    opt.neighbours.nrNeighbors = 21;
    opt.calculateResiduals.neighMethod = 'circle';
    opt.pctPointsLoess = 0;% no loess
    plotTypeResiduals = 1
    [Residuals_X,Residuals_trendline] = M2S_calculateResiduals(refSet,targetSet,Xr_connIdx,Xt_connIdx,opt.neighbours.nrNeighbors, opt.calculateResiduals.neighMethod,opt.pctPointsLoess,plotTypeResiduals)
elseif datasetName == "LPOS"
    opt.neighbours.nrNeighbors = 0.01;
    opt.calculateResiduals.neighMethod = 'cross';
    opt.pctPointsLoess = 0.1;% loess with 10% of points
    plotTypeResiduals = 1
    [Residuals_X,Residuals_trendline] = M2S_calculateResiduals(refSet,targetSet,Xr_connIdx,Xt_connIdx,opt.neighbours.nrNeighbors, opt.calculateResiduals.neighMethod,opt.pctPointsLoess,plotTypeResiduals)
end


%% Adjust the residuals
% - The residuals are in different units and need to be standardised
% to be combined into penalisation scores for each match.
% This is achieved by dividing the residuals by their median+3*MAD, or by
% a user-defined residual value.
% After this division, the value of the residual on that point is 1, so it
% is easy to visualise the relevance of each of the dimension's residuals (RT/MZ/log10FI).
%
% SETTINGS POSSIBILITIES
% opt.adjustResiduals.residPercentile = NaN 
% Automatic determination (median + 3*MAD) using threshold point method. Different value for each dimension. 
%    
% opt.adjustResiduals.residPercentile = [0.05,0.004,1.2]; 
% Set threshold points using residual values. These selected residual values become = 1 
% (residuals can be seen in previous figure "Residuals for RT and MZ")  


if datasetName == "LNEG"
    opt.adjustResiduals.residPercentile = [0.1,0.01,1.5];
    [adjResiduals_X,residPercentile] = M2S_adjustResiduals(refSet,targetSet,Residuals_X,opt.adjustResiduals.residPercentile);
elseif datasetName == "LPOS"
    [adjResiduals_X,residPercentile] = M2S_adjustResiduals(refSet,targetSet,Residuals_X,NaN);
end


%% Adjust the weight of each dimension (RT, MZ, log10FI), get penalisation scores
% - The penalisation scores for each match are the squared root of the sum 
% of squares of the weighted residuals. Many times it is desired that log10FI has weight 0.
% The highest penalty scores are attributed to the worst matches
%
% opt.weights.W = [1,1,0]; % (Default) equal weight for RT and MZ, not using FI
% opt.weights.W = [2,1,0]; % different weights
% opt.weights.W = [1,1,1]; % equal weight
%
% [penaltyScores] = M2S_defineW_getScores(refSet,targetSet,adjResiduals_X,opt.weights.W,1)
% [penaltyScores] = M2S_defineW_getScores(refSet,targetSet,adjResiduals_X); W = [1,1,0]

if datasetName == "LNEG"
    opt.weights.W = [1,1,1]; % equal weight
    [penaltyScores] = M2S_defineW_getScores(refSet,targetSet,adjResiduals_X,opt.weights.W,1); 
elseif datasetName == "LPOS"
    opt.weights.W = [1,1,0.2]; % use log10FI
    [penaltyScores] = M2S_defineW_getScores(refSet,targetSet,adjResiduals_X,opt.weights.W,1);
end

%% Decide the best of the multiple matches 
% - Recursive method for a multiple-match cluster at a time. In
% each iteration it collects the (best) match with the lowest penalisation 
% score, until there are no more selections to do in that cluster.
% NOTE:you can visualise the decomposition of clusters with multiple matches 
% by giving minNrNodesToPlot a small value, e.g. minNrNodesToPlot = 4.
% plotNetOrNot = 1; % 0/1 plots global network of all matches colored by penalisation scores.
% minNrNodesToPlot = 3
%
% [eL,eL_INFO,CC_G1]= M2S_decideBestMatches(refSet, targetSet, Xr_connIdx,Xt_connIdx, penaltyScores, plotNetOrNot, minNrNodesToPlot)
[eL,eL_INFO,CC_G1]= M2S_decideBestMatches(refSet, targetSet, Xr_connIdx,Xt_connIdx, penaltyScores);

%%*************************************************************************
%% Procedure part 3: find false positives (tighten thresholds)
%%*************************************************************************
% - The thresholds set initially are linear, and may not adapt well to the
% curved inter-dataset shifts. Here we define tighter thresholds adapting to curvature.
% - This will delete the matches with largest residuals than the value of a
% median plus N * the MAD of a quantity (scores, residuals, etc)
%
% - Methods:
% 'none' - does not apply new thresholds
% 'scores' - threshold is set at the scores median + N * MAD(scores)
% 'byBins' - residuals of each dimension divided into regions and thresholds set in each
% 'trend_mad' - thresholds are set around the inter-dataset shift value, for each dimension separately
% 'residuals_mad' - (Default) thresholds are set for each of the residuals separately
% NOTE1: scores method finds poor matches based on the weighted scores across all dimensions, while the
% other methods consider matches as poor even if they are outside threshold on a single dimension
% NOTE2: trend_mad and residuals_mad are calculated differentely and have
% different plots, but yield the same results
% opt.falsePos.methodType = 'scores'; {'none','scores','byBins','trend_mad','residuals_mad'} 
% Example:
% opt.falsePos.nrMad = 5;
% plotOrNot = 1;
% [eL_final, eL_final_INFO] =M2S_findPoorMatches(eL,refSet,targetSet,opt.falsePos.methodType,opt.falsePos.nrMad,plotOrNot);
% Example with default settings - methodType = trend_mad; nrMad = 5:
% [eL_final, eL_final_INFO] = M2S_findPoorMatches(eL,refSet,targetSet); 

if datasetName == "LNEG"
    opt.falsePos.methodType = 'trend_mad'; %{'none','scores','byBins','trend_mad','residuals_mad'} 
    opt.falsePos.nrMad = 5;
    plotOrNot = 1;
    [eL_final, eL_final_INFO] = M2S_findPoorMatches(eL,refSet,targetSet,opt.falsePos.methodType,opt.falsePos.nrMad,plotOrNot);
elseif datasetName == "LPOS"
    opt.falsePos.methodType = 'scores'; %{'none','scores','byBins','trend_mad','residuals_mad'} 
    opt.falsePos.nrMad = 5;
    plotOrNot = 1;
    [eL_final, eL_final_INFO] = M2S_findPoorMatches(eL,refSet,targetSet,opt.falsePos.methodType,opt.falsePos.nrMad,plotOrNot);
end


%% Save the results table with multiple matches
% writetable(eL_final,'M2S_edgeList_final.xlsx','Sheet',1)


% Summary with number of (multiple matches) discarded, false positive and true positive matches
tableOfMatches = array2table([nansum(isnan(eL_final.notFalsePositives)),nansum(eL_final.notFalsePositives==0),nansum(eL_final.notFalsePositives==1)],'VariableNames',{'DiscardedMatches','PoorMatches','PositiveMatches'});

%%*************************************************************************
%% RESULTS
%%*************************************************************************

% Multiple matches not selected, false positive matches and good matches:
disp(tableOfMatches)

% Table with all results : eL_final
disp(eL_final)
disp(eL_final_INFO)
 
% The options used in the methods:
disp(opt)

% THE INDICES THAT MATCH refFeatures TO targetFeatures ARE:

refFeatures_idx = eL_final.Xr_connIdx(eL_final.notFalsePositives == 1);
targetFeatures_idx = eL_final.Xt_connIdx(eL_final.notFalsePositives == 1);


% The final matched datasets are:
refMatched = refFeatures(refFeatures_idx,:);
refMatched_MZRTstr = refMZRT_str(refFeatures_idx);
targetMatched = targetFeatures(targetFeatures_idx,:);
targetMatched_MZRTstr = targetMZRT_str(targetFeatures_idx);

refTable = [table(refMatched_MZRTstr),array2table(refMatched,'VariableNames',{'rtmed','mzmed','fimed'})];
targetTable = [table(targetMatched_MZRTstr),array2table(targetMatched,'VariableNames',{'rtmed','mzmed','fimed'})];

writetable(refTable,'M2S_datasetsMatched.xlsx','Sheet',1)
writetable(targetTable,'M2S_datasetsMatched.xlsx','Sheet',2)

%% TO LOAD MICROSOFT EXCEL .xlsx FILES WITH INITIAL DATA AND CREATE MATCHED DATASETS:
% NOTE: each .xlsx file (one for reference another for target) needs to have sheets called "Data" and "VarInfo"
% Sheet "Data" contains only numerical values. N rows of samples vs P columns of metabolomic features 
% Sheet "VarInfo" contains P+1 rows of metabolomic features info (e.g., RT, mz, feature name)
% plus the column label for each of the columns

filenameRef = 'referenceDataset.xlsx'
filenameTarget = 'targetDataset.xlsx'
[refDatasetMatched,targetDatasetMatched] = M2S_applyMatchingResults(filenameRef,filenameTarget,refFeatures_idx,targetFeatures_idx);


%% ALTERNATIVELY:
%% TO REORDER AN INITIAL METABOLITE TABLE ACCORDING TO THESE RESULTS (BOTH TO NUMERICAL VALUES AND FEATURE INFO)
% - After we have matched features, we need to select the corresponding
% data from spreadsheets/text files.
% - The following lines produce MZRTstr which are strings with a unique identifier for each metabolomic feature.
% - They also produce vectors of corresponding indices which allow to match the intensity data of reference and target datasets.
% NOTE: features that did not match anything have the index NaN (empty) and can be deleted, as they did not match.

sorted_refIndices_idx = (1:length(refFeatures_idx))';
orderForData_Ref = NaN(size(refFeatures,1),1);
orderForData_Ref(refFeatures_idx) = sorted_refIndices_idx; 
orderForData_Ref_MZRTstr = M2S_createLabelMZRT('REF',refFeatures(:,2),refFeatures(:,1));
matchedRefData_idx = table(orderForData_Ref,orderForData_Ref_MZRTstr,'VariableNames',{'matchedIdxRef','MZRT_ref'});

sorted_targetIndices_idx = (1:length(targetFeatures_idx))';
orderForData_Target = NaN(size(targetFeatures,1),1);
orderForData_Target(targetFeatures_idx) = sorted_targetIndices_idx;
orderForData_Target_MZRTstr = M2S_createLabelMZRT('TARGET',targetFeatures(:,2),targetFeatures(:,1));
matchedTargetData_idx = table(orderForData_Target,orderForData_Target_MZRTstr,'VariableNames',{'matchedIdxRef','MZRT_target'});

%% Write files with these two vectors:
writetable(matchedRefData_idx,'M2S_matchedRefData_idx.xlsx')
writetable(matchedTargetData_idx,'M2S_matchedTargetData_idx.xlsx')

% To sort two data files with data (REF and TARGET) according to these indices:
% 0. Run the functions above to save two tables of "M2S_matchedData_idx" (ref and target).
% 1. Copy 'M2S_matchedRefData_idx' and 'matchedTargetData_idx' to the datasheets (e.g. Excel, csv, etc)
%    of ref and target you want to match, and paste them along the variables
% 2. Sort the variables in ascending order of the NUMERICAL indices (min to max), so the row order becomes 1, 2, 3, etc. The empty indices will be sent to the end. 
% 3. Delete the data for variables with empty indices (these features do not match anything)
% 4. Check there is the same number of variables in both datasets and these
%    have similar (matched) RT and MZ (and FI?). Check that the MZRT_string matches
%    the retention time and m/z of the corresponding features.
% DONE!


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SUPPORT FUNCTIONS

M2S_figureH(screenWidthPercentage,screenHeightPercentage) 
% Objective: create figure with % of width and height percent of the screen size

M2S_plotDelta_matchedSets(refMatched,targetMatched,Xsymbol)
% Objective: plot results of matched results.

[newOpt] = M2S_addOptions(optOld,additionalOpt)
% Objective: add single options to default options without having to define all option values.
% NOTE: all opts are struct variables. 
% Example: additionalOpt.MZ_intercept = 0.05
% [newOpt] = M2S_addOptions(optOld,additionalOpt)

M2S_colorByY_ofSubplot(subplotNr,figH)
% Objective: Color all subplots in figure by y-values of a defined subplot.
% M2S_colorByY_ofSubplot(2,gcf)% colors subplots in current figure
% according to y-value of subplot number 2
% Optionally, colour all subplots by a the values of a vector colorVec using
% M2S_colorByY_ofSubplot(colorVec,gcf)

[interceptSlope] = M2S_calculateInterceptSlope(startPoint,endPoint)
% Objective: calculate intercept and slope defined by two points.
% Two ways to use:
% 1- By clicking on two points in a plot:
% interceptSlope = M2S_calculateInterceptSlope()
% 2- By inserting the values of the two points ([x1,y1],[x2,y2]) as below:
% interceptSlope = M2S_calculateInterceptSlope([0,0.5],[12,-4.5]);

%% SUPPORT FUNCTIONS FOR OPTIMISATION
% Not final. Work in progress


[refFeatures_noBigClusters,targetFeatures_noBigClusters] = M2S_deleteLargeClusters(refSet,targetSet,maxFeaturesInCluster,opt);
% Objective: Delete large clusters of multiple matches.
% In some cases the matches contain very large clusters, and it would be
% beneficial/easier to not use those to calculate initial thresholds. This
% function can be used after 'M2S_matchAll', to delete clusters with more
% than e.g. maxFeaturesInCluster = 5. 

%% Create matrices 'Residuals_X', 'Residuals_trendline' from reduced matched sets. 
% To use after applying M2S_calculateResiduals to a reduced refSet+targetSet
[Residuals_X,Residuals_trendline] = M2S_interpolateTrendline(reduced_refSet,Reduced_trendline,refSet,targetSet)
% Match feature datasets with tight thresholds, or delete large clusters, 
% get inter-dataset shifts. Then transfer it to dataset with all features.

[genAlg_Res,optBest] = M2S_genAlg_optimisation(refSet, targetSet, RTdist, MZdist, opt, plotOrNot)
% Objective: optimise initial RT, MZ thresholds.
% Use after running the function M2S_matchAll 
% It runs a genetic algorithm for optimisation.


[G1,CC] = M2S_infoClusters(refSet,targetSet,dimNr,line_style)
% Objective: obtain properties of current network of matches.
% dimNr can be RT=1; MZ=2; FI=3
% example: [G1,CC] = M2S_infoClusters(refSet,targetSet,2,'-')
% disp(CC.freq_clustersWithNedges)



