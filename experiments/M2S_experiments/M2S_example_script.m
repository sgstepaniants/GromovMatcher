%% M2S_exampleScript: M2S example for matching two untargeted LCMS datasets

%% Load reference and target features of sets to match 
%% Load the data
% Function 'importdata.m' loads data from .csv, .txt, or .xlsx 

%Doesn't matter which dataset you set as ref or target
%You can try refFilename = 'Data1...csv' and targetFilename = 'Data2...csv'
%Careful though, you need to reformat the data first.

addpath("../M2S")

overlap = 0.5;
noise = 0.1;
drift = "piecewise2"; %"smooth";
all_noised = false;

true_coupling = load(sprintf("../Overlap_%g/true_%g.mat", overlap, overlap)).coupling;
true_matching = (true_coupling > 0);

true_refFeatures_idx = sum(true_matching, 2) > 0;
[~, true_targetFeatures_inds] = max(true_matching(true_refFeatures_idx, :), [], 2);
true_targetFeatures_idx = false(size(true_matching, 2), 1);
true_targetFeatures_idx(true_targetFeatures_inds) = true;

if all_noised
    refFilename = sprintf("../Overlap_%g/Data1_AllNoised_%g_%g_%s_reformatted.csv", overlap, overlap, noise, drift);
    targetFilename = sprintf("../Overlap_%g/Data2_AllNoised_%g_%g_%s_reformatted.csv", overlap, overlap, noise, drift);
else
    refFilename = sprintf("../Overlap_%g/Data1_%g_%g_%s_reformatted.csv", overlap, overlap, noise, drift);
    targetFilename = sprintf("../Overlap_%g/Data2_%g_%g_%s_reformatted.csv", overlap, overlap, noise, drift);
end

% Datasets in order RT, MZ, FI
[refFeatures] = importdata(refFilename);
[targetFeatures] = importdata(targetFilename);

% Create labels for all features
[refMZRT_str] = M2S_createLabelMZRT('ref', refFeatures(:,2), refFeatures(:,1));
[targetMZRT_str] = M2S_createLabelMZRT('target', targetFeatures(:,2), targetFeatures(:,1));

num_refs = length(refMZRT_str);
num_targets = length(targetMZRT_str);


% Visualize the two feature sets
% Not necessary, but you can give it a go
M2S_figureH(0.8, 0.5)
subplot(1,2,1), 
M2S_plotMZRT_featureSet(refFeatures, 1, 8, 1); title('Reference featureSet')
subplot(1, 2, 2), 
M2S_plotMZRT_featureSet(targetFeatures, 1, 8, 1); title('Target featureSet')


idx1 = any(true_matching > 0, 2);
idx2 = any(true_matching > 0, 1);
inds1 = find(idx1);
inds2 = find(idx2);

k = 3;
%diffs = targetFeatures(idx2, k) - refFeatures(idx1, k);
diffs = log10(targetFeatures(idx2, k)) - log10(refFeatures(idx1, k));

%D = abs(targetFeatures(idx2, k) - refFeatures(idx1, k)');
%M = matchpairs(D, 1e6);

figure(1)
histogram(diffs)

figure(2)
scatter(refFeatures(idx1, k), targetFeatures(idx2, k))

figure(3)
scatter(refFeatures(idx1, k), diffs)

refdiffs = abs(diff(refFeatures(idx1, k)));
histogram(refdiffs(refdiffs < 1))
histogram(abs(diff(targetFeatures(idx2, k))))

diffmat = refFeatures(:, k) - targetFeatures(:, k)';
diffmat = diffmat(~idx1, ~idx2);
triuidx = logical(triu(ones(size(diffmat)), 1));
diffmat = diffmat(triuidx);
histogram(diffmat(abs(diffmat) < 0.01))

%%*************************************************************************
%% Procedure part 1: find all possible matches
%%*************************************************************************

% create a structure to keep the options chosen at each step
opt = struct;

%% Set thresholds for matching all features 

opt.multThresh.RT_intercept = [-3.5, 3.5];
opt.multThresh.RT_slope = [0, 0];
opt.multThresh.MZ_intercept = [-noise, noise];
opt.multThresh.MZ_slope = [0, 0];
opt.multThresh.log10FI_intercept = [-0.031, 0.031];
opt.multThresh.log10FI_slope = [0, 0];
opt.FIadjustMethod = 'none'; % {'none','median','regression'}


% Match all features within defined thresholds

% Define the plot type as:
% No plot: plotType = 0
% Scatter plots: plotType = 1
% Multiple match plots: plotType = 2 (with lines connecting clusters of multiple
% matches containing the same feature). Also plots a network of all matches.

plotType = 1; 
[refSet, targetSet, Xr_connIdx, Xt_connIdx, opt] = M2S_matchAll(refFeatures, targetFeatures, opt.multThresh, opt.FIadjustMethod, plotType);


%%*************************************************************************
%% Procedure part 2: Calculate penalisation scores for each match
%%*************************************************************************

opt.neighbours.nrNeighbors = 0.01; % increase this
opt.calculateResiduals.neighMethod = 'cross'; % look at cross and circle
opt.pctPointsLoess = 0;
opt.adjustResiduals.residPercentile = NaN;
opt.weights.W = 1e-10*[1, 1, 1]; % sample simplex evenly
plotTypeResiduals = 1;

[Residuals_X, Residuals_trendline] = M2S_calculateResiduals(refSet, targetSet, Xr_connIdx, Xt_connIdx, opt.neighbours.nrNeighbors, opt.calculateResiduals.neighMethod, opt.pctPointsLoess, plotTypeResiduals)

[adjResiduals_X, residPercentile] = M2S_adjustResiduals(refSet, targetSet, Residuals_X, opt.adjustResiduals.residPercentile);
[penaltyScores] = M2S_defineW_getScores(refSet, targetSet, adjResiduals_X, opt.weights.W, 1); 


%% Decide the best of the multiple matches 
[eL, eL_INFO, CC_G1] = M2S_decideBestMatches(refSet, targetSet, Xr_connIdx, Xt_connIdx, penaltyScores);

%%*************************************************************************
%% Procedure part 3: find false positives (tighten thresholds)
%%*************************************************************************
opt.falsePos.methodType = 'scores'; %{'none','scores','byBins','trend_mad','residuals_mad'} 
opt.falsePos.nrMad = 5;
plotOrNot = 1;
[eL_final, eL_final_INFO] = M2S_findPoorMatches(eL, refSet, targetSet,opt.falsePos.methodType, opt.falsePos.nrMad, plotOrNot);


%% Save the results table with multiple matches
% writetable(eL_final,'M2S_edgeList_final.xlsx','Sheet',1)

% Summary with number of (multiple matches) discarded, false positive and true positive matches
tableOfMatches = array2table([nansum(isnan(eL_final.notFalsePositives)), nansum(eL_final.notFalsePositives==0), nansum(eL_final.notFalsePositives==1)], 'VariableNames', {'DiscardedMatches', 'PoorMatches', 'PositiveMatches'});


%% Compare found matching to true matching


refFeatures_idx = eL_final.Xr_connIdx(eL_final.notFalsePositives == 1);
targetFeatures_idx = eL_final.Xt_connIdx(eL_final.notFalsePositives == 1);
matching = zeros(num_refs, num_targets);
matching(sub2ind([num_refs, num_targets], refFeatures_idx, targetFeatures_idx)) = 1;

%save(sprintf("matchings/MS_matching_%g_%g_%s.mat", overlap, noise, drift), "matching")


figure(1)
subplot(1, 2, 1)
imagesc(true_matching)
title("True Matching")
subplot(1, 2, 2)
imagesc(matching)
title("MS Matching")

%TP = sum((matching == 1) & (true_matching == 1), 'all');
%FP = sum((matching == 1) & (true_matching == 0), 'all');
%FN = sum((matching == 0) & (true_matching == 1), 'all');
%TN = sum((matching == 0) & (true_matching == 0), 'all');

ref_pos_matches = all(matching(true_refFeatures_idx, :) == true_matching(true_refFeatures_idx, :), 2);
refTP = sum(ref_pos_matches);
refFN = sum(~ref_pos_matches);
ref_neg_matches = all(matching(~true_refFeatures_idx, :) == true_matching(~true_refFeatures_idx, :), 2);
refTN = sum(ref_neg_matches);
refFP = sum(~ref_neg_matches);
refSensitivity = refTP / (refTP + refFN)
refSpecificity = refTN / (refTN + refFP)

target_pos_matches = all(matching(true_targetFeatures_idx, :) == true_matching(true_targetFeatures_idx, :), 2);
targetTP = sum(target_pos_matches);
targetFN = sum(~target_pos_matches);
target_neg_matches = all(matching(~true_targetFeatures_idx, :) == true_matching(~true_targetFeatures_idx, :), 2);
targetTN = sum(target_neg_matches);
targetFP = sum(~target_neg_matches);
targetSensitivity = targetTP / (targetTP + targetFN)
targetSpecificity2 = targetTN / (targetTN + targetFP)
