function [] = M2S_script(batch, num_batches)
    batch = str2num(batch)
    num_batches = str2num(num_batches)
    addpath("../..")
    addpath("M2S")
    
    saved_inds = [];
    save_filename = "M2S_saved_results_normalized'.mat"
    if isfile(save_filename)
        M2S_results = load(save_filename);
        all_ref_sensitivities = M2S_results.all_ref_sensitivities;
        all_ref_specificities = M2S_results.all_ref_specificities;
        all_target_sensitivities = M2S_results.all_target_sensitivities;
        all_target_specificities = M2S_results.all_target_specificities;
        saved_inds = find(~(isnan(all_ref_sensitivities) & isnan(all_ref_specificities) & isnan(all_target_sensitivities) & isnan(all_target_specificities)));
    end
    
    overlaps = {0.25, 0.5, 0.75};
    sigmasM = {0.01};
    sigmasRT = {0, 0.2, 0.5, 1};
    sigmasFI = {0}; %{0.05, 0.1, 0.5, 1, 5};
    rhos = {0};
    drifts = {"smooth"}; %{"smooth", "piecewise2"};
    trials = num2cell(1:20);
    normalize = "False";
    
    FIadjustMethods = {"none"}; %{"none", "median", "regression"};
    nrNeighborsList = {0.01, 0.05, 0.1, 0.5, 1};
    neighMethods = {"cross", "circle"};
    pctPointsLoessList = {0, 0.1, 0.5};
    weights = {[1, 1, 0]}; %{[1, 1, 1]}
    methodTypes = {"none", "scores", "byBins", "trend_mad", "residuals_mad"};
    nrMads = {1, 3, 5};
    
    hyperparams = {overlaps, sigmasM, sigmasRT, sigmasFI, rhos, drifts, trials, FIadjustMethods, nrNeighborsList, neighMethods, pctPointsLoessList, weights, methodTypes, nrMads};
    num_hyperparams = length(hyperparams);
    sizes = zeros(1, num_hyperparams);
    for i = 1:num_hyperparams
        sizes(i) = length(hyperparams{i});
    end
    prod_inds = setdiff(1:prod(sizes), saved_inds);
    num_prods = length(prod_inds);
    rng(1)
    prod_inds = prod_inds(randperm(num_prods));
    
    batch_size = ceil(num_prods/num_batches);
    batch_inds = prod_inds(1+(batch-1)*batch_size:min(batch*batch_size, num_prods));
    
    drawnow('update')
    fprintf("%i M2S runs in this batch\n", length(batch_inds))
    drawnow('update')
    
    for ind = batch_inds
        hyperparam_ind = cell(1, num_hyperparams);
        [hyperparam_ind{:}] = ind2sub(sizes, ind);
        hyperparam_ind = cat(1, hyperparam_ind{:});
        
        overlap = overlaps{hyperparam_ind(1)};
        sigmaM = sigmasM{hyperparam_ind(2)};
        sigmaRT = sigmasRT{hyperparam_ind(3)};
        sigmaFI = sigmasFI{hyperparam_ind(4)};
        rho = rhos{hyperparam_ind(5)};
        drift = drifts{hyperparam_ind(6)};
        trial = hyperparam_ind(7);
        FIadjustMethod = FIadjustMethods{hyperparam_ind(8)};
        nrNeighbors = nrNeighborsList{hyperparam_ind(9)};
        neighMethod = neighMethods{hyperparam_ind(10)};
        pctPointsLoess = pctPointsLoessList{hyperparam_ind(11)};
        weight = weights{hyperparam_ind(12)};
        methodType = methodTypes{hyperparam_ind(13)};
        nrMad = nrMads{hyperparam_ind(14)};
        
        % Generate two LC-MS datasets using python script
        %runstr = sprintf("Run_GW_samples.py '%f' '%f' '%f' '%f' '%f' '%s' '%d'", overlap, sigmaM, sigmaRT, sigmaFI, rho, drift, trial);
        %[Data12, Data2, true_matching] = pyrunfile(runstr, ["Data1" "Data2" "true_matching"]);
        %Data1 = double(Data1);
        %Data2 = double(Data2);
        %true_matching = double(true_matching) > 0;
        
        
        % Generate two LC-MS datasets using python script (for older Matlab versions)
        dataset = "../../datasets/metabolomics_normalized_data.xlsx"
        runstr = sprintf("python ../../simulate_data.py '%s' '%f' '%f' '%f' '%f' '%f' '%s' '%d' '%s'", dataset, overlap, sigmaM, sigmaRT, sigmaFI, rho, drift, trial, normalize);
        [status, cmdout] = system(runstr);
        %cmdout
        assert(status == 0)
        
        ind1 = strfind(cmdout, "Data1");
        ind2 = strfind(cmdout, "Data2");
        ind3 = strfind(cmdout, "Matching");
        Data1 = str2num(cmdout(ind1+7:ind2-2));
        Data2 = str2num(cmdout(ind2+7:ind3-2));
        true_matching = str2num(cmdout(ind3+10:end)) > 0;
        
        % Columns in the order RT, MZ, median FI
        refFeatures = [Data1(3, :)', Data1(2, :)', median(Data1(4:end, :), 1)'];
        targetFeatures = [Data2(3, :)', Data2(2, :)', median(Data2(4:end, :), 1)'];
        
        assert(sum(sum(true_matching)) / size(true_matching, 1) == 2*overlap/(1+overlap))
        
        true_refFeatures_idx = sum(true_matching, 2) > 0;
        [~, true_targetFeatures_inds] = max(true_matching(true_refFeatures_idx, :), [], 2);
        true_targetFeatures_idx = false(size(true_matching, 2), 1);
        true_targetFeatures_idx(true_targetFeatures_inds) = true;
        
        % Create labels for all features (pass MZ first and RT second)
        refFeatures(:, 1)
        refFeatures(:, 2)
        [refMZRT_str] = M2S_createLabelMZRT('ref', refFeatures(:, 2), refFeatures(:, 1));
        [targetMZRT_str] = M2S_createLabelMZRT('target', targetFeatures(:, 2), targetFeatures(:, 1));
        
        num_refs = length(refMZRT_str);
        num_targets = length(targetMZRT_str);
        
        
        %%*************************************************************************
        %% Procedure part 1: find all possible matches
        %%*************************************************************************
        
        % create a structure to keep the options chosen at each step
        opt = struct;
        
        %% Set thresholds for matching all features 
        
        opt.multThresh.RT_intercept = [-3.5, 3.5];
        opt.multThresh.RT_slope = [0, 0];
        opt.multThresh.MZ_intercept = [-sigmaM, sigmaM];
        opt.multThresh.MZ_slope = [0, 0];
        opt.multThresh.log10FI_intercept = [-1000, 1000]; %[-0.2, 0.2];
        opt.multThresh.log10FI_slope = [0, 0];
        opt.FIadjustMethod = FIadjustMethod;
        
        
        % Match all features within defined thresholds
        
        % Define the plot type as:
        % No plot: plotType = 0
        % Scatter plots: plotType = 1
        % Multiple match plots: plotType = 2 (with lines connecting clusters of multiple
        % matches containing the same feature). Also plots a network of all matches.
        
        plotType = 0; 
        [refSet, targetSet, Xr_connIdx, Xt_connIdx, opt] = M2S_matchAll(refFeatures, targetFeatures, opt.multThresh, opt.FIadjustMethod, plotType);
        
        
        %%*************************************************************************
        %% Procedure part 2: Calculate penalisation scores for each match
        %%*************************************************************************
        
        opt.neighbours.nrNeighbors = nrNeighbors;
        opt.calculateResiduals.neighMethod = neighMethod;
        opt.pctPointsLoess = pctPointsLoess;
        opt.adjustResiduals.residPercentile = NaN;
        opt.weights.W = weight;
        plotTypeResiduals = 0;
        
        [Residuals_X, Residuals_trendline] = M2S_calculateResiduals(refSet, targetSet, Xr_connIdx, Xt_connIdx, opt.neighbours.nrNeighbors, opt.calculateResiduals.neighMethod, opt.pctPointsLoess, plotTypeResiduals);
        
        [adjResiduals_X, residPercentile] = M2S_adjustResiduals(refSet, targetSet, Residuals_X, opt.adjustResiduals.residPercentile);
        [penaltyScores] = M2S_defineW_getScores(refSet, targetSet, adjResiduals_X, opt.weights.W, 1); 
        
        
        %% Decide the best of the multiple matches 
        [eL, eL_INFO, CC_G1] = M2S_decideBestMatches(refSet, targetSet, Xr_connIdx, Xt_connIdx, penaltyScores);
        
        %%*************************************************************************
        %% Procedure part 3: find false positives (tighten thresholds)
        %%*************************************************************************
        opt.falsePos.methodType = methodType;
        opt.falsePos.nrMad = nrMad;
        plotOrNot = 0;
        [eL_final, eL_final_INFO] = M2S_findPoorMatches(eL, refSet, targetSet,opt.falsePos.methodType, opt.falsePos.nrMad, plotOrNot);
        
        %% Compare found matching to true matching
        refFeatures_idx = eL_final.Xr_connIdx(eL_final.notFalsePositives == 1 & eL_final.is_Best == 1);
        targetFeatures_idx = eL_final.Xt_connIdx(eL_final.notFalsePositives == 1 & eL_final.is_Best == 1);
        matching = zeros(num_refs, num_targets);
        matching(sub2ind([num_refs, num_targets], refFeatures_idx, targetFeatures_idx)) = 1;
        
        ref_pos_matches = all(matching(true_refFeatures_idx, :) == true_matching(true_refFeatures_idx, :), 2);
        refTP = sum(ref_pos_matches);
        refFN = sum(~ref_pos_matches);
        ref_neg_matches = all(matching(~true_refFeatures_idx, :) == true_matching(~true_refFeatures_idx, :), 2);
        refTN = sum(ref_neg_matches);
        refFP = sum(~ref_neg_matches);
        refSensitivity = refTP / (refTP + refFN);
        refSpecificity = refTN / (refTN + refFP);
        
        target_pos_matches = all(matching(:, true_targetFeatures_idx) == true_matching(:, true_targetFeatures_idx), 1);
        targetTP = sum(target_pos_matches);
        targetFN = sum(~target_pos_matches);
        target_neg_matches = all(matching(:, ~true_targetFeatures_idx) == true_matching(:, ~true_targetFeatures_idx), 1);
        targetTN = sum(target_neg_matches);
        targetFP = sum(~target_neg_matches);
        targetSensitivity = targetTP / (targetTP + targetFN);
        targetSpecificity = targetTN / (targetTN + targetFP);
        
        TP = sum((matching == 1) & (true_matching == 1), 'all');
        FP = sum((matching == 1) & (true_matching == 0), 'all');
        FN = sum((matching == 0) & (true_matching == 1), 'all');
        TN = sum((matching == 0) & (true_matching == 0), 'all');

        precision = TP / (TP + FP);
        recall = TP / (TP + FN); % sensitivity
        specificity = TN / (TN + FP);
        F1 = 2*precision*recall/(precision+recall);
        
        drawnow('update')
        fprintf("\n")
        fprintf("Combination:")
        fprintf(" %i", hyperparam_ind)
        fprintf("\n")
        fprintf("Precision: %f\n", precision)
        fprintf("Recall: %f\n", recall)
        fprintf("Specificity: %f\n", specificity)
        fprintf("F1: %f\n", F1)
        fprintf("Reference Sensitivity: %f\n", refSensitivity)
        fprintf("Reference Specificity: %f\n", refSpecificity)
        fprintf("Target Sensitivity: %f\n", targetSensitivity)
        fprintf("Target Specificity: %f\n", targetSpecificity)
        fprintf("\n")
        drawnow('update')
    end
end
