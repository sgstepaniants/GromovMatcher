files = dir("./");
filenames = [];
for fileind = 1:length(files)
    filename = string(files(fileind).name);
    if startsWith(filename, "M2Sresults")
        filenames = [filenames, filename];
    end
end

%sizes = [3, 1, 4, 5, 1, 1, 20, 1, 5, 2, 3, 1, 5, 3];
sizes = [3, 1, 4, 1, 1, 1, 20, 1, 5, 2, 3, 1, 5, 3];

save_filename = "M2S_saved_results_normalized.mat"
if isfile(save_filename)
    M2S_results = load(save_filename);
    all_precisions = M2S_results.all_precisions;
    all_recalls = M2S_results.all_recalls;
    all_specificities = M2S_results.all_specificities;
    all_F1s = M2S_results.all_F1s;
    all_ref_sensitivities = M2S_results.all_ref_sensitivities;
    all_ref_specificities = M2S_results.all_ref_specificities;
    all_target_sensitivities = M2S_results.all_target_sensitivities;
    all_target_specificities = M2S_results.all_target_specificities;
else
    all_precisions = zeros(sizes);
    all_precisions(:) = nan;
    all_recalls = zeros(sizes);
    all_recalls(:) = nan;
    all_specificities = zeros(sizes);
    all_specificities(:) = nan;
    all_F1s = zeros(sizes);
    all_F1s(:) = nan;
    all_ref_sensitivities = zeros(sizes);
    all_ref_sensitivities(:) = nan;
    all_ref_specificities = zeros(sizes);
    all_ref_specificities(:) = nan;
    all_target_sensitivities = zeros(sizes);
    all_target_sensitivities(:) = nan;
    all_target_specificities = zeros(sizes);
    all_target_specificities(:) = nan;
end

for filename = filenames
    display(filename)
    [combinations, precisions, recalls, specificities, F1s, ref_sensitivities, ref_specificities, target_sensitivities, target_specificities] = read_M2S_results(filename);
    
    minlen = min([length(precisions), length(recalls), length(specificities), length(F1s), length(ref_sensitivities), length(ref_specificities), length(target_sensitivities), length(target_specificities)]);
    assert(size(combinations, 2) == minlen)
    
    for i = 1:size(combinations, 2)
        combination = num2cell(combinations(:, i));
        %[combination{:}]
        all_precisions(combination{:}) = precisions(i);
        all_recalls(combination{:}) = recalls(i);
        all_specificities(combination{:}) = specificities(i);
        all_F1s(combination{:}) = F1s(i);
        all_ref_sensitivities(combination{:}) = ref_sensitivities(i);
        all_ref_specificities(combination{:}) = ref_specificities(i);
        all_target_sensitivities(combination{:}) = target_sensitivities(i);
        all_target_specificities(combination{:}) = target_specificities(i);
    end
end

sum(~isnan(all_ref_sensitivities), 'all')

save(save_filename, 'all_precisions', 'all_recalls', 'all_specificities', 'all_F1s', 'all_ref_sensitivities', 'all_ref_specificities', 'all_target_sensitivities', 'all_target_specificities')