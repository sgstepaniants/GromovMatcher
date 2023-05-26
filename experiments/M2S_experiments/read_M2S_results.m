function [combinations, precisions, recalls, specificities, F1s, ref_sensitivities, ref_specificities, target_sensitivities, target_specificities] = read_M2S_results(filename)
    fileID = fopen(filename, 'r');
    
    combinations = [];
    precisions = [];
    recalls = [];
    specificities = [];
    F1s = [];
    ref_sensitivities = [];
    ref_specificities = [];
    target_sensitivities = [];
    target_specificities = [];
    
    tline = fgetl(fileID);
    while ischar(tline)
        if contains(tline, ": ")
            splits = split(tline, ": ");
            name = splits{1};
            val = splits{end};

            if startsWith(name, "Combination")
                combination = cellfun(@str2num, split(val));
                combinations = [combinations, combination];
            elseif startsWith(name, "Precision")
                precisions(end+1) = str2num(val);
            elseif startsWith(name, "Recall")
                recalls(end+1) = str2num(val);
            elseif startsWith(name, "Specificity")
                specificities(end+1) = str2num(val);
            elseif startsWith(name, "F1")
                F1s(end+1) = str2num(val);
            elseif startsWith(name, "Reference Sensitivity")
                ref_sensitivities(end+1) = str2num(val);
            elseif startsWith(name, "Reference Specificity")
                ref_specificities(end+1) = str2num(val);
            elseif startsWith(name, "Target Sensitivity")
                target_sensitivities(end+1) = str2num(val);
            elseif startsWith(name, "Target Specificity")
                target_specificities(end+1) = str2num(val);
            end
        end
        
        tline = fgetl(fileID);
    end
    fclose(fileID);
end