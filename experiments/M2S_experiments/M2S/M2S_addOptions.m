%% [newOpt] = M2S_addOptions(optOld,additionalOpt)
% 
% Function to add new fields to a structure, or new values to existent fields.
% If the field already exists, it is overwritten with the new values.

function [newOpt] = M2S_addOptions(optOld,additionalOpt)
oldFields = fieldnames(optOld);
additionalFields = fieldnames(additionalOpt);
for f = 1:length(additionalFields)
        optOld.(additionalFields{f}) = additionalOpt.(additionalFields{f})
end
newOpt = optOld;
