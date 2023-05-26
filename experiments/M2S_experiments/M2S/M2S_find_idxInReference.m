%% function find_idxInReference
% idx_inReference = find_idxInReference(listWithMultiples,uniquesList)
% This function finds the strings idx of listWithMultiples in the uniquesList
% It is used e.g. when one has a list with strings occurring multiple times
% and we want to know the indices of each one in a reference list of unique
% strings (e.g. Match nodes of network edges into a dataset VarInfo)
% example: idx = find_idxInReference(listWithMultiples,uniquesList)
function idx_inReference = find_idxInReference(listWithMultiples,uniquesList)
idx_inReference = NaN(size(listWithMultiples));

classType = class(listWithMultiples);

if string(classType) == 'string'
    listWithMultiples = cellstr(listWithMultiples);
end

if string(classType) == 'cell'
    uniquesList_str = string(uniquesList);
    for a=1:length(listWithMultiples)
        tempIdx = find(uniquesList_str == listWithMultiples(a));
        if ~isempty(tempIdx)
            idx_inReference(a,1) = tempIdx;
        end
    end
else
    for a=1:length(listWithMultiples)
        tempIdx = find((uniquesList) == listWithMultiples(a));
        if ~isempty(tempIdx)
            idx_inReference(a,1) = tempIdx;
        end
    end
end
if sum(isnan(idx_inReference))>0
    fprintf('\n *** There are %d NaN indices in the vector (strings not found) ***\n',sum(isnan(idx_inReference)));
end
    
