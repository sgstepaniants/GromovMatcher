% This function is to be used with M2S_calculateResiduals

function [MatchSet_neighbors_idx]=M2S_findNeighbours(MatchSet,MatchSet_idx,Set,Set_idx,nrNeighbors,XdimForNeighbors)

MatchSet_neighbors_idx=NaN(size(Set,1),nrNeighbors);
for featurePair = 1:size(Set,1)
    current_Set_idx = Set_idx(featurePair);% current idx initial
    temp_MatchSet=MatchSet;
    temp_MatchSet(MatchSet_idx==current_Set_idx,:)=NaN;
    % calculate zscores
    %[Z_MatchSet,MU,SIGMA] = zscore(temp_MatchSet(:,Xdim));
    MU = nanmean(temp_MatchSet(:,XdimForNeighbors)); 
    SIGMA = nanstd(temp_MatchSet(:,XdimForNeighbors));
    
    Z_MatchSet = (temp_MatchSet(:,XdimForNeighbors) - MU)./ SIGMA;
    Z_Set_featurePair =( Set(featurePair,XdimForNeighbors)- MU ) ./ SIGMA ;

    % Find neighbours indices:
    % Using machine learning toolbox
    % MatchSet_neighbors_idx(featurePair,:) = knnsearch(Z_MatchSet,Z_Set_featurePair,'K',nrNeighbors);
    
    % Not using machine learning toolbox
    tempDist = Z_MatchSet - repmat(Z_Set_featurePair,size(Z_MatchSet,1),1);
    tempDist_sqrtSSQ = sqrt(sum(tempDist.*tempDist,2));
    [~,sorted_tempDist_sqrtSSQ_idx]= sort(tempDist_sqrtSSQ);
    MatchSet_neighbors_idx(featurePair,:) = (sorted_tempDist_sqrtSSQ_idx(1:nrNeighbors))';
end
