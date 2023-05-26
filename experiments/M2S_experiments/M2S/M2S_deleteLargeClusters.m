%% This function creates a ref and target featureSets (unmatched) from 
% refSet and targetSet obtained from matching. It removes large clusters 
% (with more than maxFeaturesInCluster features in cluster) from those 
% matched sets. It then makes it easier to find thresholds with 
% genetic algorithm or manually

function [refFeatures_noBigClusters,targetFeatures_noBigClusters,refSet_noBigClusters,targetSet_noBigClusters,Xr_connIdx_noBigClusters,Xt_connIdx_noBigClusters, opt_noBigClusters] = M2S_deleteLargeClusters(refSet,targetSet,maxFeaturesInCluster,opt)

M2S_figureH(0.9,0.6);
set(gcf,'Name','Initial data with all features')
load ('M2ScolorScheme.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
refMZRT_str_sel = M2S_createLabelMZRT('ref',refSet(:,2),refSet(:,1));
targetMZRT_str_sel = M2S_createLabelMZRT('target',targetSet(:,2),targetSet(:,1));

Gtemp0 =graph(refMZRT_str_sel,targetMZRT_str_sel);% String nodes
Gtemp0.Nodes.CCnr = (conncomp(Gtemp0))';

nFeaturesInCC = tabulate(Gtemp0.Nodes.CCnr);
frequencyNodesInCC = tabulate(nFeaturesInCC(:,2));
subplot(1,3,1), bar(frequencyNodesInCC(:,1),frequencyNodesInCC(:,2)) , grid on, title('Frequency of clusters with n nodes');
text(frequencyNodesInCC(:,1),frequencyNodesInCC(:,2),string(frequencyNodesInCC(:,2)),'HorizontalAlignment','center','VerticalAlignment','bottom');
idxCC = find(nFeaturesInCC(:,2)>maxFeaturesInCluster);
nodesToDelete_idx=[];
for a=1:length(idxCC)
    nodesToDelete_idx=[nodesToDelete_idx; find(Gtemp0.Nodes.CCnr == idxCC(a))];
end
Gtemp0=rmnode(Gtemp0,nodesToDelete_idx);% Line not needed if using string nodes.

% Get the relevant features (in clusters with less than n features)
[~,refIdx,~] = intersect(refMZRT_str_sel,Gtemp0.Nodes.Name);
[~,targetIdx,~] = intersect(targetMZRT_str_sel,Gtemp0.Nodes.Name);
refFeatures_noBigClusters = refSet(refIdx,:);
targetFeatures_noBigClusters = targetSet(targetIdx,:);

% Plot the graph with all clusters
G1 =digraph(refMZRT_str_sel,targetMZRT_str_sel);% String nodes
subplot(1,3,[2,3]);
G1.Nodes.refNode = double(contains(G1.Nodes.Name,'ref'));
p1 = plot(G1,'LineWidth',2,'EdgeColor',M2Scolor.lgrey); 
highlight(p1,find(G1.Nodes.refNode==1),'NodeColor',M2Scolor.dblue,'MarkerSize',3);
highlight(p1,find(G1.Nodes.refNode==0),'NodeColor',M2Scolor.lblue,'MarkerSize',3);
set(gca,'XTick',[], 'YTick', []); axis tight; drawnow;
title('Ref nodes in dark blue : Target nodes in light blue')


[refSet_noBigClusters,targetSet_noBigClusters,Xr_connIdx_noBigClusters,Xt_connIdx_noBigClusters, opt_noBigClusters] = M2S_matchAll(refFeatures_noBigClusters,targetFeatures_noBigClusters,opt.multThresh,opt.FIadjustMethod,2);
%close

