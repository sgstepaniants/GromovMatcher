function [G1,CC] = M2S_infoClusters(refSet,targetSet,dimNr,line_style)
% [G1,CC] = M2S_infoClusters(refSet,targetSet,dimNr,line_style)
% Function to inspect the results of matching.
%NOTE: at this point this only works for dimNr <= 2

% dimNr:
% 1=RT; 2=MZ; 3=log10FIdist vs log10FIref; 4 = log10FItarget vs log10FIref;
% line_style = {':','-'}

if dimNr>2
    error('dimNr must be 1 or 2')
end

if dimNr == 1
    dimLabel = 'RT';
    xdimLabel = 'RTref';
    ydimLabel = 'RTdist';
elseif dimNr == 2
    dimLabel = 'MZ';
    xdimLabel = 'MZref';
    ydimLabel = 'MZdist';
elseif dimNr == 3
    dimLabel = 'log10FI';
    xdimLabel = 'log10FIref';
    ydimLabel = 'log10FIdist';
elseif dimNr == 4
    dimLabel = 'log10FI';
    xdimLabel = 'log10FIref';
    ydimLabel = 'log10FItarget';
end

%% Create a GRAPH with all matches 
CC = struct;
MZRTstr_Ref = string(M2S_createLabelMZRT('REF',refSet(:,2),refSet(:,1)));
MZRTstr_Target = string(M2S_createLabelMZRT('TARGET',targetSet(:,2),targetSet(:,1)));

G1_temp = graph(MZRTstr_Ref,MZRTstr_Target);% Only for connected components
G1 = digraph(MZRTstr_Ref,MZRTstr_Target);

% Find if node is reference node
G1.Nodes.refNode = double(contains(G1.Nodes.Name,'REF'));
% Find in which cluster the node is
G1.Nodes.CCnr = (conncomp(G1_temp))';
% Find in which cluster the edge is
G1.Edges.CCnr = G1.Nodes.CCnr(M2S_find_idxInReference(G1.Edges.EndNodes(:,1),G1.Nodes.Name));

% Find number of nodes (features) and edges (matches) in each cluster
CC.nNodes = tabulate(G1.Nodes.CCnr); 
CC.nNodes = CC.nNodes(:,1:2);
CC.nEdges = tabulate(G1.Edges.CCnr); 
CC.nEdges = CC.nEdges(:,1:2);

% nExtraFeaturesInCC = CC.nNodes(:,2)-2;
% Find number of clusters with N nodes or N edges
CC.freq_clustersWithNnodes = tabulate(CC.nNodes(:,2));
CC.freq_clustersWithNedges = tabulate(CC.nEdges(:,2));

% Find how many nodes & edges are in the same cluster as each node
G1.Nodes.nrNodesInCC = CC.nNodes(M2S_find_idxInReference(G1.Nodes.CCnr,CC.nNodes(:,1)),2);
G1.Nodes.nrEdgesInCC = CC.nEdges(M2S_find_idxInReference(G1.Nodes.CCnr,CC.nEdges(:,1)),2);

% Find how many nodes & edges are in the same cluster as each edge
G1.Edges.nrNodesInCC = CC.nNodes(M2S_find_idxInReference(G1.Edges.CCnr,CC.nNodes(:,1)),2);
G1.Edges.nrEdgesInCC = CC.nEdges(M2S_find_idxInReference(G1.Edges.CCnr,CC.nEdges(:,1)),2);

% Find distances between single matches
allDistances = [targetSet(:,1:2) - refSet(:,1:2),log10(targetSet(:,3)) - log10(refSet(:,3))];

MatchedClusters.singleMatch.distances = allDistances(G1.Edges.nrNodesInCC==2,:);
MatchedClusters.multMatch.distances = allDistances(G1.Edges.nrNodesInCC>2,:);

M2S_figureH(.8,.3); set(gcf,'Name','Distances between target and ref features matched')
subplot(1,3,1), histogram(MatchedClusters.singleMatch.distances(:,1),41), hold on
histogram(MatchedClusters.multMatch.distances(:,1),41), axis tight, grid on, legend({'single','mult'})
xlabel('RTdist'); ylabel('Counts')
subplot(1,3,2), histogram(MatchedClusters.singleMatch.distances(:,2),41), hold on
histogram(MatchedClusters.multMatch.distances(:,2),41), axis tight, grid on, legend({'single','mult'})
xlabel('MZdist'); ylabel('Counts')
subplot(1,3,3), histogram(MatchedClusters.singleMatch.distances(:,3),41), hold on
histogram(MatchedClusters.multMatch.distances(:,3),41), axis tight, grid on, legend({'single','mult'})
xlabel('log10FIdist'); ylabel('Counts')
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plots

%% Colour definition
load ('M2ScolorScheme.mat') 

colourForEdges = CC.freq_clustersWithNedges;
localColormap = num2cell(jet(size(colourForEdges,1)),2);

coloursForEdges = G1.Edges.nrEdgesInCC;
nColoursEdges = max(unique(coloursForEdges));
coloursForNodes = G1.Edges.nrNodesInCC;
nColoursNodes = max(unique(coloursForNodes));
nColours = max([nColoursEdges,nColoursNodes]);% max between edge and node colours
nColours_idx = round(linspace(1,256,nColours));
M2Scolormap_local = M2Scolormap(nColours_idx,:);


%% Network
%figure('Position',[1 1 1000 750],

M2S_figureH(.6,.6);
%set(gcf,'Name','Nodes as metabolomic features:')
movegui(gcf,'center')
p1 = plot(G1,'LineWidth',2,'EdgeColor',M2Scolor.lgrey); 
highlight(p1,find(G1.Nodes.refNode==1),'Marker','o','MarkerSize',4);
highlight(p1,find(G1.Nodes.refNode==0),'Marker','^','MarkerSize',4);
p1.NodeCData = G1.Nodes.nrNodesInCC;
p1.EdgeCData = G1.Edges.nrEdgesInCC;
set(gca,'XTick',[], 'YTick', []); axis tight; drawnow;
set(gcf,'Name','Features matched between datasets: Ref (circles) and Target (triangles)')
title('Matched features.       Colours: nodes - number of features in cluster     edges - number of matches in cluster')
colormap(M2Scolormap_local); caxis([1,nColours]); clb = colorbar;
nTicksPoints = (linspace(1,nColours,2*nColours+1))'; nTicksPoints = nTicksPoints(2:2:end);
set(clb, 'ticks',nTicksPoints , 'ticklabels', cellstr(num2str((1:nColours)')));


%% PLOT: Distances, e.g. RTdist vs RTref

%subplot(1,3,1)
M2S_figureH(.6,.6);
set(gcf,'Name','Dots represent matches. Clusters of matches with same features are connected by lines.')
% title('Dots coloured by number of features in cluster. Lines coloured by number of matches in cluster')
if dimNr <=2
    scatter(refSet(:,dimNr),targetSet(:,dimNr)-refSet(:,dimNr),20*ones(size(refSet,1),1),G1.Edges.nrNodesInCC,'filled')
% scatter(refSet(:,dimNr),targetSet(:,dimNr)-refSet(:,dimNr),20*ones(size(refSet,1),1),M2Scolormap_local(G1.Edges.nrNodesInCC),'filled')
elseif dimNr == 3
    scatter(log10(refSet(:,dimNr)),log10(targetSet(:,dimNr))-log10(refSet(:,dimNr)),20*ones(size(refSet,1),1),G1.Edges.nrNodesInCC,'filled')
elseif dimNr == 4
    scatter(log10(refSet(:,3)),log10(targetSet(:,3)),20*ones(size(refSet,1),1),G1.Edges.nrNodesInCC,'filled')
end
axis tight, grid on 
%scatter(refSet(:,1),targetSet(:,1)-refSet(:,1),16*ones(size(refSet,1),1),G1.Edges.nrNodesInCC,'filled')
hold on
Dist_sameRef = [];
Dist_sameTarget = [];
Dist_TargetRef_sameRef1 = [];
Dist_TargetRef_sameRef2 = [];
Dist_TargetRef_sameTarget1 = [];
Dist_TargetRef_sameTarget2 = [];
connectedMatches_global_idx1r = [];
connectedMatches_global_idx2r = [];
connectedMatches_global_idx1t = [];
connectedMatches_global_idx2t = [];
    
    % For each cluster
for CCnr = 1:max(G1.Edges.CCnr)
    % Find the indices of edges that make part of cluster
    CCedges_idx = find(G1.Edges.CCnr == CCnr);
    % localEdgeColour_idx = G1.Edges.nrEdgesInCC(CCedges_idx(1));
    
    % Find same ref nodes in cluster edges
    connectedMatches_local_idx1r = [];
    connectedMatches_local_idx2r = [];
    connectedMatches_local_idx1t = [];
    connectedMatches_local_idx2t = [];
    
    if length(CCedges_idx)>1
        % For the reference features in these edges:
        % for each of the edges in the cluster
        for r = 1:length(CCedges_idx)            
            % find edges containing the ref feature of this edge
            ref_local_idx = find(MZRTstr_Ref(CCedges_idx) == MZRTstr_Ref(CCedges_idx(r)));
            ref_local_idx = ref_local_idx(ref_local_idx>r);% only accept features that were not yet used
            % save the edge index of the ref current feature 
            connectedMatches_local_idx1r = [connectedMatches_local_idx1r;repmat(r,length(ref_local_idx),1)];
            % save the edge index of the feature(s) that match(es) the current one
            connectedMatches_local_idx2r=[connectedMatches_local_idx2r;ref_local_idx];
% NOTE: connectedMatches_local_idx1r is the ref feature index of a feature
% that also exists in the indices connectedMatches_local_idx2r
            
            
        end
        % find which of these edges contain the target feature
        for t = 1:length(CCedges_idx)
            target_local_idx = find(MZRTstr_Target(CCedges_idx) == MZRTstr_Target(CCedges_idx(t)));
            target_local_idx=target_local_idx(target_local_idx>t);
            connectedMatches_local_idx1t = [connectedMatches_local_idx1t;repmat(t,length(target_local_idx),1)];
            connectedMatches_local_idx2t=[connectedMatches_local_idx2t;target_local_idx];
        end
        % plot
        for cm = 1:length(connectedMatches_local_idx1r)
            Dist_TargetRef_sameRef1 = [Dist_TargetRef_sameRef1; targetSet(CCedges_idx(connectedMatches_local_idx1r(cm)),dimNr) - refSet(CCedges_idx(connectedMatches_local_idx1r(cm)),dimNr)];
            Dist_TargetRef_sameRef2 = [Dist_TargetRef_sameRef2; targetSet(CCedges_idx(connectedMatches_local_idx2r(cm)),dimNr) - refSet(CCedges_idx(connectedMatches_local_idx2r(cm)),dimNr)];
            Dist_sameRef = [Dist_sameRef;diff( [targetSet([CCedges_idx(connectedMatches_local_idx1r(cm));CCedges_idx(connectedMatches_local_idx2r(cm))],dimNr)]-...
               [refSet([CCedges_idx(connectedMatches_local_idx1r(cm));CCedges_idx(connectedMatches_local_idx2r(cm))],dimNr);])];
            plot([refSet([CCedges_idx(connectedMatches_local_idx1r(cm));CCedges_idx(connectedMatches_local_idx2r(cm))],dimNr)],...
               [targetSet([CCedges_idx(connectedMatches_local_idx1r(cm));CCedges_idx(connectedMatches_local_idx2r(cm))],dimNr)]-...
               [refSet([CCedges_idx(connectedMatches_local_idx1r(cm));CCedges_idx(connectedMatches_local_idx2r(cm))],dimNr)],line_style,'Color',M2Scolormap_local(G1.Edges.nrEdgesInCC(CCedges_idx(1)),:));
        end   
        for cm = 1:length(connectedMatches_local_idx1t)
            Dist_TargetRef_sameTarget1 = [Dist_TargetRef_sameTarget1; targetSet(CCedges_idx(connectedMatches_local_idx1t(cm)),dimNr) - refSet(CCedges_idx(connectedMatches_local_idx1t(cm)),dimNr)];
            Dist_TargetRef_sameTarget2 = [Dist_TargetRef_sameTarget2; targetSet(CCedges_idx(connectedMatches_local_idx2t(cm)),dimNr) - refSet(CCedges_idx(connectedMatches_local_idx2t(cm)),dimNr)];
            Dist_sameTarget = [Dist_sameTarget;diff( [targetSet([CCedges_idx(connectedMatches_local_idx1t(cm));CCedges_idx(connectedMatches_local_idx2t(cm))],dimNr)]-...
               [refSet([CCedges_idx(connectedMatches_local_idx1t(cm));CCedges_idx(connectedMatches_local_idx2t(cm))],dimNr)])];
            plot([refSet([CCedges_idx(connectedMatches_local_idx1t(cm));CCedges_idx(connectedMatches_local_idx2t(cm))],dimNr)],...
               [targetSet([CCedges_idx(connectedMatches_local_idx1t(cm));CCedges_idx(connectedMatches_local_idx2t(cm))],dimNr)]-...
               [refSet([CCedges_idx(connectedMatches_local_idx1t(cm));CCedges_idx(connectedMatches_local_idx2t(cm))],dimNr)],line_style,'Color',M2Scolormap_local(G1.Edges.nrEdgesInCC(CCedges_idx(1)),:));
        end  
    end
    % The global edges indices are
    connectedMatches_global_idx1r = [connectedMatches_global_idx1r;CCedges_idx(connectedMatches_local_idx1r)];
    % NOTE: connectedMatches_global_idx1r are the global edge indices of
    % features that also exist on indices connectedMatches_global_idx2r
    connectedMatches_global_idx2r = [connectedMatches_global_idx2r;CCedges_idx(connectedMatches_local_idx2r)];
    connectedMatches_global_idx1t = [connectedMatches_global_idx1t;CCedges_idx(connectedMatches_local_idx1t)];
    connectedMatches_global_idx2t = [connectedMatches_global_idx2t;CCedges_idx(connectedMatches_local_idx2t)];
end
colormap(M2Scolormap_local); caxis([1,nColours]); clb = colorbar;
nTicksPoints = (linspace(1,nColours,2*nColours+1))'; nTicksPoints = nTicksPoints(2:2:end);
set(clb, 'ticks',nTicksPoints , 'ticklabels', cellstr(num2str((1:nColours)')));
xlabel(xdimLabel); ylabel(ydimLabel);
title('Dots coloured by number of features in cluster. Lines coloured by number of matches in cluster')


%% This works but is not used
%{

% Edge features (e.g. RTdist) distances for the target features matching the same reference
RTref_sameRefFeature = refSet([connectedMatches_global_idx1r;connectedMatches_global_idx2r],1);
RTdist_sameRefFeature = targetSet([connectedMatches_global_idx1r;connectedMatches_global_idx2r],1) - refSet([connectedMatches_global_idx1r;connectedMatches_global_idx2r],1);
diffRT_sameRefFeature = [targetSet(connectedMatches_global_idx2r,1) - targetSet(connectedMatches_global_idx1r,1); targetSet(connectedMatches_global_idx2r,1) - targetSet(connectedMatches_global_idx1r,1)];
%diff_sameRefFeature = refSet(connectedMatches_global_idx2r) - refSet(connectedMatches_global_idx1r) % this is zero because it subtracts the same values

MZref_sameRefFeature = refSet([connectedMatches_global_idx1r;connectedMatches_global_idx2r],2);
MZdist_sameRefFeature = targetSet([connectedMatches_global_idx1r;connectedMatches_global_idx2r],2) - refSet([connectedMatches_global_idx1r;connectedMatches_global_idx2r],2);
diffMZ_sameRefFeature = [targetSet(connectedMatches_global_idx2r,2) - targetSet(connectedMatches_global_idx1r,2); targetSet(connectedMatches_global_idx2r,2) - targetSet(connectedMatches_global_idx1r,2)];

FIref_sameRefFeature = log10(refSet([connectedMatches_global_idx1r;connectedMatches_global_idx2r],3));
FIdist_sameRefFeature = log10(targetSet([connectedMatches_global_idx1r;connectedMatches_global_idx2r],3)) - log10(refSet([connectedMatches_global_idx1r;connectedMatches_global_idx2r],3));
diffFI_sameRefFeature = [log10(targetSet(connectedMatches_global_idx2r,3)) - log10(targetSet(connectedMatches_global_idx1r,3)); log10(targetSet(connectedMatches_global_idx2r,3)) - log10(targetSet(connectedMatches_global_idx1r,3))];


RT_penaltyScore = abs(RTdist_sameRefFeature).*abs(diffRT_sameRefFeature);
MZ_penaltyScore = abs(MZdist_sameRefFeature).*abs(diffMZ_sameRefFeature);
FI_penaltyScore = abs(FIdist_sameRefFeature).*abs(diffFI_sameRefFeature);

W = [1,1,0];
penaltyScore = W(1) * zscore(RT_penaltyScore) + W(2) * MZ_penaltyScore + W(3) * FI_penaltyScore;


M2S_figureH(0.9,0.6) 
subplot(1,3,1)
plot(refSet(:,1),targetSet(:,1)-refSet(:,1),'.k'), hold on
scatter(RTref_sameRefFeature,RTdist_sameRefFeature,18*ones(size(RTdist_sameRefFeature)),RT_penaltyScore,'filled'), axis tight, grid on
subplot(1,3,2)
plot(refSet(:,2),targetSet(:,2)-refSet(:,2),'.k'), hold on
scatter(MZref_sameRefFeature,MZdist_sameRefFeature,18*ones(size(MZdist_sameRefFeature)),MZ_penaltyScore,'filled'), axis tight, grid on
subplot(1,3,3)
plot(log10(refSet(:,3)),log10(targetSet(:,3))-log10(refSet(:,3)),'.k'), hold on
scatter(FIref_sameRefFeature,FIdist_sameRefFeature,18*ones(size(FIdist_sameRefFeature)),FI_penaltyScore,'filled'), axis tight, grid on
%}


% figure, plot(RTdist_sameRefFeature,abs(diffRT_sameRefFeature),'.'), grid on, axis tight
% xlabel('RTdist (between sets)'), ylabel('diff of target with same ref feature (only target set)')
% figure, plot3(RTref_sameRefFeature,RTdist_sameRefFeature,abs(diffRT_sameRefFeature),'.'), hold on, axis tight
% xlabel('RTref'), ylabel('RTdist (between sets)'), zlabel('diff of target with same ref feature (only target set)'), grid on, axis tight

%{
%% THIS SEEMS TO BE GOOD TO CREATE SCORES
figure, scatter(RTref_sameRefFeature,RTdist_sameRefFeature,18*ones(size(RTdist_sameRefFeature)),abs(RTdist_sameRefFeature).*abs(diffRT_sameRefFeature),'filled')


%% distances target-ref and targets that match the same ref
M2S_figureH(0.8,0.8); set(gcf,'Name',['Multiple match only (',dimLabel,' domain): Absolute difference between target features connected the same reference (and vice versa) vs ',ydimLabel])
subplot(1,2,1), plot([Dist_TargetRef_sameRef1;Dist_TargetRef_sameRef2],[abs(Dist_sameRef);abs(Dist_sameRef)],'.'), xlabel(ydimLabel), ylabel([dimLabel,' abs difference of features conn to same ref']), grid on, axis tight
%subplot(2,2,2), plot(Dist_TargetRef_sameRef2,abs(Dist_sameRef),'.'), xlabel(ydimLabel), ylabel([dimLabel,' abs difference of features conn to same ref']), grid on, axis tight
subplot(1,2,2), plot([Dist_TargetRef_sameTarget1;Dist_TargetRef_sameTarget2],[abs(Dist_sameTarget);abs(Dist_sameTarget)],'.'), xlabel(ydimLabel), ylabel([dimLabel,' abs difference of features conn to same target']), grid on, axis tight
%subplot(2,2,4), plot(Dist_TargetRef_sameTarget2,abs(Dist_sameTarget),'.'), xlabel(ydimLabel), ylabel([dimLabel,' abs difference of features conn to same target']), grid on, axis tight


M2S_figureH(0.8,0.8); 
%subplot(2,1,1), 
plot([Dist_TargetRef_sameRef1;Dist_TargetRef_sameRef2],[abs(Dist_sameRef);abs(Dist_sameRef)],'o'), hold on
for a=1:length(Dist_TargetRef_sameRef1)
     plot([Dist_TargetRef_sameRef1(a);Dist_TargetRef_sameRef2(a)],[abs(Dist_sameRef(a));abs(Dist_sameRef(a))],'-')
    % plot([Dist_TargetRef_sameTarget1(a);Dist_TargetRef_sameTarget2(a)],[abs(Dist_sameRef(a));abs(Dist_sameRef(a))],'-'),
end

%% Plot of      
M2S_figureH(0.8,0.3);
set(gcf,'Name',['Multiple match only (',dimLabel,' domain): Absolute difference between target features connected the same reference (and vice versa)'])
subplot(1,3,1),histogram(abs(Dist_sameRef),41), axis tight, grid on, xlabel([ydimLabel,' of features conn to same ref'])
subplot(1,3,2),histogram(abs(Dist_sameTarget),41), axis tight, grid on, xlabel([ydimLabel,' of features conn to same target'])
subplot(1,3,3),histogram(abs([Dist_sameRef;Dist_sameTarget]),41), axis tight, grid on, xlabel([ydimLabel,' of both'])

%}

%% Format the output:

%% CC
CC.nNodes = array2table(CC.nNodes,'VariableNames',{'clusterNumber','numberOfFeatures'});
CC.nEdges = array2table(CC.nEdges,'VariableNames',{'clusterNumber','numberOfMatches'});
CC.freq_clustersWithNnodes = array2table(CC.freq_clustersWithNnodes,'VariableNames',{'numberOfFeatures','Counts','Percentage'})
CC.freq_clustersWithNedges = array2table(CC.freq_clustersWithNedges,'VariableNames',{'numberOfMatches','Counts','Percentage'})

disp('DONE')
    
%     
% x = 1:100;
% ii = (1:size(CC.freq_clustersWithNedges,1))';
% y = ii*log(x);
% h = plot(x,y);
% set(h, {'color'}, num2cell(jet(length(ii)), 2));
%     
% M2S_plotLinkedPoints(refSet(:,1),targetSet(:,1)-refSet(:,1),Xr_connIdx_i,'ok',':',M2Scolor.lgrey,[]) 
% M2S_plotLinkedPoints(refSet(:,1),targetSet(:,1)-refSet(:,1),Xt_connIdx_i,'ok',':',M2Scolor.orange,[]) 
% %plot(refSet(FIdistOK_01==0,1),targetSet(FIdistOK_01==0,1)-refSet(FIdistOK_01==0,1),'.','MarkerSize',8,'Color',M2Scolor.orange) , grid on
% plot(refSet(multMatch_idx,1),targetSet(multMatch_idx,1)-refSet(multMatch_idx,1),'o','MarkerSize',5,'Color',M2Scolor.lblue) % with multiple matches
% xlabel('RT ref (minutes)'),ylabel('RTdist (minutes)') ; axis tight %xlim([0,max(refSet(:,1))])
% 
% % PLOT: MZdist vs MZref    
% subplot(1,3,2)
% plot([0;max(refSet(:,2))],[multThresh.MZ_intercept(1);max(refSet(:,2))*multThresh.MZ_slope(1)+multThresh.MZ_intercept(1)],'-','LineWidth',2,'Color',M2Scolor.dblue), hold on
% plot([0;max(refSet(:,2))],[multThresh.MZ_intercept(2);max(refSet(:,2))*multThresh.MZ_slope(2)+multThresh.MZ_intercept(2)],'-','LineWidth',2,'Color',M2Scolor.dblue)
% M2S_plotLinkedPoints(refSet(:,2),targetSet(:,2)-refSet(:,2),Xr_connIdx_i,'ok',':',M2Scolor.lgrey,{'MZ ref','MZdist'}) 
% M2S_plotLinkedPoints(refSet(:,2),targetSet(:,2)-refSet(:,2),Xt_connIdx_i,'ok',':',M2Scolor.orange,{'MZ ref','MZdist'}) 
% plot(refSet(FIdistOK_01==0,2),targetSet(FIdistOK_01==0,2)-refSet(FIdistOK_01==0,2),'.','MarkerSize',8,'Color',M2Scolor.orange) , grid on
% plot(refSet(multMatch_idx,2),targetSet(multMatch_idx,2)-refSet(multMatch_idx,2),'o','MarkerSize',5,'Color',M2Scolor.lblue) % with multiple matches
% xlabel('MZ ref (m/z units)'),ylabel('MZdist (m/z units)');hold on; axis tight
% 
% % PLOT: log10FIdist vs FIref         
% subplot(1,3,3)
% plot([min(log10(refSet(:,3)));max(log10(refSet(:,3)))],[ylimFImin(1);ylimFImin(2)],'-','LineWidth',2,'Color',M2Scolor.dblue), hold on
% plot([min(log10(refSet(:,3)));max(log10(refSet(:,3)))],[ylimFImax(1);ylimFImax(2)],'-','LineWidth',2,'Color',M2Scolor.dblue)
% 
% M2S_plotLinkedPoints(log10(refSet(:,3)),log10(targetSet(:,3))-log10(refSet(:,3)),Xr_connIdx_i,'ok',':',M2Scolor.lgrey,{'log10FI ref','log10FIdist'}), hold on
% M2S_plotLinkedPoints(log10(refSet(:,3)),log10(targetSet(:,3))-log10(refSet(:,3)),Xt_connIdx_i,'ok',':',M2Scolor.orange,{'log10FI ref','log10FIdist'}), hold on
% plot([min(log10(refSet(:,3)));max(log10(refSet(:,3)))],[0,0],'k')
% plot(log10(refSet(FIdistOK_01==0,3)),log10(targetSet(FIdistOK_01==0,3))-log10(refSet(FIdistOK_01==0,3)),'.','MarkerSize',8,'Color',M2Scolor.orange)
% plot(log10(refSet(multMatch_idx,3)),log10(targetSet(multMatch_idx,3))-log10(refSet(multMatch_idx,3)),'o','MarkerSize',5,'Color',M2Scolor.lblue) % with multiple matches
% xlabel('log10FI ref'),ylabel('log10FIdist') ; axis tight
% drawnow
% 



