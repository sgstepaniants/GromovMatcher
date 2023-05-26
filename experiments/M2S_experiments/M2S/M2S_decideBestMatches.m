%% Do recursive CCs
% The function v2 works, but I created on 200502 function v3 to accelerate.
% NOTE: I think the function v1 also works!

%function [eL,eL_INFO,CC_G1]= matchRTMZ_selectMatchesWithScores_v1(Xref_connected_idx,Xtarget_connected_idx,RTMZdist_medNeigh_to_TRpair_scaled);

function [eL,eL_INFO,CC_G1]= M2S_decideBestMatches(refSet, targetSet, Xr_connIdx,Xt_connIdx, penaltyScores, plotNetOrNot, minNrNodesToPlot)

fprintf('\n\n Function M2S_decideBestMatches\n')
fprintf(' Calculating best matches in multiple match clusters using penalisation scores\n')

% Clusters with >= minNrNodesToPlot number of nodes are plotted to show iterative process
if nargin <=5
    plotNetOrNot = 1;
    minNrNodesToPlot = 1000000; % Do not plot. 
elseif nargin <=6
    minNrNodesToPlot = 1000000; % Do not plot. 
end

MZRTstr_Ref = M2S_createLabelMZRT('REF',refSet(:,2),refSet(:,1));
MZRTstr_Target = M2S_createLabelMZRT('TARGET',targetSet(:,2),targetSet(:,1));

eL_all = [table([MZRTstr_Ref,MZRTstr_Target],'VariableNames',{'EndNodes'}), ...
    table((1:length(Xr_connIdx))',penaltyScores,'VariableNames',{'rowNrInMatchedSets','matchScore'})];

%% Create a GRAPH 
G1 = digraph(eL_all);
G1forCC = graph(eL_all);

refString = G1.Edges.EndNodes{1,1}(1:3);
G1.Nodes.refNode = double(contains(G1.Nodes.Name,refString));

if plotNetOrNot == 1 % plot with less information
    load ('M2ScolorScheme.mat')
    M2S_figureH(0.65,0.9)
    set(gcf,'Name','All matches (edges) colored by penalty scores. Metabolomic features (nodes) of Reference in black, Target features in red');
    movegui(gcf,'center')
    f1 = plot(G1,'interpreter','None','EdgeCData',G1.Edges.matchScore,'LineWidth',2); 
    colorbar('Location','southoutside'), xlabel('Penalisation scores')
    highlight(f1,find(G1.Nodes.refNode==1),'NodeColor','k','MarkerSize',3);
    highlight(f1,find(G1.Nodes.refNode==0),'NodeColor','b','MarkerSize',3);
    set(gca,'XTick',[], 'YTick', []); axis tight        
    colormap(M2Scolormap);
    drawnow;
elseif plotNetOrNot == 2 % plot with more information 
    load ('M2ScolorScheme.mat')
    M2S_figureH(0.65,0.9)
    set(gcf,'Name','All matches (edges) colored by penalty scores. Metabolomic features (nodes) of Reference in black, Target features in red')
    movegui(gcf,'center')
    f1 = plot(G1,'NodeLabel',G1.Nodes.Name,'EdgeLabel',G1.Edges.matchScore,'interpreter','None','EdgeCData',G1.Edges.matchScore,'LineWidth',2); 
    colorbar('Location','southoutside'), xlabel('Penalisation scores')
    highlight(f1,1:size(G1.Nodes,1));
    highlight(f1,find(G1.Nodes.refNode==1),'NodeColor','k','MarkerSize',3);
    highlight(f1,find(G1.Nodes.refNode==0),'NodeColor','b','MarkerSize',3);
    set(gca,'XTick',[], 'YTick', []); axis tight
    colormap(M2Scolormap);
    drawnow;
end
    
  
%% Find connected components in graph G1forCC
CC_G1=[];
CC_G1.CCs = (conncomp(G1forCC,'OutputForm','cell'))'; % Original CC needed later!
CC_G1.nrNodes_inCC = cellfun(@length,CC_G1.CCs);

%% Define the conncomp of each node
G1.Nodes.inCCnr = (conncomp(G1forCC))';

%% Find which matches make part of each CC
%% Additionally, for each row of eL_all, find the CC its members are in
eL_all_inCCnr = NaN(size(eL_all.EndNodes,1),1);
[~,idx11,idx12] = intersect(eL_all.EndNodes(:,1),G1.Nodes.Name,'stable');% This will find most of them
eL_all_inCCnr(idx11) = G1.Nodes.inCCnr(idx12);
notYetDefined = find(isnan(eL_all_inCCnr));
wb1 = waitbar(0,'Finding the cluster number of each feature in a multiple match','Name','Info');
for n=1:length(notYetDefined)% this will find the rest of them
    waitbar(n/length(notYetDefined),wb1,'Finding the cluster number of each feature in a multiple match','Name','Info');
     idxInNodes = (string(G1.Nodes.Name) == eL_all.EndNodes(notYetDefined(n),1));
     eL_all_inCCnr(notYetDefined(n)) = G1.Nodes.inCCnr(idxInNodes);
end    
waitbar(1,wb1,'Done!','Name','Info');
pause(1)
close(wb1);

eL_all_nrEdgesInSameCC = NaN(size(eL_all.matchScore));
for rowInEdges = 1:size(eL_all,1)
    eL_all_nrEdgesInSameCC(rowInEdges,1) = nansum(eL_all_inCCnr == eL_all_inCCnr(rowInEdges));
end
eL_all.inCCnr = eL_all_inCCnr;        
eL_all.nrEdgesInSameCC=eL_all_nrEdgesInSameCC;


%% make recursive connected components
eL_Best=[];
eL_Best_nrIterations=[];

CCs_NOTforIterativeProcess = find(CC_G1.nrNodes_inCC==2); 
CCs_forIterativeProcess = find(CC_G1.nrNodes_inCC>2); % Make it only on CCs with >2 nodes

wb2 = waitbar(0,'Calculating clusters with multiple matches','Name','Info');
fprintf('\n\n')

for iterativeCCnr = 1:length(CCs_forIterativeProcess)
    waitbar(iterativeCCnr/length(CCs_forIterativeProcess),wb2,'Calculating best matches in clusters with multiple matches','Name','Info');
    %fprintf('Iterating cluster (with multiple matches) number %d\n',CCs_forIterativeProcess(iterativeCCnr))
    if exist('f','var');close(f); clear f;end
    nrIterations = 0;
    now_eL = eL_all(eL_all.inCCnr == CCs_forIterativeProcess(iterativeCCnr),:);
    nowG = digraph(now_eL);
    
    % if there are minNrNodesToPlot in the node, plot the CC
    if size(nowG.Nodes,1)>=minNrNodesToPlot && nrIterations ==0
        disp('interesting CC to visualise')     
        f = M2S_figureH(0.3,0.5);
        plotOrNot = 1;
    elseif size(nowG.Nodes,1)<minNrNodesToPlot && nrIterations ==0
    %elseif size(nowG.Nodes,1)<minNrNodesToPlot && nrIterations ~=0

        plotOrNot = 0;
        %close(gcf)
        if exist('f','var') ;close(f); clear f;end
    end
    
    if plotOrNot
        load ('M2ScolorScheme.mat')
%     p = plot(nowG,'NodeLabel',nowG.Nodes.Name,'EdgeLabel',nowG.Edges.inCCnr,'interpreter','None','EdgeCData',nowG.Edges.matchScore,'LineWidth',2);
    p = plot(nowG,'NodeLabel',nowG.Nodes.Name,'EdgeLabel',nowG.Edges.matchScore,'interpreter','None','EdgeCData',nowG.Edges.matchScore,'LineWidth',2);
    nowG.Nodes.XData = p.XData'; nowG.Nodes.YData = p.YData'; axislim = axis;
    axis(axislim);colorbar('Location','southoutside');xlabel('Penalisation scores'); title(['Cluster number ',num2str(CCs_forIterativeProcess(iterativeCCnr))]);set(gca,'XTick',[], 'YTick', [])
    colormap(M2Scolormap);
    drawnow; pause(2)
    end % IF
    while size(nowG.Edges,1)>0              
        if plotOrNot == 1 && nrIterations > 0
        %clf
        p = plot(nowG,'NodeLabel',nowG.Nodes.Name,'XData',nowG.Nodes.XData,'YData',nowG.Nodes.YData,'EdgeLabel',nowG.Edges.matchScore,'interpreter','None','EdgeCData',nowG.Edges.matchScore,'LineWidth',2);
        axis(axislim);colorbar('Location','southoutside');xlabel('Penalisation scores'); title(['Cluster number ',num2str(CCs_forIterativeProcess(iterativeCCnr))]);set(gca,'XTick',[], 'YTick', [])
        colormap(M2Scolormap);
        drawnow; pause(1)
        end 
        nrIterations = nrIterations+1;
        if size(nowG.Edges,1) == 1
            eL_Best = [eL_Best;nowG.Edges(1,:)];
            nowG = rmnode(nowG,nowG.Edges.EndNodes(1,:));
            %now_eL=[];
            eL_Best_nrIterations = [eL_Best_nrIterations;nrIterations];
        else
            [minScore,minScore_idx]=min(nowG.Edges.matchScore);
            eL_Best = [eL_Best;nowG.Edges(minScore_idx,:)];% collect the good edge
            eL_Best_nrIterations = [eL_Best_nrIterations;nrIterations];
            %% the nodes in the best edge are found. All edges containing those nodes are deleted
            % find the best nodes to delete from the nowG graph
            bestNodesTemp = nowG.Edges.EndNodes(minScore_idx,:);
            nowG = rmnode(nowG,bestNodesTemp); % by index
            if plotOrNot
                p = plot(nowG,'NodeLabel',nowG.Nodes.Name,'XData',nowG.Nodes.XData,'YData',nowG.Nodes.YData,'EdgeLabel',nowG.Edges.matchScore,'interpreter','None','EdgeCData',nowG.Edges.matchScore,'LineWidth',2);
                axis(axislim);colorbar('Location','southoutside');xlabel('Penalisation scores'); title(['Cluster number ',num2str(CCs_forIterativeProcess(iterativeCCnr))]);set(gca,'XTick',[], 'YTick', [])
                colormap(M2Scolormap);
                drawnow; pause(1)
            end % IF
        end
    end
    
end
waitbar(1,wb2,'Done!','Name','Info');
pause(1)
close(wb2);% status bar
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NEW STUFF
eL_Best_singleMatches = eL_all(eL_all.nrEdgesInSameCC == 1,:);
eL_Best_singleMatches.nrIterations = zeros(size(eL_Best_singleMatches,1),1);
eL_Best.nrIterations = ones(size(eL_Best,1),1);
eL_Best = [eL_Best_singleMatches;eL_Best];
eL_Best = sortrows(eL_Best,'inCCnr');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% add nr of recursive iterations until it decided an edge is good
%eL_Best.nrIterations = eL_Best_nrIterations;
% If a row is not "best", then it is "worst":
eL_Worst_rowNr = setdiff((1:length(Xr_connIdx))',eL_Best.rowNrInMatchedSets);
eL_Worst = table(Xr_connIdx(eL_Worst_rowNr),...
                     Xt_connIdx(eL_Worst_rowNr),...
                     eL_Worst_rowNr,...
                     penaltyScores(eL_Worst_rowNr),...
                     eL_all.inCCnr(eL_Worst_rowNr),...
                     'VariableNames',{'Xr_connIdx','Xt_connIdx','rowNrInMatchedSets','matchScore','inCCnr'});


                 % add nrEdgesInSameCC to the the eL_Worst   
eL_Worst_nrEdgesInSameCC = NaN(size(eL_Worst,1),1);
for edgeNr = 1:size(eL_Worst,1)
    tempIdx = find(eL_Best.inCCnr== eL_Worst.inCCnr(edgeNr));
    eL_Worst_nrEdgesInSameCC(edgeNr) = eL_Best.nrEdgesInSameCC(tempIdx(1));
end
eL_Worst.nrEdgesInSameCC = eL_Worst_nrEdgesInSameCC;
eL_Worst.nrIterations = NaN(size(eL_Worst,1),1);


%% create a final eL_Best
eL_Best(:,1) = [];
eL_Best = [table(Xr_connIdx(eL_Best.rowNrInMatchedSets),Xt_connIdx(eL_Best.rowNrInMatchedSets),'VariableNames',{'Xr_connIdx','Xt_connIdx'}), eL_Best];
% 
% eL_Best.Xref_connected_idx = Xref_connected_idx(eL_Best.rowNrInMatchedSets);
% eL_Best.Xtarget_connected_idx = Xtarget_connected_idx(eL_Best.rowNrInMatchedSets);

% eL_Best.EndNodes(:,1) = MZRT_string_Ref(eL_Best.Xref_connected_idx);
% eL_Best.EndNodes(:,2) = MZRT_string_Target(eL_Best.Xtarget_connected_idx);


%% add descriptive vectors
eL= [eL_Best; eL_Worst];
eL.is_Best = [ones(size(eL_Best,1),1) ; zeros(size(eL_Worst,1),1)];
eL.is_Worst = [zeros(size(eL_Best,1),1) ; ones(size(eL_Worst,1),1)];
%eL.is_GoodClass = [double(eL_Best.nrEdgesInSameCC>1) ; zeros(size(eL_Worst,1),1)];
%eL.is_BadClass = [zeros(size(eL_Best,1),1) ; ones(size(eL_Worst,1),1)];
%eL.is_toPredict = [ones(size(eL_Best,1),1) ; zeros(size(eL_Worst,1),1)];

eL = sortrows(eL,'rowNrInMatchedSets');
eL = [table(MZRTstr_Ref),table(MZRTstr_Target),eL];

eL_INFO{1,1} = '- This table contains matches between Ref and Target. It is in the same order as refSet, targetSet, Xr_connIdx and Xt_connIdx';
eL_INFO{2,1} = 'matchScore is the penalty score for the match. The smaller the better';
eL_INFO{3,1} = '- inCCnr is the initial multiple match cluster (graph connected component, or CC). That conncomp had "nrEdgesInSameCC" edges';
eL_INFO{4,1} = '- This match was in a cluster. The nrEdgesInSameCC shows the number of matches in the same CC as this edge. If ==1 means it was'; 
eL_INFO{5,1} = 'a single match, while >1 means there were multiple matches in the same CC';
eL_INFO{6,1} = '- nrIterations shows the number of iterations needed until this match was decided it is a good one. If ==0 means there';
eL_INFO{7,1} = 'were no other possible matches for the two features in it. NaN means it was never selected as a good match.';
eL_INFO{8,1} = '- is_Best indicates if this match was chosen as a good one during the iterative scores selection (even if not in the first iteration)';
eL_INFO{9,1} = '- is_Worst indicates that this match was not selected after the iterative process.';



