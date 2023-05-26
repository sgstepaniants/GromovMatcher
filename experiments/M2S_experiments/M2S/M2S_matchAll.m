function [refSet,targetSet,Xr_connIdx,Xt_connIdx,opt]=M2S_matchAll(refFeatures,targetFeatures,multThresh,FIadjustMethod,plotType)
%% M2S_matchAll
% Function to calculate matches between two LCMS datasets according to 
% specified RT/MZ/log10FI distances between the features in two datasets.
%
% Background:
% Matches are sets of two molecular features that are less distant than
% defined RT (min), MZ (ppm), log10FI thresholds, meaning the two features
% may potentially be the same molecular species. It may find multiple
% matches for the same feature.
%
%
% INPUT:
% refFeatures, targetFeatures: two matrices of [RT,MZ,FI] 
% RT is a column of retention time in minutes
% MZ is a column of m/z in m/z units
% FI is the feature intensity (peak area)
% 
% multThresh: values for intercept and slope for two threshold lines in
% each of the three dimensions. Default as below:
%
% multThresh.MZ_intercept = 0.025;
% multThresh.MZ_slope = 0;
% multThresh.RT_intercept = 1;
% multThresh.RT_slope = 0;
% multThresh.log10FI_intercept = 1000; 
% multThresh.log10FI_slope = 0;
%
% FIadjustMethod: method to adjust the FI of target to reference {'none','median','regression'}
% The 'none' (default) method does not do anything
% The 'median' method simply subtracts the median of ref to target, including all multiple matches.
% The 'regression' method adjust target to ref using linear regression, including all multiple matches.
%
% plotType: 0 - no plots; 1 - (default) plots of feature intensity and of
% matches; 2 - plots of feature intensity, multiple matches, and network of matches
%
% OUTPUT:
% refSet, targetSet: matched sets, with non-unique features in each set.
% Xr_connIdx, Xt_connIdx: non-unique indices (in refFeatures, targetFeatures)
% of the features in each match.
%
% opt: The options in use.
%
% EXAMPLES
% [refSet,targetSet,Xr_connIdx,Xt_connIdx, opt]=M2S_matchAll(refFeatures,targetFeatures);
% [refSet,targetSet,Xr_connIdx,Xt_connIdx,opt]=M2S_matchAll(refFeatures,targetFeatures,definedThresholds,'median',2);
%
% M2S toolbox to match LCMS metabolomics features of untargeted datasets.
% *** Rui Climaco Pinto ***
% *** Imperial College London, 2021 ***





if nargin == 2
    multThresh.MZ_intercept = 0.02;
    multThresh.MZ_slope = 0;
    multThresh.RT_intercept = abs(max([refFeatures(:,1);targetFeatures(:,1)])-min([targetFeatures(:,1);refFeatures(:,1)]));
    multThresh.RT_slope = 0;
    multThresh.log10FI_intercept = 1000; 
    multThresh.log10FI_slope = 0;
    multThresh.selfMatchingOrNot = 0;% May exist or not
    FIadjustMethod = 'none';
    plotType = 1;
elseif nargin == 3    
    FIadjustMethod = 'none';
    plotType = 1;
elseif nargin == 4
    %FIadjustMethod = 'median';
    plotType = 1;
end

opt.multThresh = multThresh;
opt.FIadjustMethod = FIadjustMethod;
RTref = refFeatures(:,1);
MZref = refFeatures(:,2);
% FIref = refFeatures(:,3);

RTtarget = targetFeatures(:,1);
MZtarget = targetFeatures(:,2);
% FItarget = targetFeatures(:,3);

fprintf('\n *** Executing function M2S_matchAll ***\n')
fprintf(' *** Matching all features within thresholds ***\n')

% If thresholds contain only one digit, make it a vector with two symmetrical values
if length(multThresh.MZ_intercept) ==1
    multThresh.MZ_intercept = [-1*abs(multThresh.MZ_intercept), abs(multThresh.MZ_intercept)];
end
if length(multThresh.RT_intercept)==1
    multThresh.RT_intercept = [-1*abs(multThresh.RT_intercept), abs(multThresh.RT_intercept)];
end
if length(multThresh.MZ_slope)==1
    multThresh.MZ_slope = [-1*abs(multThresh.MZ_slope), abs(multThresh.MZ_slope)];
end
if length(multThresh.RT_slope)==1
    multThresh.RT_slope = [-1*abs(multThresh.RT_slope), abs(multThresh.RT_slope)];
end
if length(multThresh.log10FI_intercept) ==1
    multThresh.log10FI_intercept = [-1*abs(multThresh.log10FI_intercept), abs(multThresh.log10FI_intercept)];
end
if length(multThresh.log10FI_slope) ==1
    multThresh.log10FI_slope = [-1*abs(multThresh.log10FI_slope), abs(multThresh.log10FI_slope)];
end


%% Calculate the individual RTthresh_intercept and MZthresh 
RTthresh = repmat(multThresh.RT_intercept,length(RTref),1) + [multThresh.RT_slope(1)*RTref , multThresh.RT_slope(2)*RTref]; 
MZthresh = repmat(multThresh.MZ_intercept,length(MZref),1) + [multThresh.MZ_slope(1)*MZref , multThresh.MZ_slope(2)*MZref]; 

%% For each ref feature, find target features within RT and MZ threshold distances
Xr_connIdx_i = [];
Xt_connIdx_i = []; %NaN(5000,1);
wb1 = waitbar(0,'Calculating all matches within thresholds','Name','Info');
for refFeatureNr = 1 : length(RTref)
    waitbar(refFeatureNr/length(RTref),wb1,'Calculating all matches within thresholds','Name','Info');
    RTdist_individual = RTtarget - RTref(refFeatureNr);
    MZdist_individual = MZtarget - MZref(refFeatureNr);
        
    RTdistOK =  (RTdist_individual > RTthresh(refFeatureNr,1) ) &  (RTdist_individual < RTthresh(refFeatureNr,2));
    MZdistOK =  (MZdist_individual > MZthresh(refFeatureNr,1) ) &  (MZdist_individual < MZthresh(refFeatureNr,2));
    RTMZdistOK = RTdistOK .* MZdistOK;

    Xt_connIdx_i = [Xt_connIdx_i; find(RTMZdistOK)];
    Xr_connIdx_i =[Xr_connIdx_i; repmat(refFeatureNr,sum(RTMZdistOK),1)];
end
waitbar(1,wb1,'Done!','Name','Info');
pause(1)
close(wb1);


%% If matching a dataset to itself, need to delete self and double matches:
if isfield(opt.multThresh,'selfMatchingOrNot')
    if opt.multThresh.selfMatchingOrNot == 1
        % Delete self matches
        del_idx = (Xr_connIdx_i == Xt_connIdx_i);
        Xr_connIdx_i(del_idx) = [];
        Xt_connIdx_i(del_idx) = [];
        % Delete double matches
        del_idx2 = find(Xr_connIdx_i>Xt_connIdx_i);
        Xr_connIdx_i(del_idx2) = [];
        Xt_connIdx_i(del_idx2) = [];
    end
end


%% For the matched features, select the ones inside log10FI threshold
% Input a value of log10FI for threhsold. Use a large one, e.g. 100 for no threshold.

refSet_i = refFeatures(Xr_connIdx_i,:);
targetSet_i = targetFeatures(Xt_connIdx_i,:);

%% First homogenise target FI in relation to FIreference
if plotType == 0
    [targetSet_i(:,3)] = M2S_adjustFImed(refSet_i(:,3),targetSet_i(:,3),FIadjustMethod,0);
else
    [targetSet_i(:,3)] = M2S_adjustFImed(refSet_i(:,3),targetSet_i(:,3),FIadjustMethod,1);
end

%% Calculate distances for FI in the features matched by RT and MZ only
log10FIdist_i = log10(targetSet_i(:,3)) - log10(refSet_i(:,3));

log10FIthresh_i = repmat(multThresh.log10FI_intercept,length(log10FIdist_i),1) + ...
    [multThresh.log10FI_slope(1)*log10(refSet_i(:,3)) , multThresh.log10FI_slope(2)*log10(refSet_i(:,3))]; 

% select the ones that are within FI threshold limits. Boolean vector: 0 is bad, 1 is good
FIdistOK_01 =  (log10FIdist_i > log10FIthresh_i(:,1) ) &  (log10FIdist_i < log10FIthresh_i(:,2));

%% *** Get the final results ***

refSet = refSet_i(FIdistOK_01,:);
targetSet = targetSet_i(FIdistOK_01,:);

Xr_connIdx = Xr_connIdx_i(FIdistOK_01);
Xt_connIdx = Xt_connIdx_i(FIdistOK_01);


%% PLOTS
if isempty(Xr_connIdx)
    disp('THERE ARE NO HITS')
else
    if plotType > 0
        load ('M2ScolorScheme.mat');  
        % The following is needed for subplot 3 (log10FI)

        % If there are no features with log10FI lower than lower threshold
        if sum(  (log10(targetSet_i(:,3))-log10(refSet_i(:,3)))<log10FIthresh_i(:,1)   ) == 0
            ylimFImin = repmat(min( log10(targetSet_i(:,3))-log10(refSet_i(:,3))),1,2);
        else
            ylimFImin = [multThresh.log10FI_intercept(1) + min(log10(refSet_i(:,3)))*multThresh.log10FI_slope(1) ;multThresh.log10FI_intercept(1) + max(log10(refSet_i(:,3)))*multThresh.log10FI_slope(1)];
        end

        % If there are no features with log10FI higher than high threshold
        if sum(log10(targetSet_i(:,3))-log10(refSet_i(:,3))>log10FIthresh_i(:,2)) == 0
            ylimFImax = repmat(max( log10(targetSet_i(:,3))-log10(refSet_i(:,3))),1,2);
        else
            ylimFImax = [multThresh.log10FI_intercept(2) + min(log10(refSet_i(:,3)))*multThresh.log10FI_slope(2);multThresh.log10FI_intercept(2) + max(log10(refSet_i(:,3)))*multThresh.log10FI_slope(2) ];
        end
        
      % Find the multiple matches to plot (indices in refSet_i and targetSet_i)
        
        table_ref_idx = tabulate(Xr_connIdx_i);
        table_ref_idx(table_ref_idx(:,2) <= 1,:) = [];
        table_target_idx = tabulate(Xt_connIdx_i);
        table_target_idx(table_target_idx(:,2) <= 1,:) = [];
        multMatch_idx = [];
        for r = 1:size(table_ref_idx,1)
            multMatch_idx = [multMatch_idx;find(Xr_connIdx_i == table_ref_idx(r,1))];
        end
        for t = 1:size(table_target_idx,1)
            multMatch_idx = [multMatch_idx;find(Xt_connIdx_i == table_target_idx(t,1))];
        end
        multMatch_idx = unique(multMatch_idx);
        
        
        %% PLOT

        if plotType == 1

            M2S_figureH(0.8,0.5);
            set(gcf,'Name','Black dots represent matches within threshold, blue circles highlight multiple matches, orange dots are matches outside log10FI threshold.');
            % PLOT: RTdist vs RTref
            subplot(1,3,1)
%             fill([0; max(refSet_i(:,1));  max(refSet_i(:,1)); 0],...
%                 [multThresh.RT_intercept(1) ;multThresh.RT_intercept(1) + max(refSet_i(:,1))*multThresh.RT_slope(1);multThresh.RT_intercept(2) + max(refSet_i(:,1))*multThresh.RT_slope(2);multThresh.RT_intercept(2)],'b','FaceAlpha',0.1), hold on
            plot([0;max(refSet_i(:,1))],[multThresh.RT_intercept(1);max(refSet_i(:,1))*multThresh.RT_slope(1)+multThresh.RT_intercept(1)],'-','LineWidth',2,'Color',M2Scolor.dblue), hold on
%             plot([0;max(refSet_i(:,1))],[multThresh.RT_intercept(1);max(refSet_i(:,1))*multThresh.RT_slope(1)+multThresh.RT_intercept(1)],'-b','LineWidth',2), hold on
            plot([0;max(refSet_i(:,1))],[multThresh.RT_intercept(2);max(refSet_i(:,1))*multThresh.RT_slope(2)+multThresh.RT_intercept(2)],'-','LineWidth',2,'Color',M2Scolor.dblue)
            plot(refSet_i(FIdistOK_01==1,1),targetSet_i(FIdistOK_01==1,1)-refSet_i(FIdistOK_01==1,1),'.k','MarkerSize',11) , grid on
            plot(refSet_i(FIdistOK_01==0,1),targetSet_i(FIdistOK_01==0,1)-refSet_i(FIdistOK_01==0,1),'.','MarkerSize',11,'Color',M2Scolor.orange) , grid on
            plot(refSet_i(multMatch_idx,1),targetSet_i(multMatch_idx,1)-refSet_i(multMatch_idx,1),'o','MarkerSize',5,'Color',M2Scolor.lblue) % with multiple matches
            xlabel('RT ref (minutes)'),ylabel('RTdist (minutes)') ; axis tight 

            % PLOT: MZdist vs MZref    
            subplot(1,3,2)
%             fill([0; max(refSet_i(:,2));  max(refSet_i(:,2)); 0],...
%                 [multThresh.MZ_intercept(1) ;multThresh.MZ_intercept(1) + max(refSet_i(:,2))*multThresh.MZ_slope(1);multThresh.MZ_intercept(2) + max(refSet_i(:,2))*multThresh.MZ_slope(2);multThresh.MZ_intercept(2)],'b','FaceAlpha',0.1), hold on
            plot([0;max(refSet_i(:,2))],[multThresh.MZ_intercept(1);max(refSet_i(:,2))*multThresh.MZ_slope(1)+multThresh.MZ_intercept(1)],'-','LineWidth',2,'Color',M2Scolor.dblue), hold on
            plot([0;max(refSet_i(:,2))],[multThresh.MZ_intercept(2);max(refSet_i(:,2))*multThresh.MZ_slope(2)+multThresh.MZ_intercept(2)],'-','LineWidth',2,'Color',M2Scolor.dblue)
            plot(refSet_i(FIdistOK_01==1,2),targetSet_i(FIdistOK_01==1,2)-refSet_i(FIdistOK_01==1,2),'.k','MarkerSize',11) , grid on
            plot(refSet_i(FIdistOK_01==0,2),targetSet_i(FIdistOK_01==0,2)-refSet_i(FIdistOK_01==0,2),'.','MarkerSize',11,'Color',M2Scolor.orange) , grid on
            plot(refSet_i(multMatch_idx,2),targetSet_i(multMatch_idx,2)-refSet_i(multMatch_idx,2),'o','MarkerSize',5,'Color',M2Scolor.lblue) % with multiple matches
            xlabel('MZ ref (m/z units)'),ylabel('MZdist (m/z units)');hold on; axis tight

            % PLOT: log10FIdist vs FIref
            subplot(1,3,3)

%             fill([min(log10(refSet_i(:,3))); max(log10(refSet_i(:,3)));  max(log10(refSet_i(:,3)));  min(log10(refSet_i(:,3)))],...
%                 [ylimFImin(1) ;ylimFImin(2);ylimFImax(2);ylimFImax(1) ],'b','FaceAlpha',0.1), hold on
            plot([min(log10(refSet_i(:,3)));max(log10(refSet_i(:,3)))],[ylimFImin(1);ylimFImin(2)],'-','LineWidth',2,'Color',M2Scolor.dblue), hold on
            plot([min(log10(refSet_i(:,3)));max(log10(refSet_i(:,3)))],[ylimFImax(1);ylimFImax(2)],'-','LineWidth',2,'Color',M2Scolor.dblue)

            plot(log10(refSet_i(FIdistOK_01==1,3)),log10(targetSet_i(FIdistOK_01==1,3))-log10(refSet_i(FIdistOK_01==1,3)),'.k','MarkerSize',11), grid on
            plot([min(log10(refSet_i(:,3)));max(log10(refSet_i(:,3)))],[0,0],'k')
            plot(log10(refSet_i(FIdistOK_01==0,3)),log10(targetSet_i(FIdistOK_01==0,3))-log10(refSet_i(FIdistOK_01==0,3)),'ok','MarkerSize',3)
            plot(log10(refSet_i(FIdistOK_01==0,3)),log10(targetSet_i(FIdistOK_01==0,3))-log10(refSet_i(FIdistOK_01==0,3)),'.','MarkerSize',8,'Color',M2Scolor.orange)
            plot(log10(refSet_i(multMatch_idx,3)),log10(targetSet_i(multMatch_idx,3))-log10(refSet_i(multMatch_idx,3)),'o','MarkerSize',5,'Color',M2Scolor.lblue) % with multiple matches
            xlabel('log10FI ref'),ylabel('log10FIdist') ; axis tight

            drawnow
            elseif plotType == 2 % SHOW MULTIPLE MATCHES CONNECTIONS

            M2S_figureH(0.8,0.5);
            set(gcf,'Name','Black dots represent matches within threshold, blue squares highlight multiple matches, orange dots are matches outside log10FI threshold. Dotted lines: blue when a reference feature matches two or more target features; Orange for the opposite.');
            % PLOT: RTdist vs RTref
            subplot(1,3,1)
            plot([0;max(refSet_i(:,1))],[multThresh.RT_intercept(1);max(refSet_i(:,1))*multThresh.RT_slope(1)+multThresh.RT_intercept(1)],'-','LineWidth',2,'Color',M2Scolor.dblue), hold on
            plot([0;max(refSet_i(:,1))],[multThresh.RT_intercept(2);max(refSet_i(:,1))*multThresh.RT_slope(2)+multThresh.RT_intercept(2)],'-','LineWidth',2,'Color',M2Scolor.dblue)
            M2S_plotLinkedPoints(refSet_i(:,1),targetSet_i(:,1)-refSet_i(:,1),Xr_connIdx_i,'ok',':',M2Scolor.lgrey,[]) 
            M2S_plotLinkedPoints(refSet_i(:,1),targetSet_i(:,1)-refSet_i(:,1),Xt_connIdx_i,'ok',':',M2Scolor.orange,[]) 
            plot(refSet_i(FIdistOK_01==0,1),targetSet_i(FIdistOK_01==0,1)-refSet_i(FIdistOK_01==0,1),'.','MarkerSize',8,'Color',M2Scolor.orange) , grid on
            plot(refSet_i(multMatch_idx,1),targetSet_i(multMatch_idx,1)-refSet_i(multMatch_idx,1),'o','MarkerSize',5,'Color',M2Scolor.lblue) % with multiple matches
            xlabel('RT ref (minutes)'),ylabel('RTdist (minutes)') ; axis tight %xlim([0,max(refSet_i(:,1))])

            % PLOT: MZdist vs MZref    
            subplot(1,3,2)
            plot([0;max(refSet_i(:,2))],[multThresh.MZ_intercept(1);max(refSet_i(:,2))*multThresh.MZ_slope(1)+multThresh.MZ_intercept(1)],'-','LineWidth',2,'Color',M2Scolor.dblue), hold on
            plot([0;max(refSet_i(:,2))],[multThresh.MZ_intercept(2);max(refSet_i(:,2))*multThresh.MZ_slope(2)+multThresh.MZ_intercept(2)],'-','LineWidth',2,'Color',M2Scolor.dblue)
            M2S_plotLinkedPoints(refSet_i(:,2),targetSet_i(:,2)-refSet_i(:,2),Xr_connIdx_i,'ok',':',M2Scolor.lgrey,{'MZ ref','MZdist'}) 
            M2S_plotLinkedPoints(refSet_i(:,2),targetSet_i(:,2)-refSet_i(:,2),Xt_connIdx_i,'ok',':',M2Scolor.orange,{'MZ ref','MZdist'}) 
            plot(refSet_i(FIdistOK_01==0,2),targetSet_i(FIdistOK_01==0,2)-refSet_i(FIdistOK_01==0,2),'.','MarkerSize',8,'Color',M2Scolor.orange) , grid on
            plot(refSet_i(multMatch_idx,2),targetSet_i(multMatch_idx,2)-refSet_i(multMatch_idx,2),'o','MarkerSize',5,'Color',M2Scolor.lblue) % with multiple matches
            xlabel('MZ ref (m/z units)'),ylabel('MZdist (m/z units)');hold on; axis tight

            % PLOT: log10FIdist vs FIref         
            subplot(1,3,3)
            plot([min(log10(refSet_i(:,3)));max(log10(refSet_i(:,3)))],[ylimFImin(1);ylimFImin(2)],'-','LineWidth',2,'Color',M2Scolor.dblue), hold on
            plot([min(log10(refSet_i(:,3)));max(log10(refSet_i(:,3)))],[ylimFImax(1);ylimFImax(2)],'-','LineWidth',2,'Color',M2Scolor.dblue)

            M2S_plotLinkedPoints(log10(refSet_i(:,3)),log10(targetSet_i(:,3))-log10(refSet_i(:,3)),Xr_connIdx_i,'ok',':',M2Scolor.lgrey,{'log10FI ref','log10FIdist'}), hold on
            M2S_plotLinkedPoints(log10(refSet_i(:,3)),log10(targetSet_i(:,3))-log10(refSet_i(:,3)),Xt_connIdx_i,'ok',':',M2Scolor.orange,{'log10FI ref','log10FIdist'}), hold on
            plot([min(log10(refSet_i(:,3)));max(log10(refSet_i(:,3)))],[0,0],'k')
            plot(log10(refSet_i(FIdistOK_01==0,3)),log10(targetSet_i(FIdistOK_01==0,3))-log10(refSet_i(FIdistOK_01==0,3)),'.','MarkerSize',8,'Color',M2Scolor.orange)
            plot(log10(refSet_i(multMatch_idx,3)),log10(targetSet_i(multMatch_idx,3))-log10(refSet_i(multMatch_idx,3)),'o','MarkerSize',5,'Color',M2Scolor.lblue) % with multiple matches
            xlabel('log10FI ref'),ylabel('log10FIdist') ; axis tight
            drawnow
        end
        
        %% Create a GRAPH with all matches 
        if plotType == 2
            CC = struct;
            MZRTstr_Ref = M2S_createLabelMZRT('REF',refSet_i(:,2),refSet_i(:,1));
            MZRTstr_Target = M2S_createLabelMZRT('TARGET',targetSet_i(:,2),targetSet_i(:,1));
            eL_allMatches = [table([MZRTstr_Ref,MZRTstr_Target],'VariableNames',{'EndNodes'}), ...
                table((1:length(Xr_connIdx_i))','VariableNames',{'rowNrInMatchedSets'})];
            G1_temp = graph(eL_allMatches);% Only for connected components
            G1 = digraph(eL_allMatches);
            % Find if node is reference node
            G1.Nodes.refNode = double(contains(G1.Nodes.Name,'REF'));
            % Find in which cluster the node is
            G1.Nodes.CCnr = (conncomp(G1_temp))';
            % Find in which cluster the edge is
            G1.Edges.CCnr = G1.Nodes.CCnr(M2S_find_idxInReference(G1.Edges.EndNodes(:,1),G1.Nodes.Name));

            % Find number of nodes and edges in each cluster
            CC.nNodes = tabulate(G1.Nodes.CCnr); CC.nNodes = CC.nNodes(:,1:2);
            CC.nEdges = tabulate(G1.Edges.CCnr); CC.nEdges = CC.nEdges(:,1:2);

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

            opt.G1 = G1;
            M2S_figureH(0.6,0.8);
            set(gcf,'Name','Nodes as metabolomic features: Reference in dark blue, Target in light blue. Grey edges are matches within all thresholds, orange edges are matches outside log10FI threshold.');
            movegui(gcf,'center')
            p1 = plot(G1,'LineWidth',2,'EdgeColor',M2Scolor.lgrey); 
            highlight(p1,find(G1.Nodes.refNode==1),'NodeColor',M2Scolor.dblue,'MarkerSize',3);
            highlight(p1,find(G1.Nodes.refNode==0),'NodeColor',M2Scolor.lblue,'MarkerSize',3);
            highlight(p1,find(G1.Edges.rowNrInMatchedSets(FIdistOK_01==0)),'EdgeColor',M2Scolor.orange,'LineWidth',1.5)
            set(gca,'XTick',[], 'YTick', []); axis tight; drawnow;
        end
    end
end

% Check if RT is in seconds and not in minutes
if max(refFeatures(:,1))>60 
    warndlg('The retention time of refFeatures seems to be in seconds. Please divide it by 60 to make it in minutes','Warning');
elseif max(targetFeatures(:,1))>60
    warndlg('The retention time of targetFeatures seems to be in seconds. Please divide it by 60 to make it in minutes','Warning');
elseif max(refFeatures(:,1))>60 || max(targetFeatures(:,1))>60
    warndlg('The retention time of both refFeatures and targetFeatures seems to be in seconds. Please divide it by 60 to make it in minutes','Warning');
end





