% [Residuals_X,Residuals_trendline] = M2S_calculateResiduals(refSet,targetSet,Xr_connIdx,Xt_connIdx, nrNeighbors, neighMethod, plotOrNot)
% This function is part of M2S toolbox to match metabolomics features across untargeted datasets.
%
% It calculates the difference (residuals) between each match and the 
% inter-dataset shift of two datasets, in each dimension.
%
% INPUT:
% refSet,targetSet: nx3 matrices of MZ,RT,FI that have been matched
% (non-unique entries, containing clusters of multiple matches).
% Xr_connIdx,Xt_connIdx: single-column indices of features in the initial
% datasets that have been matched to each other.
% nrNeighbors: a percentage of the total of features in the refSet (e.g. 0.05) or a specific number (e.g. 25).
% neighMethod:method to calculate inter-dataset shifts can be "cross" (uses RT, MZ, FI)
% or (default) "circle" (uses only RT, MZ). "Circle" does not subtract the shift from FI.
% plotOrNot: 0 - no plots; 1 - default plots; 2 - extra plots
% NOTE: % The following call makes automatic choice of neighbour number:
% [Residuals_X,Residuals_trendline] = M2S_findNeighTrendsResiduals(ref_MatchSet, target_MatchSet, refSet, targetSet, Xr_connIdx_inMatchSet, Xr_connIdx) 
%
% OUTPUT:
% Residuals_X: the distances between each match and the inter-dataset shift
% Residuals_trendline: the inter-dataset shift at each reference 
%
%   Rui Climaco Pinto, 2021
%   Imperial College London


function [Residuals_X,Residuals_trendline] = M2S_calculateResiduals ...
    (refSet,targetSet,Xr_connIdx,Xt_connIdx, nrNeighbors, neighMethod, pctPointsLoess, plotOrNot)


% Run the function to define the ref and target sets from which to choose
% neighbours. These only contain single matches, not multiple ones (in clusters).

if plotOrNot == 2
[ref_MatchSet,target_MatchSet,Xr_connIdx_inMatchSet,Xt_connIdx_inMatchSet] = ...
    M2S_createMatchSets(refSet,targetSet,Xr_connIdx,Xt_connIdx,1);
else
    [ref_MatchSet,target_MatchSet,Xr_connIdx_inMatchSet,Xt_connIdx_inMatchSet] = ...
    M2S_createMatchSets(refSet,targetSet,Xr_connIdx,Xt_connIdx,0);
end

if nargin == 4
    nrNeighbors = 10 + ceil(0.01*size(ref_MatchSet,1));% at least 10 neighbours plus 1% of the size of the reference set
    neighMethod = 'circle';
    pctPointsLoess = 0;
    plotOrNot=1;
elseif nargin == 5
    neighMethod = 'circle';
    pctPointsLoess = 0;
    plotOrNot=1;
elseif nargin == 6
    plotOrNot=1;
    pctPointsLoess = 0;
elseif nargin == 7
    plotOrNot=1;
end
   
if nrNeighbors<1 % it is a fraction of the size of ref_MatchSet
    nrNeighbors = round(nrNeighbors*size(ref_MatchSet,1));
end  

% Transform FImed TO LOG10 
ref_MatchSet(:,3) = log10(ref_MatchSet(:,3));
target_MatchSet(:,3) = log10(target_MatchSet(:,3));
refSet(:,3) = log10(refSet(:,3));
targetSet(:,3) = log10(targetSet(:,3));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. FIND THE NEIGHBOURS OF EACH FEATURE AND DISTANCES TARGET to REF (TR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% CROSS-TYPE NEIGHB0URS (uses RT, MZ and FI)
if strcmp(neighMethod,'cross')   
    % This is a function to find neighbours distances.
    % It calculates RTdist_neighbours and MZdist_neighbours with neighbours in
    % the RT and in the MZ domain separately, resulting in a "cross" shape.
    wb1 = waitbar(0,'Calculating distances across sets for the neighbours of each feature','Name','Info');
    idx_neighbors_Ref_RT=M2S_findNeighbours(ref_MatchSet,Xr_connIdx_inMatchSet,refSet,Xr_connIdx,nrNeighbors,1);
    waitbar(0.33,wb1,'Calculating distances across sets for the neighbours of each feature','Name','Info');
    
    idx_neighbors_Ref_MZ=M2S_findNeighbours(ref_MatchSet,Xr_connIdx_inMatchSet,refSet,Xr_connIdx,nrNeighbors,2);
    waitbar(0.66,wb1,'Calculating distances across sets for the neighbours of each feature','Name','Info');

    idx_neighbors_Ref_FI=M2S_findNeighbours(ref_MatchSet,Xr_connIdx_inMatchSet,refSet,Xr_connIdx,nrNeighbors,3);
    waitbar(0.99,wb1,'Calculating distances across sets for the neighbours of each feature','Name','Info');
    
    % calculate median RTdist of neighborsTR (Target-Reference)
    RTdist_neighborsTR=[];
    MZdist_neighborsTR=[];
    FIdist_neighborsTR=[];
    for columnNr = 1:nrNeighbors        
        RTdist_neighborsTR(:,columnNr) = target_MatchSet(idx_neighbors_Ref_RT(:,columnNr),1) - ref_MatchSet(idx_neighbors_Ref_RT(:,columnNr),1);
        MZdist_neighborsTR(:,columnNr) = target_MatchSet(idx_neighbors_Ref_MZ(:,columnNr),2) - ref_MatchSet(idx_neighbors_Ref_MZ(:,columnNr),2);
        FIdist_neighborsTR(:,columnNr) = target_MatchSet(idx_neighbors_Ref_FI(:,columnNr),3) - ref_MatchSet(idx_neighbors_Ref_FI(:,columnNr),3);
    end
      
    waitbar(1,wb1,'Done!','Name','Info');
    pause(1)
    close(wb1);
    
%% CIRCLE-TYPE NEIGHB0URS (uses only RT and MZ)
elseif strcmp(neighMethod,'circle')
    % This is a function to find neighbours distances.
    % It calculates RTdis_neighbours and MZdist_neighbours with neighbours in
    % the RT and in the MZ domain together, resulting in a "circle" shape.
    % FI does not enter the calculation.
    wb1 = waitbar(0,'Calculating distances across sets for the neighbours of each feature','Name','Info');
    idx_neighbors_Ref=M2S_findNeighbours(ref_MatchSet,Xr_connIdx_inMatchSet,refSet,Xr_connIdx,nrNeighbors,[1,2]);% only neighbours in RT + MZ

    % calculate median RTdist of neighborsTR (Target-Reference)
    RTdist_neighborsTR = [];
    MZdist_neighborsTR = [];
    FIdist_neighborsTR = [];
    for columnNr = 1:nrNeighbors
        waitbar(columnNr/nrNeighbors,wb1,'Calculating distances across sets for the neighbours of each feature','Name','Info');
        RTdist_neighborsTR(:,columnNr) = target_MatchSet(idx_neighbors_Ref(:,columnNr),1) - ref_MatchSet(idx_neighbors_Ref(:,columnNr),1);
        MZdist_neighborsTR(:,columnNr) = target_MatchSet(idx_neighbors_Ref(:,columnNr),2) - ref_MatchSet(idx_neighbors_Ref(:,columnNr),2);
        %FIdist_neighborsTR(:,columnNr) = target_MatchSet(idx_neighbors_Ref(:,columnNr),3) - ref_MatchSet(idx_neighbors_Ref(:,columnNr),3);       
    end
    FIdist_neighborsTR = zeros(size(MZdist_neighborsTR));% defined as zero
    waitbar(1,wb1,'Done!','Name','Info');
    pause(1)
    close(wb1);
end


%% 2. CALCULATE THE NEIGHBOUR DISTANCES IN EACH DIMENSION
% For each feature find the median difference of distances of its
% neighbours from reference to target (SHIFTS TRENDLINE). Normalize.
% NOTICE that in "circle" method the median_FIdist_neighborsTR is zero (does not correct FI_dist_TR)

median_RTdist_neighborsTR = nanmedian(RTdist_neighborsTR,2);% REPRESENTS TRENDLINE OF RT(target-ref)
median_MZdist_neighborsTR = nanmedian(MZdist_neighborsTR,2);% REPRESENTS TRENDLINE OF MZ(target-ref)
median_FIdist_neighborsTR = nanmedian(FIdist_neighborsTR,2);% REPRESENTS TRENDLINE OF FI(target-ref)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if strcmp(neighMethod,'cross')
% pctPointsLoess is a percentage ]0-0.75[ of the number of points in the
% reference matched dataset (refSet), to define span of loess.
% e.g. use pctPointsLoess = 0.3
if pctPointsLoess > 0 && pctPointsLoess <0.75

    %% Use robust loess to calculate trend (only for "cross" method)
   
    % nrPointsLoess = 0.1 * length(unique(refSet(:,1)));
    nrPointsLoess = round(pctPointsLoess * length(unique(refSet(:,1))));
    median_RTdist_neighborsTR = smooth(refSet(:,1),median_RTdist_neighborsTR,nrPointsLoess,'rloess');% to do loess
    median_MZdist_neighborsTR = smooth(refSet(:,2),median_MZdist_neighborsTR,nrPointsLoess,'rloess');% to do loess
    if strcmp(neighMethod,'cross')% method 'circle' does not calculate FI
        median_FIdist_neighborsTR = smooth(refSet(:,3),median_FIdist_neighborsTR,nrPointsLoess,'rloess');% to do loess
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Get RTdist etc of actual feature pairs
RT_dist_TR = targetSet(:,1) - refSet(:,1);% FEATURES TARGET - REF
MZ_dist_TR = targetSet(:,2) - refSet(:,2);% FEATURES TARGET - REF
FI_dist_TR = targetSet(:,3) - refSet(:,3);% FEATURES TARGET - REF

% Calculate deltaRTdist (RESIDUALS) between each median_RTdist_neighborsTR and RTdist (the deltaRT to neighbors)
RTdist_medNeigh_to_TRpair = RT_dist_TR - median_RTdist_neighborsTR; % IMPORTANT ONE
MZdist_medNeigh_to_TRpair = MZ_dist_TR - median_MZdist_neighborsTR; % IMPORTANT ONE
FIdist_medNeigh_to_TRpair = FI_dist_TR - median_FIdist_neighborsTR; % IMPORTANT ONE

Residuals_trendline = [median_RTdist_neighborsTR,median_MZdist_neighborsTR,median_FIdist_neighborsTR];
Residuals_X = [RTdist_medNeigh_to_TRpair,MZdist_medNeigh_to_TRpair,FIdist_medNeigh_to_TRpair];
% THESE ARE THE RESIDUALS USED TO CALCULATE THE PENALISATION SCORES 

% NOTE: Residuals_X are the "RTMZFIdist_medNeigh_to_TRpair"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLOTS ******************************************************
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
greyColor = [0.5 0.5 0.5];

if plotOrNot >= 1

featureNr = round(size(refSet,1)/2); % only used for the plot of neighbours of a feature

if strcmp(neighMethod,'cross')
    
    
    if plotOrNot==2
    %% Figure to check the neighbours of one of the features
    temp_idx_neighbours_Ref_RT = (idx_neighbors_Ref_RT(featureNr,:))';
    temp_idx_neighbours_Ref_MZ = (idx_neighbors_Ref_MZ(featureNr,:))';
    temp_idx_neighbours_Ref_FI = (idx_neighbors_Ref_FI(featureNr,:))';
    M2S_figureH(0.65,0.35);
    set(gcf,'Name',['Example of neighbours for feature number ',num2str(featureNr)])
    subplot(1,4,1)
    plot(ref_MatchSet(:,1),ref_MatchSet(:,2),'.k'), hold on
    plot(ref_MatchSet(temp_idx_neighbours_Ref_RT,1),ref_MatchSet(temp_idx_neighbours_Ref_RT,2),'.','Color','r','MarkerSize',14)
    plot(ref_MatchSet(temp_idx_neighbours_Ref_MZ,1),ref_MatchSet(temp_idx_neighbours_Ref_MZ,2),'.','Color','b','MarkerSize',14)
    plot(refSet(featureNr,1),refSet(featureNr,2),'x','Color',greyColor,'MarkerSize',14,'LineWidth',3)
    plot(nanmedian(ref_MatchSet(temp_idx_neighbours_Ref_RT,1)),nanmedian(ref_MatchSet(temp_idx_neighbours_Ref_MZ,2)),'o','Color',greyColor,'MarkerSize',12,'LineWidth',3)
    xlabel('RT reference (Minutes)'), ylabel('MZ reference (m/z units)'), grid on, axis tight
    subplot(1,4,2)
    plot(ref_MatchSet(:,1),target_MatchSet(:,1)-ref_MatchSet(:,1),'.k'), hold on
    plot(ref_MatchSet(temp_idx_neighbours_Ref_RT,1),target_MatchSet(temp_idx_neighbours_Ref_RT,1)-ref_MatchSet(temp_idx_neighbours_Ref_RT,1),'.','Color','r','MarkerSize',14)
    plot(refSet(featureNr,1),targetSet(featureNr,1)-refSet(featureNr,1),'x','Color',greyColor,'MarkerSize',14,'LineWidth',3)
    plot(refSet(featureNr,1),nanmedian(target_MatchSet(temp_idx_neighbours_Ref_RT,1)-ref_MatchSet(temp_idx_neighbours_Ref_RT,1)),'o','Color',greyColor,'MarkerSize',12,'LineWidth',3)
    xlabel('RT reference (Minutes)'), ylabel('RTdist (Minutes)'), grid on, axis tight
    subplot(1,4,3)
    plot(ref_MatchSet(:,2),target_MatchSet(:,2)-ref_MatchSet(:,2),'.k'), hold on
    plot(ref_MatchSet(temp_idx_neighbours_Ref_MZ,2),target_MatchSet(temp_idx_neighbours_Ref_MZ,2)-ref_MatchSet(temp_idx_neighbours_Ref_MZ,2),'.','Color','b','MarkerSize',14)
    plot(refSet(featureNr,2),targetSet(featureNr,2)-refSet(featureNr,2),'x','Color',greyColor,'MarkerSize',14,'LineWidth',3)
    plot(refSet(featureNr,2),nanmedian(target_MatchSet(temp_idx_neighbours_Ref_MZ,2)-ref_MatchSet(temp_idx_neighbours_Ref_MZ,2)),'o','Color',greyColor,'MarkerSize',12,'LineWidth',3)
    xlabel('MZ reference (Minutes)'), ylabel('MZdist (m/z units)'), grid on, axis tight
    subplot(1,4,4)
    plot(ref_MatchSet(:,3),target_MatchSet(:,3)-ref_MatchSet(:,3),'.k'), hold on
    plot(ref_MatchSet(temp_idx_neighbours_Ref_FI,3),target_MatchSet(temp_idx_neighbours_Ref_FI,3)-ref_MatchSet(temp_idx_neighbours_Ref_FI,3),'.','Color','b','MarkerSize',14)
    plot(refSet(featureNr,3),targetSet(featureNr,3)-refSet(featureNr,3),'x','Color',greyColor,'MarkerSize',14,'LineWidth',3)
    plot(refSet(featureNr,3),nanmedian(target_MatchSet(temp_idx_neighbours_Ref_FI,3)-ref_MatchSet(temp_idx_neighbours_Ref_FI,3)),'o','Color',greyColor,'MarkerSize',12,'LineWidth',3)
    xlabel('log10FI reference (Minutes)'), ylabel('log10FIdist (m/z units)'), grid on, axis tight
    drawnow
    end
    
    
    %% Figure with the trends for RT, MZ, FI
    M2S_figureH(0.65,0.35); set(gcf,'Name','Trends for RT, MZ and log10FI')
    subplot(1,3,1),
    plot(refSet(:,1),targetSet(:,1) - refSet(:,1),'.k'), axis tight, grid on
    y_lim=ylim; x_lim = xlim; hold on; plot(refSet(:,1),median_RTdist_neighborsTR,'or','MarkerSize',4), grid on
    ylim(y_lim); xlim(x_lim); xlabel('RT reference (Minutes)'), ylabel('RTdist (Minutes)')
    subplot(1,3,2)
    plot(refSet(:,2),targetSet(:,2) - refSet(:,2),'.k'), axis tight, grid on
    y_lim=ylim; x_lim = xlim; hold on; plot(refSet(:,2),median_MZdist_neighborsTR,'or','MarkerSize',4), grid on
    ylim(y_lim); xlim(x_lim); xlabel('MZ reference (m/z units)'), ylabel('MZdist (m/z units)')
    subplot(1,3,3)
    plot(refSet(:,3),targetSet(:,3) - refSet(:,3),'.k'), axis tight, grid on
    y_lim=ylim; x_lim = xlim; hold on; plot(refSet(:,3),median_FIdist_neighborsTR,'or','MarkerSize',4), grid on
    ylim(y_lim); xlim(x_lim); xlabel('log10FI reference'), ylabel('log10FIdist')
    drawnow  
    
   
    %% Figure with the Residuals_X
    M2S_figureH(0.65,0.35); set(gcf,'Name','Residuals for RT, MZ and log10FI')
    subplot(1,3,1),plot(refSet(:,1),Residuals_X(:,1),'.k'), axis tight, hold on, xlim1 = xlim; grid on
    plot(xlim',[0;0],'-k')
    xlabel('RT of reference feature'), ylabel('RT residuals') 
    subplot(1,3,2),plot(refSet(:,2),Residuals_X(:,2),'.k'), axis tight, hold on, xlim1 = xlim; grid on
    plot(xlim',[0;0],'-k')
    xlabel('MZ of reference feature'), ylabel('MZ residuals')
    subplot(1,3,3),plot(refSet(:,3),Residuals_X(:,3),'.k'), axis tight, hold on, xlim1 = xlim; grid on
    plot(xlim',[0;0],'-k')
    xlabel('Log10FI of reference feature'), ylabel('Log10FI residuals')

elseif  strcmp(neighMethod,'circle')
    
    if plotOrNot==2
    % Figure to check the neighbours of one of the features
    temp_idx_neighbours_Ref = (idx_neighbors_Ref(featureNr,:))';
    M2S_figureH(0.65,0.35); set(gcf,'Name',['Example of neighbours for feature number ',num2str(featureNr)])
    subplot(1,3,1)
    plot(ref_MatchSet(:,1),ref_MatchSet(:,2),'.k'), hold on
    plot(refSet(featureNr,1),refSet(featureNr,2),'ok')
    plot(ref_MatchSet(temp_idx_neighbours_Ref,1),ref_MatchSet(temp_idx_neighbours_Ref,2),'.','Color','r','MarkerSize',14)
    plot(refSet(featureNr,1),refSet(featureNr,2),'x','Color',greyColor,'MarkerSize',14,'LineWidth',3)
    plot(nanmedian(ref_MatchSet(temp_idx_neighbours_Ref,1)),nanmedian(ref_MatchSet(temp_idx_neighbours_Ref,2)),'o','Color',greyColor,'MarkerSize',12,'LineWidth',3)
    xlabel('RT reference (Minutes)'), ylabel('MZ reference (m/z units)'), grid on, axis tight
    subplot(1,3,2)
    plot(ref_MatchSet(:,1),target_MatchSet(:,1)-ref_MatchSet(:,1),'.k'), hold on
    plot(refSet(featureNr,1),targetSet(featureNr,1)-refSet(featureNr,1),'ok')
    plot(ref_MatchSet(temp_idx_neighbours_Ref,1),target_MatchSet(temp_idx_neighbours_Ref,1)-ref_MatchSet(temp_idx_neighbours_Ref,1),'.','Color','r','MarkerSize',14)
    plot(refSet(featureNr,1),targetSet(featureNr,1)-refSet(featureNr,1),'x','Color',greyColor,'MarkerSize',14,'LineWidth',3)
    plot(refSet(featureNr,1),nanmedian(target_MatchSet(temp_idx_neighbours_Ref,1)-ref_MatchSet(temp_idx_neighbours_Ref,1)),'o','Color',greyColor,'MarkerSize',12,'LineWidth',3)
    xlabel('RT reference (Minutes)'), ylabel('RTdist (Minutes)'), grid on, axis tight
    subplot(1,3,3)
    plot(ref_MatchSet(:,2),target_MatchSet(:,2)-ref_MatchSet(:,2),'.k'), hold on
    plot(refSet(featureNr,2),targetSet(featureNr,2)-refSet(featureNr,2),'ok')
    plot(ref_MatchSet(temp_idx_neighbours_Ref,2),target_MatchSet(temp_idx_neighbours_Ref,2)-ref_MatchSet(temp_idx_neighbours_Ref,2),'.','Color','r','MarkerSize',14)
    plot(refSet(featureNr,2),targetSet(featureNr,2)-refSet(featureNr,2),'x','Color',greyColor,'MarkerSize',14,'LineWidth',3)
    plot(refSet(featureNr,2),nanmedian(target_MatchSet(temp_idx_neighbours_Ref,2)-ref_MatchSet(temp_idx_neighbours_Ref,2)),'o','Color',greyColor,'MarkerSize',12,'LineWidth',3)
    xlabel('MZ reference (Minutes)'), ylabel('MZdist (m/z units)'), grid on, axis tight
    drawnow
    end
    
    
    % Figure with the trends for RT and MZ
    M2S_figureH(0.65,0.35); set(gcf,'Name','Trends for RT and MZ')
    subplot(1,2,1),plot(refSet(:,1),targetSet(:,1) - refSet(:,1),'.k'), axis tight, grid on
    y_lim=ylim; x_lim = xlim;
    %subplot(2,1,2), 
    hold on
    plot(refSet(:,1),median_RTdist_neighborsTR,'or','MarkerSize',4), grid on
    ylim(y_lim); xlim(x_lim);
    xlabel('RT reference (Minutes)'), ylabel('RTdist (Minutes)')
    subplot(1,2,2),plot(refSet(:,2),targetSet(:,2) - refSet(:,2),'.k'), axis tight, grid on
    y_lim=ylim; x_lim = xlim;
    %subplot(2,1,2), 
    hold on
    plot(refSet(:,2),median_MZdist_neighborsTR,'or','MarkerSize',4), grid on
    ylim(y_lim); xlim(x_lim);
    xlabel('MZ reference (m/z units)'), ylabel('MZdist (m/z units)')
    
    % Figure with the Residuals_X
    M2S_figureH(0.65,0.35); set(gcf,'Name','Residuals for RT and MZ')
    subplot(1,3,1),plot(refSet(:,1),Residuals_X(:,1),'.k'), axis tight, hold on, xlim1 = xlim; grid on
    plot(xlim',[0;0],'-k')
    xlabel('RT of reference feature'), ylabel('RT residuals (minutes)') 
    subplot(1,3,2),plot(refSet(:,2),Residuals_X(:,2),'.k'), axis tight, hold on, xlim1 = xlim; grid on
    plot(xlim',[0;0],'-k')
    xlabel('MZ of reference feature'), ylabel('MZ residuals (m/z units)')
    subplot(1,3,3),plot(refSet(:,3),Residuals_X(:,3),'.k'), axis tight, hold on, xlim1 = xlim; grid on
    plot(xlim',[0;0],'-k')
    xlabel('Log10FI of reference feature'), ylabel('Log10FI residuals')
end

end