%function [RTdist_medNeigh_to_TRpair,MZdist_medNeigh_to_TRpair,RTMZdist_medNeigh_to_TRpair_scaled,median_RTdist_neighborsTR,median_MZdist_neighborsTR] =...
%matchRTMZ_calcNeighbours_v3(ref_MatchSet,Xref_connected_idx_inMatchSet,target_MatchSet, refSet,Xref_connected_idx,targetSet,nrNeighbors,XdimForScores,neighMethod,W)
% function [RTdist_medNeigh_to_TRpair,MZdist_medNeigh_to_TRpair,median_RTdist_neighborsTR,median_MZdist_neighborsTR,RTMZFIdist_medNeigh_to_TRpair] =...
%     M2S_findNeighTrendsResiduals(ref_MatchSet,Xref_connected_idx_inMatchSet,target_MatchSet, refSet,Xref_connected_idx,targetSet,nrNeighbors,neighMethod)
% function [RTdist_medNeigh_to_TRpair,MZdist_medNeigh_to_TRpair,median_RTdist_neighborsTR,median_MZdist_neighborsTR,RTMZFIdist_medNeigh_to_TRpair] =...
%     M2S_findNeighTrendsResiduals(ref_MatchSet, target_MatchSet, refSet, targetSet, Xr_connIdx_inMatchSet, Xr_connIdx, nrNeighbors, neighMethod)

% function [Residuals_X,Residuals_trendline] = M2S_calculateResiduals(ref_MatchSet, target_MatchSet, refSet, targetSet, Xr_connIdx_inMatchSet, Xr_connIdx, nrNeighbors, neighMethod, plotOrNot)
function [Residuals_X,Residuals_trendline] = M2S_calculateResiduals(refSet,targetSet,Xr_connIdx,Xt_connIdx, nrNeighbors, neighMethod, plotOrNot)

% The following call makes automatic choice of neighbour number
% [Residuals_X] = M2S_findNeighTrendsResiduals(ref_MatchSet, target_MatchSet, refSet, targetSet, Xr_connIdx_inMatchSet, Xr_connIdx) 

% Run the function to define the ref and target sets from which to choose
% neighbours. These only contain single matches, not any multiple ones.
[ref_MatchSet,target_MatchSet,Xr_connIdx_inMatchSet,Xt_connIdx_inMatchSet] = ...
    M2S_createMatchSets(refSet,targetSet,Xr_connIdx,Xt_connIdx,1);

lightBlueColor = uint8([82, 142, 173]);
orangeColor = uint8([255,154,16]);
mediumGreyColor = uint8([165,166,165]);


if nargin == 4
    nrNeighbors = 10 + ceil(0.01*size(ref_MatchSet,1));% at least 10 plus 1% of the size of the reference set
    neighMethod = 'cross';
    plotOrNot=1;
elseif nargin == 5
    neighMethod = 'cross';
    plotOrNot=1;
elseif nargin == 6
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


%% 1. FIND THE NEIGHBOURS OF EACH FEATURE AND DISTANCES TARGET to REF (TR)

% CROSS-TYPE NEIGHB0URS
if strcmp(neighMethod,'cross')   
    % This is a function to find neighbours distances.
    % It calculates RTdist_neighbours and MZdist_neighbours with neighbours in
    % the RT and in the MZ domain separately, resulting in a "cross" shape.
    wb1 = waitbar(0,'Calculating distances across sets for the neighbours of each feature','Name','Info');
    idx_neighbors_Ref_RT=M2S_findNeighbours(ref_MatchSet,Xr_connIdx_inMatchSet,refSet,Xr_connIdx,nrNeighbors,1);
    waitbar(0.33,wb1,'Calculating distances across sets for the neighbours of each feature','Name','Info');
    
    idx_neighbors_Ref_MZ=M2S_findNeighbours(ref_MatchSet,Xr_connIdx_inMatchSet,refSet,Xr_connIdx,nrNeighbors,2);
    waitbar(0.66,wb1,'Calculating distances across sets for the neighbours of each feature','Name','Info');
    
    idx_neighbors_Ref_FI= [idx_neighbors_Ref_RT,idx_neighbors_Ref_MZ];
    waitbar(0.99,wb1,'Calculating distances across sets for the neighbours of each feature','Name','Info');
    
    % calculate median RTdist of neighborsTR (Target-Reference)
    RTdist_neighborsTR=[];
    MZdist_neighborsTR=[];
    FIdist_neighborsTR=[];

    
    for columnNr = 1:nrNeighbors        
        RTdist_neighborsTR(:,columnNr) = target_MatchSet(idx_neighbors_Ref_RT(:,columnNr),1) - ref_MatchSet(idx_neighbors_Ref_RT(:,columnNr),1);
        MZdist_neighborsTR(:,columnNr) = target_MatchSet(idx_neighbors_Ref_MZ(:,columnNr),2) - ref_MatchSet(idx_neighbors_Ref_MZ(:,columnNr),2);
    end
   
    
    for columnNr = 1:size(idx_neighbors_Ref_FI,2)
        FIdist_neighborsTR(:,columnNr) = target_MatchSet(idx_neighbors_Ref_FI(:,columnNr),3) - ref_MatchSet(idx_neighbors_Ref_FI(:,columnNr),3);
    end

    waitbar(1,wb1,'Done!','Name','Info');
    pause(1)
    close(wb1);
    
% CIRCLE-TYPE NEIGHB0URS
elseif strcmp(neighMethod,'circle')
    % This is a function to find neighbours distances.
    % It calculates RTdis_neighbours and MZdist_neighbours with neighbours in
    % the RT and in the MZ domain together, resulting in a "circle" shape.
    wb1 = waitbar(0,'Calculating distances across sets for the neighbours of each feature','Name','Info');
    idx_neighbors_Ref=M2S_findNeighbours(ref_MatchSet,Xr_connIdx_inMatchSet,refSet,Xr_connIdx,nrNeighbors,[1,2]);

    % calculate median RTdist of neighborsTR (Target-Reference)
    RTdist_neighborsTR = [];
    MZdist_neighborsTR = [];
    FIdist_neighborsTR = []
    for columnNr = 1:nrNeighbors
        waitbar(columnNr/nrNeighbors,wb1,'Calculating distances across sets for the neighbours of each feature','Name','Info');
        RTdist_neighborsTR(:,columnNr) = target_MatchSet(idx_neighbors_Ref(:,columnNr),1) - ref_MatchSet(idx_neighbors_Ref(:,columnNr),1);
        MZdist_neighborsTR(:,columnNr) = target_MatchSet(idx_neighbors_Ref(:,columnNr),2) - ref_MatchSet(idx_neighbors_Ref(:,columnNr),2);
        FIdist_neighborsTR(:,columnNr) = target_MatchSet(idx_neighbors_Ref(:,columnNr),3) - ref_MatchSet(idx_neighbors_Ref(:,columnNr),3);
    end
    waitbar(1,wb1,'Done!','Name','Info');
    pause(1)
    close(wb1);
end


%% 2. CALCULATE THE NEIGHBOUR DISTANCES IN EACH DIMENSION
% For each feature find the median difference of distances of its
% neighbours from reference to target (SHIFTS TRENDLINE). Normalize.

median_RTdist_neighborsTR = nanmedian(RTdist_neighborsTR,2);% REPRESENTS TRENDLINE OF RT(target-ref)
median_MZdist_neighborsTR = nanmedian(MZdist_neighborsTR,2);% REPRESENTS TRENDLINE OF MZ(target-ref)
median_FIdist_neighborsTR = nanmedian(FIdist_neighborsTR,2);% REPRESENTS TRENDLINE OF FI(target-ref)
median_RTMZFIdist_neighborsTR = [median_RTdist_neighborsTR,median_MZdist_neighborsTR,median_FIdist_neighborsTR]; % ALL TOGETHER

% Get RTdist etc of actual feature pairs
RT_dist_TR = targetSet(:,1) - refSet(:,1);% FEATURES TARGET - REF
MZ_dist_TR = targetSet(:,2) - refSet(:,2);% FEATURES TARGET - REF
FI_dist_TR = targetSet(:,3) - refSet(:,3);% FEATURES TARGET - REF

% Calculate deltaRTdist (RESIDUALS) between each median_RTdist_neighborsTR and RTdist (the deltaRT to neighbors)
RTdist_medNeigh_to_TRpair = RT_dist_TR - median_RTdist_neighborsTR; % IMPORTANT ONE
MZdist_medNeigh_to_TRpair = MZ_dist_TR - median_MZdist_neighborsTR; % IMPORTANT ONE
FIdist_medNeigh_to_TRpair = FI_dist_TR - median_FIdist_neighborsTR; % IMPORTANT ONE

Residuals_trendline = [median_RTdist_neighborsTR,median_MZdist_neighborsTR,median_FIdist_neighborsTR];
% Residuals_X = [RTdist_medNeigh_to_TRpair,MZdist_medNeigh_to_TRpair,FIdist_medNeigh_to_TRpair];
% RESIDUALS USED TO CALCULATE THE PENALISATION SCORES ***
Residuals_X = [RTdist_medNeigh_to_TRpair,MZdist_medNeigh_to_TRpair,targetSet(:,3)-refSet(:,3)];% RESIDUALS USED TO CALCULATE THE PENALISATION SCORES

% NOTE: Residuals_X are the "RTMZFIdist_medNeigh_to_TRpair"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLOTS ******************************************************
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if plotOrNot == 1
    
redColor = [1,0,0];
blackColor = [0,0,0];
blueColor = [0,0,1];
greyColor = [0.5,0.5,0.5];
featureNr = round(size(refSet,1)/2); % only used for the plot of neighbours of a feature

if strcmp(neighMethod,'cross')
    
    % NOTE: to see a plot of the neighbours of a feature,  uncomment the following lines
    
    % Figure to check the neighbours of one of the features
    temp_idx_neighbours_Ref_RT = (idx_neighbors_Ref_RT(featureNr,:))';
    temp_idx_neighbours_Ref_MZ = (idx_neighbors_Ref_MZ(featureNr,:))';
    figure('Position',[1 732 1226 263],'Name',['Example of neighbours for feature number ',num2str(featureNr)]), 
    subplot(1,3,1)
    plot(ref_MatchSet(:,1),ref_MatchSet(:,2),'.k'), hold on
    plot(ref_MatchSet(temp_idx_neighbours_Ref_RT,1),ref_MatchSet(temp_idx_neighbours_Ref_RT,2),'.','Color',redColor,'MarkerSize',14)
    plot(ref_MatchSet(temp_idx_neighbours_Ref_MZ,1),ref_MatchSet(temp_idx_neighbours_Ref_MZ,2),'.','Color',blueColor,'MarkerSize',14)
    plot(refSet(featureNr,1),refSet(featureNr,2),'x','Color',greyColor,'MarkerSize',14,'LineWidth',3)
    xlabel('RT reference (Minutes)'), ylabel('MZ reference (m/z units)'), grid on, axis tight
    subplot(1,3,2)
    plot(ref_MatchSet(:,1),target_MatchSet(:,1)-ref_MatchSet(:,1),'.k'), hold on
    plot(ref_MatchSet(temp_idx_neighbours_Ref_RT,1),target_MatchSet(temp_idx_neighbours_Ref_RT,1)-ref_MatchSet(temp_idx_neighbours_Ref_RT,1),'.','Color',redColor,'MarkerSize',14)
    plot(refSet(featureNr,1),targetSet(featureNr,1)-refSet(featureNr,1),'x','Color',greyColor,'MarkerSize',14,'LineWidth',3)
    xlabel('RT reference (Minutes)'), ylabel('RTdist (Minutes)'), grid on, axis tight
    subplot(1,3,3)
    plot(ref_MatchSet(:,2),target_MatchSet(:,2)-ref_MatchSet(:,2),'.k'), hold on
    plot(ref_MatchSet(temp_idx_neighbours_Ref_MZ,2),target_MatchSet(temp_idx_neighbours_Ref_MZ,2)-ref_MatchSet(temp_idx_neighbours_Ref_MZ,2),'.','Color',blueColor,'MarkerSize',14)
    plot(refSet(featureNr,2),targetSet(featureNr,2)-refSet(featureNr,2),'x','Color',greyColor,'MarkerSize',14,'LineWidth',3)
    xlabel('MZ reference (Minutes)'), ylabel('MZdist (m/z units)'), grid on, axis tight
    drawnow
    
    % Figure with the trends for RT and MZ
    figure('Position',[5 403 1226 245],'Name','Trends for RT and MZ'), 
    subplot(1,2,1),
    plot(refSet(:,1),targetSet(:,1) - refSet(:,1),'.k'), axis tight, grid on
    y_lim=ylim; x_lim = xlim; hold on; plot(refSet(:,1),median_RTdist_neighborsTR,'or','MarkerSize',4), grid on
    ylim(y_lim); xlim(x_lim); xlabel('RT reference (Minutes)'), ylabel('RTdist (Minutes)')
    subplot(1,2,2)
    plot(refSet(:,2),targetSet(:,2) - refSet(:,2),'.k'), axis tight, grid on
    y_lim=ylim; x_lim = xlim; hold on; plot(refSet(:,2),median_MZdist_neighborsTR,'or','MarkerSize',4), grid on
    ylim(y_lim); xlim(x_lim); xlabel('MZ reference (m/z units)'), ylabel('MZdist (m/z units)')
    drawnow  
   
    % Figure with the Residuals_X
    figure('Position',[3 54 1226 263],'Name','Residuals for for RT and MZ')   
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
    
    % Figure to check the neighbours of one of the features
    temp_idx_neighbours_Ref = (idx_neighbors_Ref(featureNr,:))';
    figure('Position',[1 732 1226 263],'Name',['Example of neighbours for feature number ',num2str(featureNr)]), 
    subplot(1,3,1)
    plot(ref_MatchSet(:,1),ref_MatchSet(:,2),'.k'), hold on
    plot(refSet(featureNr,1),refSet(featureNr,2),'ok')
    plot(ref_MatchSet(temp_idx_neighbours_Ref,1),ref_MatchSet(temp_idx_neighbours_Ref,2),'.','Color',redColor,'MarkerSize',14)
    plot(refSet(featureNr,1),refSet(featureNr,2),'x','Color',greyColor,'MarkerSize',14,'LineWidth',3)
    xlabel('RT reference (Minutes)'), ylabel('MZ reference (m/z units)'), grid on, axis tight
    subplot(1,3,2)
    plot(ref_MatchSet(:,1),target_MatchSet(:,1)-ref_MatchSet(:,1),'.k'), hold on
    plot(refSet(featureNr,1),targetSet(featureNr,1)-refSet(featureNr,1),'ok')
    plot(ref_MatchSet(temp_idx_neighbours_Ref,1),target_MatchSet(temp_idx_neighbours_Ref,1)-ref_MatchSet(temp_idx_neighbours_Ref,1),'.','Color',redColor,'MarkerSize',14)
    plot(refSet(featureNr,1),targetSet(featureNr,1)-refSet(featureNr,1),'x','Color',greyColor,'MarkerSize',14,'LineWidth',3)
    xlabel('RT reference (Minutes)'), ylabel('RTdist (Minutes)'), grid on, axis tight
    subplot(1,3,3)
    plot(ref_MatchSet(:,2),target_MatchSet(:,2)-ref_MatchSet(:,2),'.k'), hold on
    plot(refSet(featureNr,2),targetSet(featureNr,2)-refSet(featureNr,2),'ok')
    plot(ref_MatchSet(temp_idx_neighbours_Ref,2),target_MatchSet(temp_idx_neighbours_Ref,2)-ref_MatchSet(temp_idx_neighbours_Ref,2),'.','Color',redColor,'MarkerSize',14)
    plot(refSet(featureNr,2),targetSet(featureNr,2)-refSet(featureNr,2),'x','Color',greyColor,'MarkerSize',14,'LineWidth',3)
    xlabel('MZ reference (Minutes)'), ylabel('MZdist (m/z units)'), grid on, axis tight
    drawnow
    
    % Figure with the trends for RT and MZ
    figure('Position',[5 403 1226 245],'Name','Trends for RT and MZ'), 
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
    figure('Position',[3 54 1226 263],'Name','Residuals for for RT and MZ')   
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