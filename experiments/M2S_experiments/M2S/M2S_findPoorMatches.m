function [eL_final, eL_final_INFO] = M2S_findPoorMatches(eL,refSet,targetSet,methodType,nrMad,plotOrNot)

% methodType = {'scores','byBins','trend_mad','residuals_mad'}
% nrMad = 5
% plotOrNot = 1
fprintf('\n\n Function M2S_findPoorMatches\n')
fprintf(' Tighten thresholds to detect matches far away from the trends\n')

if nargin == 3
    methodType = 'residuals_mad';
    nrMad = 5;
    plotOrNot = 1;
elseif nargin ==4   
    nrMad = 5;
    plotOrNot = 1;
elseif nargin == 5
    plotOrNot = 1;
end


eL_Best_idx = find(eL.is_Best == 1);
eL_Best = eL(eL_Best_idx,:);

%% Method 0: Do not look for poor matches

if strcmp(methodType,'none')
    eL.notFalsePositives = ones(size(eL,1),1);
    eL.notFalsePositives(find(eL.is_Worst)) = NaN;
    disp('The method did not look for poor matches')
    
%% Method 1: find threshold limit in scores of best connections
elseif strcmp(methodType,'scores')

% Choose one or the other: 'mad' or 'outliers'  *******************
[eL_Best_FPresults,~] = M2S_threshSelection(eL_Best.matchScore, 'mad', nrMad, 0);
%[eL_Best_FPresults,~] = M2S_threshSelection(eL_Best.matchScore, 'outliers', 0.01, 1)

eL.notFalsePositives = NaN(size(eL,1),1);
eL.notFalsePositives(eL_Best_idx(eL_Best_FPresults.S_used_insideOfLimit_idx),:) = 1;
eL.notFalsePositives(eL_Best_idx(eL_Best_FPresults.S_allOutOfLimit_idx),:) = 0;
eL.notFalsePositives(find(eL.is_Worst)) = NaN;

if plotOrNot == 1
grpType = [NaN,0,1];
grpColor = 'brk';
grpMarker = 'so.';
grpMarkerSize = [4,4,8];
M2S_figureH(0.8,0.5);
set(gcf,'Name','Method - scores: Not matched (blue), poor matches (red) and good matches (black)'); 
for g=1:3
    %find row indices of refSet
    if g==1
        tempIdx = (eL.rowNrInMatchedSets(isnan(eL.notFalsePositives)));
    else
        tempIdx = (eL.rowNrInMatchedSets(eL.notFalsePositives==grpType(g)));
    end
    subplot(1,3,1), plot(refSet(tempIdx,1),targetSet(tempIdx,1)-refSet(tempIdx,1),[grpColor(g),grpMarker(g)],'MarkerSize',grpMarkerSize(g)), hold on, axis tight, grid on
    subplot(1,3,2), plot(refSet(tempIdx,2),targetSet(tempIdx,2)-refSet(tempIdx,2),[grpColor(g),grpMarker(g)],'MarkerSize',grpMarkerSize(g)), hold on, axis tight, grid on
    subplot(1,3,3), plot(log10(refSet(tempIdx,3)),log10(targetSet(tempIdx,3))-log10(refSet(tempIdx,3)),[grpColor(g),grpMarker(g)],'MarkerSize',grpMarkerSize(g)), hold on, axis tight, grid on
end
end
subplot(1,3,1), xlabel('RTreference (min)'), ylabel('RTdist (min)')
subplot(1,3,2), xlabel('MZreference (m/z units)'), ylabel('MZdist (m/z units)')
subplot(1,3,3), xlabel('log10FIreference'), ylabel('log10FIdist')


%M2S_plotMZRT_featureSet(targetFeatures,1,8,1); axis([0 10 0 2000])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Method 2: recalculate residuals only for best matches. Find threshold limit in RT, MZ and FI
elseif strcmp(methodType,'byBins')
%[Residuals_X,Residuals_trendline] = M2S_calculateResiduals(refSet,targetSet,Xr_connIdx,Xt_connIdx,nrNeighbors, neighMethod,pctPointsLoess,plotTypeResiduals)
[ResidualsReset,trendsReset] = M2S_calculateResiduals(refSet(eL_Best.rowNrInMatchedSets,:), targetSet(eL_Best.rowNrInMatchedSets,:), eL_Best.Xr_connIdx, eL_Best.Xt_connIdx,  round(0.01*size(eL_Best,1)), 'cross', 0, 0);


ResidualsReset_abs = abs(ResidualsReset);
nBins = 5;
binNumber=NaN(length(eL_Best.rowNrInMatchedSets),3);
binEdges = NaN(nBins+1,3);
for dimNr=1:size(ResidualsReset,2)
    if dimNr<3
        [binNumber(:,dimNr),binEdgesTemp] = discretize(refSet(eL_Best.rowNrInMatchedSets,dimNr),linspace(nanmin(refSet(:,dimNr)),nanmax(refSet(:,dimNr)),nBins+1));
    else
        [binNumber(:,dimNr),binEdgesTemp] = discretize(log10(refSet(eL_Best.rowNrInMatchedSets,dimNr)),nBins);
    end
    binEdges(:,dimNr) = binEdgesTemp';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The following contains a function similar to M2S_threshSelection (but single MAD, approx. normal distribution)
% applied to a number of bins (e.g.5), to find the MAD in each bin and
% define thresholds in each of those bins.

% nrMad = 5;
minNrValuesInBin=5;% minimum nr accepted per Bin, or emergency actions will apply

binCentres = binEdges(1:end-1,:) + 0.5 * diff(binEdges);
nrValues_inBin = NaN(nBins,3);
trendDimensionMedian = NaN(nBins,3);
trendDimensionMedian_noNaN = NaN(nBins,3);
median_residVals_inBin = NaN(nBins,3);
mad_residVals_inBin = NaN(nBins,3);

for dimNr=1:size(ResidualsReset,2)
    for binNr = 1:nBins
        %% For each dimension and bin:
        % get the median of the trend 
        trendDimensionMedian(binNr,dimNr) = nanmedian(trendsReset(binNumber(:,dimNr) == binNr,dimNr));
        % get the residuals of this bin in this dimension
        residVals_inBin = ResidualsReset(binNumber(:,dimNr) == binNr,dimNr);
        nrValues_inBin(binNr,dimNr) = length(residVals_inBin);
        median_residVals_inBin(binNr,dimNr) = median(residVals_inBin);
        mad_residVals_inBin(binNr,dimNr) =  mad(residVals_inBin);
    end
end

% Emergency actions: Find the bins that had no values, or a small number, and solve the issue

highLimit = NaN(nBins,3);
lowLimit = NaN(nBins,3);
for dimNr=1:size(ResidualsReset,2)
    for binNr = 1:nBins
        % by giving them NaN values
        if nrValues_inBin(binNr,dimNr)==0
            highLimit(binNr,dimNr) = NaN;
            lowLimit(binNr,dimNr) = NaN;
            % by making it larger than the largest residual
        elseif nrValues_inBin(binNr,dimNr)<minNrValuesInBin
            highLimit(binNr,dimNr) = trendDimensionMedian(binNr,dimNr) + nanmax(residVals_inBin) + 0.01*nanmax(residVals_inBin);
            lowLimit(binNr,dimNr) = trendDimensionMedian(binNr,dimNr) - nanmax(residVals_inBin) - 0.01*nanmin(residVals_inBin);
        else
            highLimit(binNr,dimNr) = trendDimensionMedian(binNr,dimNr) + median_residVals_inBin(binNr,dimNr) + mad_residVals_inBin(binNr,dimNr) * nrMad;
            lowLimit(binNr,dimNr) = trendDimensionMedian(binNr,dimNr) + median_residVals_inBin(binNr,dimNr) - mad_residVals_inBin(binNr,dimNr) * nrMad;
        end   
    end
end

binCentres_plusMinMax = [nanmin([refSet(:,1:2),log10(refSet(:,3))]);   binCentres;   nanmax([refSet(:,1:2),log10(refSet(:,3))])];
highLimit_plusMinMax = [highLimit(1,:);highLimit;highLimit(end,:)];
lowLimit_plusMinMax = [lowLimit(1,:);lowLimit;lowLimit(end,:)];

 
% Interpolate the High and Low limit threshold lines (using pchip) 
lowLimit_pchip=NaN(size(refSet)); highLimit_pchip=NaN(size(refSet));
for dimNr=1:size(ResidualsReset,2)
    if dimNr<3
        lowLimit_pchip(:,dimNr) = pchip(binCentres_plusMinMax(:,dimNr),lowLimit_plusMinMax(:,dimNr),refSet(:,dimNr));
        highLimit_pchip(:,dimNr) = pchip(binCentres_plusMinMax(:,dimNr),highLimit_plusMinMax(:,dimNr),refSet(:,dimNr));
    else
        lowLimit_pchip(:,dimNr) = pchip(binCentres_plusMinMax(:,dimNr),lowLimit_plusMinMax(:,dimNr),log10(refSet(:,dimNr)));
        highLimit_pchip(:,dimNr) = pchip(binCentres_plusMinMax(:,dimNr),highLimit_plusMinMax(:,dimNr),log10(refSet(:,dimNr)));
    end
end


% Get final results for method 2
eL.notFalsePositives = double((targetSet(:,1)-refSet(:,1) > lowLimit_pchip(:,1)) & (targetSet(:,1)-refSet(:,1) < highLimit_pchip(:,1)) &...
                       (targetSet(:,2)-refSet(:,2) > lowLimit_pchip(:,2)) & (targetSet(:,2)-refSet(:,2) < highLimit_pchip(:,2)) &...
                       ((  log10(targetSet(:,3)) - log10(refSet(:,3)) ) >lowLimit_pchip(:,3)) & ((  log10(targetSet(:,3)) - log10(refSet(:,3)) ) <highLimit_pchip(:,3)));
eL.notFalsePositives(find(eL.is_Worst)) = NaN;

if plotOrNot == 1
M2S_figureH(0.8,0.5);
set(gcf,'Name','Method - byBins: Not matched (blue), false positives (red) and good matches (black)')
subplot(1,3,1), 
%plot(refSet(:,1),targetSet(:,1)-refSet(:,1),'k.')
hold on, axis tight, grid on, 
%plot(refSet(eL_Best.rowNrInMatchedSets,1), trendsReset(:,1),'.b')
hold on, plot(binCentres(:,1),lowLimit(:,1),'or'); hold on, plot(binCentres(:,1),highLimit(:,1),'or')
hold on, plot(refSet(:,1),lowLimit_pchip(:,1),'.r'); plot(refSet(:,1),highLimit_pchip(:,1),'.r')
M2S_plotaxes('-k',[NaN,0])

subplot(1,3,2), 
%plot(refSet(:,2),targetSet(:,2)-refSet(:,2),'k.') 
hold on, axis tight, grid on
%plot(refSet(eL_Best.rowNrInMatchedSets,2), trendsReset(:,2),'.b')
hold on, plot(binCentres(:,2),lowLimit(:,2),'or'); hold on, plot(binCentres(:,2),highLimit(:,2),'or')
hold on, plot(refSet(:,2),lowLimit_pchip(:,2),'.r'); plot(refSet(:,2),highLimit_pchip(:,2),'.r')
M2S_plotaxes('-k',[NaN,0])

subplot(1,3,3), 
%plot(log10(refSet(:,3)),log10(targetSet(:,3))-log10(refSet(:,3)),'k.')
hold on, axis tight, grid on
%plot(log10(refSet(eL_Best.rowNrInMatchedSets,3)), trendsReset(:,3),'.b')
hold on, plot(binCentres(:,3),lowLimit(:,3),'or'); hold on, plot(binCentres(:,3),highLimit(:,3),'or')
hold on, plot(log10(refSet(:,3)),lowLimit_pchip(:,3),'.r'); plot(log10(refSet(:,3)),highLimit_pchip(:,3),'.r')
M2S_plotaxes('-k',[NaN,0])

% Plot the false positives
grpType = [NaN,0,1];
grpColor = 'brk';
grpMarker = 'so.';
grpMarkerSize = [6,4,8];
for g=1:3
    %find row indices of refSet
    if g==1
        tempIdx = (eL.rowNrInMatchedSets(isnan(eL.notFalsePositives)));
    else
        tempIdx = (eL.rowNrInMatchedSets(eL.notFalsePositives==grpType(g)));
    end
    subplot(1,3,1), plot(refSet(tempIdx,1),targetSet(tempIdx,1)-refSet(tempIdx,1),[grpColor(g),grpMarker(g)],'MarkerSize',grpMarkerSize(g)), hold on, axis tight, grid on
    subplot(1,3,2), plot(refSet(tempIdx,2),targetSet(tempIdx,2)-refSet(tempIdx,2),[grpColor(g),grpMarker(g)],'MarkerSize',grpMarkerSize(g)), hold on, axis tight, grid on
    subplot(1,3,3), plot(log10(refSet(tempIdx,3)),log10(targetSet(tempIdx,3))-log10(refSet(tempIdx,3)),[grpColor(g),grpMarker(g)],'MarkerSize',grpMarkerSize(g)), hold on, axis tight, grid on
end

subplot(1,3,1), xlabel('RTreference (min)'), ylabel('RTdist (min)')
subplot(1,3,2), xlabel('MZreference (m/z units)'), ylabel('MZdist (m/z units)')
subplot(1,3,3), xlabel('log10FIreference'), ylabel('log10FIdist')
end

%% Method 3: use a defined threshold for each domain
elseif strcmp(methodType,'trend_mad')
% recalculate residuals only for best matches. Use defined threshold limit in RT, MZ and FI

%[Residuals_X,Residuals_trendline] = M2S_calculateResiduals(refSet,targetSet,Xr_connIdx,Xt_connIdx,nrNeighbors, neighMethod,pctPointsLoess,plotTypeResiduals)
[ResidualsReset,trendsReset] = M2S_calculateResiduals(refSet(eL_Best.rowNrInMatchedSets,:), targetSet(eL_Best.rowNrInMatchedSets,:), eL_Best.Xr_connIdx, eL_Best.Xt_connIdx,  round(0.01*size(eL_Best,1)), 'cross',0, 0);

median_Residuals = nanmedian(ResidualsReset);
mad_Residuals =   mad(ResidualsReset);


highLimit = trendsReset + repmat(median_Residuals,size(eL_Best,1),1) + nrMad *repmat(mad_Residuals,size(eL_Best,1),1);
lowLimit = trendsReset + repmat(median_Residuals,size(eL_Best,1),1) - nrMad *repmat(mad_Residuals,size(eL_Best,1),1);

% Get final results for method 2
eL.notFalsePositives = NaN(size(eL,1),1);
eL.notFalsePositives(eL_Best.rowNrInMatchedSets) = double((targetSet(eL_Best.rowNrInMatchedSets,1)-refSet(eL_Best.rowNrInMatchedSets,1) > lowLimit(:,1)) & (targetSet(eL_Best.rowNrInMatchedSets,1)-refSet(eL_Best.rowNrInMatchedSets,1) < highLimit(:,1)) &...
                       (targetSet(eL_Best.rowNrInMatchedSets,2)-refSet(eL_Best.rowNrInMatchedSets,2) > lowLimit(:,2)) & (targetSet(eL_Best.rowNrInMatchedSets,2)-refSet(eL_Best.rowNrInMatchedSets,2) < highLimit(:,2)) &...
                       ((  log10(targetSet(eL_Best.rowNrInMatchedSets,3)) - log10(refSet(eL_Best.rowNrInMatchedSets,3)) ) >lowLimit(:,3)) & ((  log10(targetSet(eL_Best.rowNrInMatchedSets,3)) - log10(refSet(eL_Best.rowNrInMatchedSets,3)) ) <highLimit(:,3)));
eL.notFalsePositives(find(eL.is_Worst)) = NaN;

if plotOrNot == 1
M2S_figureH(0.8,0.5);
set(gcf,'Name','Method - trend_mad: Not matched (blue), false positives (red) and good matches (black)')
subplot(1,3,1), 
%plot(refSet(:,1),targetSet(:,1)-refSet(:,1),'k.')
hold on, axis tight, grid on, 
%plot(refSet(eL_Best.rowNrInMatchedSets,1), trendsReset(:,1),'.b')
hold on, plot(refSet(eL_Best.rowNrInMatchedSets,1),lowLimit(:,1),'.r'); plot(refSet(eL_Best.rowNrInMatchedSets,1),highLimit(:,1),'.r')
M2S_plotaxes('-k',[NaN,0]);

subplot(1,3,2), 
%plot(refSet(:,2),targetSet(:,2)-refSet(:,2),'k.') 
hold on, axis tight, grid on
%plot(refSet(eL_Best.rowNrInMatchedSets,2), trendsReset(:,2),'.b')
hold on, plot(refSet(eL_Best.rowNrInMatchedSets,2),lowLimit(:,2),'.r'); plot(refSet(eL_Best.rowNrInMatchedSets,2),highLimit(:,2),'.r')
M2S_plotaxes('-k',[NaN,0]);

subplot(1,3,3), 
%plot(log10(refSet(:,3)),log10(targetSet(:,3))-log10(refSet(:,3)),'k.')
hold on, axis tight, grid on
%plot(log10(refSet(eL_Best.rowNrInMatchedSets,3)), trendsReset(:,3),'.b')
hold on, plot(log10(refSet(eL_Best.rowNrInMatchedSets,3)),lowLimit(:,3),'.r'); plot(log10(refSet(eL_Best.rowNrInMatchedSets,3)),highLimit(:,3),'.r')
M2S_plotaxes('-k',[NaN,0])

% Plot the false positives
grpType = [NaN,0,1];
grpColor = 'brk';
grpMarker = 'so.';
grpMarkerSize = [6,4,8];
for g=1:3
    %find row indices of refSet
    if g==1
        tempIdx = (eL.rowNrInMatchedSets(isnan(eL.notFalsePositives)));
    else
        tempIdx = (eL.rowNrInMatchedSets(eL.notFalsePositives==grpType(g)));
    end
    subplot(1,3,1), plot(refSet(tempIdx,1),targetSet(tempIdx,1)-refSet(tempIdx,1),[grpColor(g),grpMarker(g)],'MarkerSize',grpMarkerSize(g)), hold on, axis tight, grid on
    subplot(1,3,2), plot(refSet(tempIdx,2),targetSet(tempIdx,2)-refSet(tempIdx,2),[grpColor(g),grpMarker(g)],'MarkerSize',grpMarkerSize(g)), hold on, axis tight, grid on
    subplot(1,3,3), plot(log10(refSet(tempIdx,3)),log10(targetSet(tempIdx,3))-log10(refSet(tempIdx,3)),[grpColor(g),grpMarker(g)],'MarkerSize',grpMarkerSize(g)), hold on, axis tight, grid on
end

subplot(1,3,1), xlabel('RTreference (min)'), ylabel('RTdist (min)')
subplot(1,3,2), xlabel('MZreference (m/z units)'), ylabel('MZdist (m/z units)')
subplot(1,3,3), xlabel('log10FIreference'), ylabel('log10FIdist')
end



%% Method 4: similar to method 3 but using (and plotting) only residuals (not the trends)
% Does not calculate residuals for matches not found previously. Only plots
% true positives and false positives (only uses eL_Best features)
elseif strcmp(methodType,'residuals_mad')
% Then one can consider the trend is zero for all dimensions.
% One can pchip x=[minRT, medianRT, maxRT] and use  y = nrMad * mad;
% [ResidualsReset,trendsReset] = M2S_calculateResiduals(refSet(eL_Best.rowNrInMatchedSets,:), targetSet(eL_Best.rowNrInMatchedSets,:), refSet(eL_Best.rowNrInMatchedSets,:), targetSet(eL_Best.rowNrInMatchedSets,:), (1:length(eL_Best.Xr_connIdx))', (1:length(eL_Best.Xr_connIdx))', round(0.01*size(eL_Best,1)), 'cross', 0);
[ResidualsReset,trendsReset] = M2S_calculateResiduals(refSet(eL_Best.rowNrInMatchedSets,:), targetSet(eL_Best.rowNrInMatchedSets,:), eL_Best.Xr_connIdx, eL_Best.Xt_connIdx,  round(0.01*size(eL_Best,1)), 'cross',0, 0);

% [ResidualsReset,trendsReset] = M2S_calculateResiduals(refSet(eL_Best.rowNrInMatchedSets,:), targetSet(eL_Best.rowNrInMatchedSets,:), refSet(eL_Best.rowNrInMatchedSets,:), targetSet(eL_Best.rowNrInMatchedSets,:), (1:length(eL_Best.Xr_connIdx))', (1:length(eL_Best.Xr_connIdx))', round(0.01*size(eL_Best,1)), 'cross', 0);
median_Residuals = nanmedian(ResidualsReset);
mad_Residuals =   mad(ResidualsReset);

lowLimit = median_Residuals - nrMad * mad_Residuals;
highLimit = median_Residuals + nrMad * mad_Residuals;

residuals_notFalsePositives = double(  (ResidualsReset(:,1)>lowLimit(1,1) & ResidualsReset(:,1)<highLimit(1,1)) & ...
                              (ResidualsReset(:,2)>lowLimit(1,2) & ResidualsReset(:,2)<highLimit(1,2)) & ...
                              (ResidualsReset(:,3)>lowLimit(1,3) & ResidualsReset(:,3)<highLimit(1,3))  );
residuals_notFalsePositives_idx = find(residuals_notFalsePositives);                         
residuals_falsePosIdx1 = setdiff((1:size(ResidualsReset,1))',residuals_notFalsePositives_idx);
refSet_notFalsePositivesIdx = eL_Best.rowNrInMatchedSets(residuals_notFalsePositives_idx);
refSet_falsePosIdx = eL_Best.rowNrInMatchedSets(residuals_falsePosIdx1);

eL.notFalsePositives = NaN(size(eL,1),1);
eL.notFalsePositives(refSet_notFalsePositivesIdx) = 1;
eL.notFalsePositives(refSet_falsePosIdx) = 0;

% Plots Method 4
if plotOrNot == 1

M2S_figureH(0.8,0.5);
set(gcf,'Name','Method - residuals_mad: Residuals of false positives (red) and good matches (black)')
subplot(1,3,1)
plot([min(refSet(eL_Best.rowNrInMatchedSets,1));max(refSet(eL_Best.rowNrInMatchedSets,1))],[lowLimit(1,1);lowLimit(1,1)],'-r')
hold on, axis tight, grid on
plot([min(refSet(eL_Best.rowNrInMatchedSets,1));max(refSet(eL_Best.rowNrInMatchedSets,1))],[highLimit(1,1);highLimit(1,1)],'-r')
plot(refSet(refSet_notFalsePositivesIdx,1),ResidualsReset(residuals_notFalsePositives_idx,1),'.k','MarkerSize',8)
plot(refSet(refSet_falsePosIdx,1),ResidualsReset(residuals_falsePosIdx1,1),'or','MarkerSize',4)
M2S_plotaxes('-k',[NaN,0]); 

subplot(1,3,2)
plot([min(refSet(eL_Best.rowNrInMatchedSets,2));max(refSet(eL_Best.rowNrInMatchedSets,2))],[lowLimit(1,2);lowLimit(1,2)],'-r')
hold on, axis tight, grid on
plot([min(refSet(eL_Best.rowNrInMatchedSets,2));max(refSet(eL_Best.rowNrInMatchedSets,2))],[highLimit(1,2);highLimit(1,2)],'-r')
plot(refSet(refSet_notFalsePositivesIdx,2),ResidualsReset(residuals_notFalsePositives_idx,2),'.k','MarkerSize',8)
plot(refSet(refSet_falsePosIdx,2),ResidualsReset(residuals_falsePosIdx1,2),'or','MarkerSize',4)
M2S_plotaxes('-k',[NaN,0]); 


subplot(1,3,3)
plot([min(log10(refSet(eL_Best.rowNrInMatchedSets,3)));max(log10(refSet(eL_Best.rowNrInMatchedSets,3)))],[lowLimit(1,3);lowLimit(1,3)],'-r')
hold on, axis tight, grid on
plot([min(log10(refSet(eL_Best.rowNrInMatchedSets,3)));max(log10(refSet(eL_Best.rowNrInMatchedSets,3)))],[highLimit(1,3);highLimit(1,3)],'-r')
plot(log10(refSet(refSet_notFalsePositivesIdx,3)),ResidualsReset(residuals_notFalsePositives_idx,3),'.k','MarkerSize',8)
plot(log10(refSet(refSet_falsePosIdx,3)),ResidualsReset(residuals_falsePosIdx1,3),'or','MarkerSize',4)
M2S_plotaxes('-k',[NaN,0]); 


subplot(1,3,1), xlabel('RTreference (min)'), ylabel('RT residuals (min)')
subplot(1,3,2), xlabel('MZreference (m/z units)'), ylabel('MZ residuals (m/z units)')
subplot(1,3,3), xlabel('log10FIreference'), ylabel('log10FI residuals')
end

end


eL_final_INFO{1,1} = '- This table contains matches between Ref and Target. It is in the same order as refSet, targetSet, Xr_connIdx and Xt_connIdx';
eL_final_INFO{2,1} = 'matchScore is the penalty score for the match. The smaller the better';
eL_final_INFO{3,1} = '- inCCnr is the initial multiple match cluster (graph connected component, or CC). That conncomp had "nrEdgesInSameCC" edges';
eL_final_INFO{4,1} = '- This match was in a cluster. The nrEdgesInSameCC shows the number of matches in the same CC as this edge. If ==1 means it was'; 
eL_final_INFO{5,1} = 'a single match, while >1 means there were multiple matches in the same CC';
eL_final_INFO{6,1} = '- nrIterations shows the number of iterations needed until this match was decided it is a good one. If ==0 means there';
eL_final_INFO{7,1} = 'were no other possible matches for the two features in it. NaN means it was never selected as a good match.';
eL_final_INFO{8,1} = '- is_Best indicates if this match was chosen as a good one during the iterative scores selection (even if not in the first iteration)';
eL_final_INFO{9,1} = '- is_Worst indicates that this match was not selected after the iterative process.';
eL_final_INFO{10,1} = '- notFalsePositives indicates the poor matches (==0), the good matches (==1) and the discarded matches (==NaN).';

eL_final = eL;
