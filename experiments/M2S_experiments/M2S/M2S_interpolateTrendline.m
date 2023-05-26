
function [Residuals_X,Residuals_trendline] = M2S_interpolateTrendline(reduced_refSet,Reduced_trendline,refSet,targetSet)


% loess_IQCimputed = smooth(iQC_runOrder,iQCdata_noLowOutliers_imputed(:,varNr),nrPoints,'loess');% to do loess
% valuesOfBio_inIQC = interp1(iQC_runOrder,loess_IQCimputed,bio_runOrder,'pchip',NaN);
Residuals_X=NaN(size(refSet,1),3);
Residuals_trendline=NaN(size(refSet,1),3);
for v = 1:3
    [~,uniqueIdx]= unique(reduced_refSet(:,v),'stable');
    %Residuals_trendline(:,v) = interp1(reduced_refSet(:,v),Reduced_trendline(:,v),refSet(:,v),'pchip',NaN);
    Residuals_trendline(:,v) = interp1(reduced_refSet(uniqueIdx,v),Reduced_trendline(uniqueIdx,v),refSet(:,v),'pchip',NaN);

    Residuals_X(:,v) = (targetSet(:,v)-refSet(:,1)) - Residuals_trendline(:,v);
end

%% Sort all dimensions by RT, only to be able to plot lines
M2S_figureH(0.8,0.4)

for v=1:3
    % Get the reduced sets sorted by reference
    [reduced_refSet_sortedVar, reduced_refSet_sortedVar_idx] = sort(reduced_refSet(:,v));
    Reduced_trendline_sortedVar = Reduced_trendline(reduced_refSet_sortedVar_idx,v);

    % Get the normal sets sorted by reference
    [refSet_sortedVar, refSet_sortedVar_idx] = sort(refSet(:,v));
    Residuals_trendline_sortedVar = Residuals_trendline(refSet_sortedVar_idx,v);

    % Plot
    subplot(1,3,v), plot(refSet(:,v),targetSet(:,v)-refSet(:,v),'.k'), hold on
    plot(reduced_refSet_sortedVar,Reduced_trendline_sortedVar,'x-b')

    % plot(reduced_refSet(:,v),Reduced_trendline(:,v),'xb')
    hold on, plot(refSet_sortedVar,Residuals_trendline_sortedVar,'-or')
    grid on
end
disp('Done!')

