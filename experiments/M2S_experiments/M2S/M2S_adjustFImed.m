function [newFItarget] = M2S_adjustFImed(FIref,FItarget,FIadj_method,plotOrNot)
%% M2S_adjustFImed
% calculate the relation between medians and Adjust the target FImed to
% the reference FImed
%
% INPUT:
% FIref, FItarget: row-matched single columns.
% They come from matched matrices with columns RT, MZ, FI (feature intensity)
% values or median values across the samples, e.g. representing the feature after peak picking)
% FIadj_method = 'median'; %{'none','median','regression'}
% The 'median' (default) method simply subtracts the median of ref to target.
% The 'regression' method adjust target to ref using linear regression.
%
% plotOrNot = 1; % 0 - no plots; 1 - plot
%
% 'FIadj_method' Methods description
% 'median': Subtract difference between intensity log10 medians of FIref and
%           FItarget. Robust (uses medians), but only does offset adjustment
% 'regression: Regression of refFI in targetFI, then use the predicted
%           values of targetFI as the new values of targetFI. Robust (uses robust
%           regression) and adjusts for slope and intercept differences in the two sets.
%           Useful if the FI deviation increases with increase of FItarget (and vice-versa).
%    
% OUTPUT
% newFItarget: in the same units as the input (not as log10)
fprintf('\n *** Executing function M2S_adjustFImed ***')
if nargin <=2
    FIadj_method = 'none';
    plotOrNot = 1;
elseif nargin <=3
    plotOrNot = 1;
end

if  strcmp(FIadj_method,'none')
    log10Med_offset = 0;
    targetSet_Log10MedAdj = log10(FItarget)- log10Med_offset; 
    fprintf('\n *** Adjustment of FItarget to FImedian was not performed. ***\n')
elseif strcmp(FIadj_method,'median')
    %% Method 1: median log10(FImed) difference
    log10Med_offset = nanmedian(log10(FItarget)-log10(FIref)); % offset of Median intensity to subtract to MedianIntensity_target
    targetSet_Log10MedAdj = log10(FItarget)- log10Med_offset; 
    fprintf('\n *** Adjusting FItarget to FImedian using the method: median ***\n')

elseif strcmp(FIadj_method,'regression')
    %% Method 2: robust linear regression target~1+ref)
    % subtracts the intercept from FItarget. Best for offset adjustment
    rlm_logFIi = fitlm(log10(FItarget),log10(FIref),'RobustOpts','on');
    targetSet_Log10MedAdj = [ones(size(FItarget,1),1),log10(FItarget)] * rlm_logFIi.Coefficients.Estimate;
    fprintf('\n *** Adjusting FItarget to FImedian using the method: regression*** \n')
end
newFItarget = power(10,targetSet_Log10MedAdj); % RESULT

%% PLOT
if plotOrNot == 1
    load('M2ScolorScheme.mat');
    M2S_figureH(0.6,0.4);
    %figure ('Position',[1 1 1122 287])
    set(gcf,'Name',['Adjustment of FItarget using the method "',FIadj_method,'"']), movegui(gcf,'center')
    
    subplot(1,3,1), plot(log10(FIref),log10(FItarget),'.k'), axis tight; grid, hold on, xlabel('Log10 FIref'),ylabel('Log10 FItarget');
    plot([nanmin([log10(FIref);log10(FItarget)]);nanmax([log10(FIref);log10(FItarget)])],...
        [nanmin([log10(FIref);log10(FItarget)]);nanmax([log10(FIref);log10(FItarget)])],'-','LineWidth',2,'Color',M2Scolor.dblue);
    
    if strcmp(FIadj_method,'median')
        subplot(1,3,2), plot(sort(log10(FItarget)-log10(FIref)),'.k'), axis tight; grid, xlabel('FI ratios sorted order'), ylabel('Log10(FItarget) - Log10(FIref)');
        subplot(1,3,2), hold on, plot(round(length(FIref)/2),nanmedian(log10(FItarget)-log10(FIref)),'o','Color',M2Scolor.orange);
        text( round(length(FIref)/2),nanmedian(log10(FItarget)-log10(FIref)),['FI offset = ',num2str(log10Med_offset)],'VerticalAlignment','top');
    
    elseif contains(FIadj_method,'regression')
        subplot(1,3,2), hold on, plot(log10(FItarget),log10(FIref),'.k'), hold on, axis tight, grid on   
        ylabel('Log10 FIref'),xlabel('Log10 FItarget');
        plot(xlim',[[1;1],xlim'] * rlm_logFIi.Coefficients.Estimate,'-','LineWidth',2,'Color',M2Scolor.orange)
        plot([nanmin([log10(FIref);log10(FItarget)]);nanmax([log10(FIref);log10(FItarget)])],...
        [nanmin([log10(FIref);log10(FItarget)]);nanmax([log10(FIref);log10(FItarget)])],'-','LineWidth',2,'Color',M2Scolor.dblue);
    end
    
    
    subplot(1,3,3), subplot(1,3,3), plot(log10(FIref),log10(FItarget),'.k','MarkerSize',5); hold on
    plot(log10(FIref),log10(newFItarget),'.','Color',M2Scolor.orange);
    
    axis tight; grid, hold on, xlabel('Log10 FIref'),ylabel('Log10 FItarget adjusted (in orange)'),% title('Log10 FI medians adjusted')
     plot([nanmin([log10(FIref);log10(newFItarget)]);nanmax([log10(FIref);log10(newFItarget)])],...
        [nanmin([log10(FIref);log10(newFItarget)]);nanmax([log10(FIref);log10(newFItarget)])],'-','LineWidth',2,'Color',M2Scolor.dblue);
    drawnow
end






