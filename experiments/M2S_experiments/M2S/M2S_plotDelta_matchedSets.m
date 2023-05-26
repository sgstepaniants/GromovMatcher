%% This is a plot of results of matching. Can be used with matched datasets
%s with multiple or unique matches.
% Input data are two matrices with [RT, MZ, FI} of reference and target.
% The number of rows must be the same in both.

function M2S_plotDelta_matchedSets(refMatched,targetMatched,Xsymbol,plotRow)

if nargin == 2
    %createFigOrNot = 1;
    M2S_figureH(0.8,0.9) 
    Xsymbol = '.k';
    plotRow = 2;
elseif nargin == 3
    plotRow = 2;
end


subplot(plotRow,3,1); hold on
plot(refMatched(:,1),targetMatched(:,1)-refMatched(:,1), Xsymbol)
grid on, axis tight
xlabel('RT reference'), ylabel('RTdist')

subplot(plotRow,3,2); hold on
plot(refMatched(:,2),targetMatched(:,2)-refMatched(:,2), Xsymbol)
grid on, axis tight
xlabel('MZ reference'), ylabel('MZdist')

subplot(plotRow,3,3); hold on
plot(log10(refMatched(:,3)),log10(targetMatched(:,3))-log10(refMatched(:,3)), Xsymbol)
grid on, axis tight
xlabel('log10 FI reference'), ylabel('log10 FIdist')

if plotRow == 2
    subplot(plotRow,3,4); hold on
    plot(refMatched(:,1),targetMatched(:,1), Xsymbol)
    grid on, axis tight
    title('RT'), xlabel('RT reference'), ylabel('RT target')

    subplot(plotRow,3,5); hold on
    plot(refMatched(:,2),targetMatched(:,2), Xsymbol)
    grid on, axis tight
    title('MZ'), xlabel('MZ reference'), ylabel('MZ target')

    subplot(plotRow,3,6); hold on
    plot(log10(refMatched(:,3)),log10(targetMatched(:,3)), Xsymbol)
    grid on, axis tight
    title('log10 FI'), xlabel('log10 FI reference'), ylabel('log10 FI target')
end
