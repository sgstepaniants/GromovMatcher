function Hplot = M2S_plotMZRT_featureSet(featureSet,FIcolorOrNot,marker_size,log10orNot)
% M2S_plotMZRT_featureSet: plot MZ vs RT of a metabolomics dataset colored by FI
%
% Hplot = plotMZRT_featureSet(featureSet,FIcolorOrNot,marker_size,log10orNot)
%
% INPUT:
% featureSet: 3 columns (RT in minutes, MZ, FI)
% FIcolorOrNot = 0: no;  1: yes (defaut)
% marker_size: integer for marker size
% log10orNot: 0: do not log10 scale; 1: use log10 scale (default); if
% log10orNot has more than one value, it is used to color the plot
%   
% The following can be used:
% plotMZRT_featureSet(featureSet)
% plotMZRT_featureSet(featureSet,FIcolorOrNot,marker_size,log10orNot)
%   
% OUTPUT:
% Hplot: the handle for the plot
%   
% M2S toolbox to match LCMS metabolomics features of untargeted datasets.
% *** Rui Climaco Pinto ***
% *** Imperial College London, 2021 ***



load('M2ScolorScheme.mat');

if nargin == 1
    FIcolorOrNot = 0;
    marker_size = 12;
    log10orNot = 1;
elseif nargin == 2
    if length(FIcolorOrNot) == size(featureSet,1)
        colorForPlot = FIcolorOrNot;
    end
    marker_size = 12;
    log10orNot = 0;
elseif nargin == 3
    if length(FIcolorOrNot)>1
        colorForPlot = FIcolorOrNot;
    end
    log10orNot = 0;
end

if length(FIcolorOrNot) == 1
    if FIcolorOrNot == 1
        if log10orNot == 1
            Hplot = scatter(featureSet(:,1),featureSet(:,2),marker_size*ones(size(featureSet,1),1),log10(featureSet(:,3)),'filled')
            disp('NOTE: FI (feature intensity) was log10 transformed to color the points')
        elseif length(log10orNot) == size(featureSet,1)
            Hplot = scatter(featureSet(:,1),featureSet(:,2),marker_size*ones(size(featureSet,1),1),featureSet(:,3),'filled')
            disp('NOTE: FI (feature intensity) was NOT log10 transformed to color the points')
        else
            disp('log10orNot must be 0, 1 or a vector with same length as the number of rows of featureSet')
        end
        colorbar
    else
    plot(featureSet(:,1),featureSet(:,2),'.k','MarkerSize',marker_size)
    end
else
    if log10orNot == 1
        Hplot = scatter(featureSet(:,1),featureSet(:,2),marker_size*ones(size(featureSet,1),1),log10(colorForPlot),'filled')
        disp('NOTE: color for plot was log10 transformed to color the points')
    elseif length(log10orNot) == size(featureSet,1)
        Hplot = scatter(featureSet(:,1),featureSet(:,2),marker_size*ones(size(featureSet,1),1),colorForPlot,'filled')
        disp('NOTE: color for plot was NOT log10 transformed to color the points')
    else
        disp('log10orNot must be 0, 1 or a vector with same length as the number of rows of featureSet')
    end
end
    
xlabel('RT (minutes)'),ylabel('MZ (m/z units)'),axis tight, grid on
title('MZ vs RT colored by log10FI')
colormap(M2Scolormap)
