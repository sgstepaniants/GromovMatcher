function figureH = M2S_figureH(widthPercent,heightPercent,figH)
%% M2S_figureH
% Create or position a figure in the centre of the screen
% 
% figureH = M2S_figureH(widthPercent,heightPercent,figH)
%
% INPUT:
% To create:
% figureH = M2S_figureH(widthPercent,heightPercent)
% To position
% figureH = M2S_figureH(widthPercent,heightPercent,figH)
% NOTE: widthPercent and heightPercent are between 0 and 1
%
% OUTPUT
% figureH: figure handle
%
% M2S toolbox to match LCMS metabolomics features of untargeted datasets.
% *** Rui Climaco Pinto ***
% *** Imperial College London, 2021 ***


% Ensure root units are pixels and get the size of the screen:
set(0,'Units','pixels') 
scnsize = get(0,'ScreenSize');

%Define the size and location of the figures:
pos1  = [round(0.5*(1-widthPercent) * scnsize(3)),... 
    round(0.5*(1-heightPercent) *scnsize(4)),...
    round(widthPercent * scnsize(3)),...
    round(heightPercent * scnsize(4))];

if nargin==2
    %Create the figures:
    figure('OuterPosition',pos1) 
    figureH=gcf;
    
else
    figureH = figH;
    set(figH,'OuterPosition',pos1) 
end
