% M2S_colorByY_ofSubplot(subplotNr,figH)
% Function M2S_colorByY_ofSubplot
% This function is part of M2S toolbox to match metabolomics features across untargeted datasets.
% 
% Colour all subplots in a figure by teh y-values of one one those plots,
% or by a colour vector.
%
% There are several ways to call this function:
% 1. M2S_colorByY_ofSubplot(3,1) % colours subplots in fig1 according to 
% y-values of subplot 3
% 2. M2S_colorByY_ofSubplot(3,gcf) % colours subplots in current figure (gcf) 
% according to y-values of subplot 3
% 3. M2S_colorByY_ofSubplot(3) % colours subplots in current figure (gcf) 
% according to y-values of subplot 3
% 3. M2S_colorByY_ofSubplot(colVec) % colours subplots in current figure (gcf)
% according to the values in colVec
%
% *** Rui Climaco Pinto ***
% *** Imperial College London, 2021 ***


function M2S_colorByY_ofSubplot(subplotNr,figH)
load ('M2ScolorScheme.mat');

if nargin == 1
    figH = gcf;
end


% Find number of subplots
n=numel(figH.Children);

% Find how many subplot rows and columns
axx = findobj(figH,'type','axes');
if n > 1
    pos = cell2mat(get(axx,'position'));
else
    pos = get(axx,'position');
end
nrows = numel(unique(pos(:,2))); % the same Y position means the same row
ncols = numel(unique(pos(:,1))); % the same X position means the same column

if length(subplotNr) == 1
% Get the y values of the desired subplot, to use as colours

    subplot(nrows,ncols,subplotNr)
    ax = gca; 
    h = findobj(gca,'Type','line');
    x = (h(1).XData)'; 
    y = (h(1).YData)'; 
    hold on, scatter(x,y,20*ones(size(x)),y,'filled')
    colormap(M2Scolormap)

    pause(1)
    %% Colour the other subplots using the same colours as above
    not_subplotNr = setdiff(1:n,subplotNr);


    for s = 1:length(not_subplotNr)
        subplot(nrows,ncols,not_subplotNr(s))
        ax = gca; 
        h1 = findobj(gca,'Type','line');
        x1 = (h1(1).XData)'; 
        y1 = (h1(1).YData)'; 
        hold on, scatter(x1,y1,20*ones(size(x1)),y,'filled')
    end
else
    for s = 1:n
        subplot(nrows,ncols,s)
        ax = gca; 
        h1 = findobj(gca,'Type','line');
        x1 = (h1(1).XData)'; 
        y1 = (h1(1).YData)'; 
        hold on, scatter(x1,y1,20*ones(size(x1)),subplotNr,'filled')
    end
end
