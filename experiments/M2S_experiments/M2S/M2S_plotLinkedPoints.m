function M2S_plotLinkedPoints(allDotsX,allDotsY,allDotsLinkingIdx,markerTypes,lineTypes,lineColor,xylabels)

if isempty(markerTypes)
    markerTypes = '.k';
end
if isempty(lineTypes)
    lineTypes = '-';
end
if isempty(lineColor)
    lineColor = 'k';
end

% Plots

plot(allDotsX,allDotsY,markerTypes,'MarkerSize',3)

hold on
if length(xylabels)==2
    xlabel(xylabels{1});ylabel(xylabels{2});
end

% find multiple dot links
uniqueDotsLinkingIdx = unique(allDotsLinkingIdx);

counterX=1;
multIndexes={};
for a=1:length(uniqueDotsLinkingIdx)
    if sum(ismember(allDotsLinkingIdx,uniqueDotsLinkingIdx(a)))>1
        multIndexes{counterX,1} = find(ismember(allDotsLinkingIdx,uniqueDotsLinkingIdx(a))) ;
        counterX=counterX+1;
    end
end

% draw all links
for b = 1 : size(multIndexes,1)
    hold on, plot(allDotsX(multIndexes{b,1}),allDotsY(multIndexes{b,1}),lineTypes,'Color',lineColor)
end



