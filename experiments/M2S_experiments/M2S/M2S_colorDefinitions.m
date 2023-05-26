%% Plot properties
% 'MarkerEdgeColor'
% 'MarkerFaceColor'

% color by labels
M2Scolor.black = [0, 0, 0];
M2Scolor.dblue = 1/256 * [28, 83, 92];
M2Scolor.lblue = 1/256 * [1, 164, 205];
M2Scolor.lgrey = 1/256 * [191, 191, 191];
M2Scolor.yellow = 1/256 * [245,211,0];
M2Scolor.orange = 1/256 * [249, 88, 0];

% color by cell
M2ScolorsCell{1,1} = M2Scolor.black;
M2ScolorsCell{2,1} = M2Scolor.dblue;
M2ScolorsCell{3,1} = M2Scolor.lblue;
M2ScolorsCell{4,1} = M2Scolor.lgrey;
M2ScolorsCell{5,1} = M2Scolor.yellow;
M2ScolorsCell{6,1} = M2Scolor.orange;

% visualise
figure, hold on
plot([0;10],[1;1],'-','LineWidth',10,'Color',M2Scolor.black)
plot([0;10],[2;2],'-','LineWidth',10,'Color',M2Scolor.dblue)
plot([0;10],[3;3],'-','LineWidth',10,'Color',M2Scolor.lblue), 
plot([0;10],[4;4],'-','LineWidth',10,'Color',M2Scolor.lgrey)
plot([0;10],[5;5],'-','LineWidth',10,'Color',M2Scolor.yellow)
plot([0;10],[6;6],'-','LineWidth',10,'Color',M2Scolor.orange)

% create colormap
intervalX=52;
M2Scolormap = [linspace(M2ScolorsCell{1,1}(1),M2ScolorsCell{2,1}(1),intervalX)', linspace(M2ScolorsCell{1,1}(2),M2ScolorsCell{2,1}(2),intervalX)', linspace(M2ScolorsCell{1,1}(3),M2ScolorsCell{2,1}(3),intervalX)']
for colorNr = 2:5
    CMtemp =  [linspace(M2ScolorsCell{colorNr,1}(1),M2ScolorsCell{colorNr+1,1}(1),intervalX)', linspace(M2ScolorsCell{colorNr,1}(2),M2ScolorsCell{colorNr+1,1}(2),intervalX)', linspace(M2ScolorsCell{colorNr,1}(3),M2ScolorsCell{colorNr+1,1}(3),intervalX)']
    M2Scolormap = [M2Scolormap;CMtemp(2:end,:)];
end

% save M2ScolorScheme M2Scolormap M2Scolor 

% To use it:
load ('M2ScolorScheme.mat')

% for continuous plot colors
colormap(M2Scolormap)

