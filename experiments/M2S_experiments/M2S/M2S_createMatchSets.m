function [ref_MatchSet,target_MatchSet,Xr_connIdx_inMatchSet,Xt_connIdx_inMatchSet] =...
    M2S_createMatchSets(refSet,targetSet,Xr_connIdx,Xt_connIdx,plotOrNot)

%% 4. create Match ref/target datasets 

ref_MatchSet = refSet;
target_MatchSet = targetSet;
Xr_connIdx_inMatchSet = Xr_connIdx;
Xt_connIdx_inMatchSet = Xt_connIdx;

% to use only unique connections:
[~, ic1, iu1] = unique(refSet,'rows','stable');
count1 = histcounts(iu1, 1:numel(ic1));
ic1(count1>1)=[];

[~, ic2, iu2] = unique(targetSet,'rows','stable');
count2 = histcounts(iu2, 1:numel(ic2));
ic2(count2>1)=[];

ic12 = intersect(ic1,ic2); % rows of features without multiples 

ref_MatchSet = refSet(ic12,:);
Xr_connIdx_inMatchSet=Xr_connIdx(ic12,:);
target_MatchSet = targetSet(ic12,:);
Xt_connIdx_inMatchSet=Xt_connIdx(ic12,:);

if plotOrNot == 1
    limitsPlot = max([max(refSet(:,1:2));max(targetSet(:,1:2))]);
    figure('units','Normalized','Position',[0.01 0.01 0.7,0.7],'Name','Top: all features    Bottom: only features with a single match, used to calculate neighbours')
    movegui(gcf,'center')
    subplot(2,2,1), plot(refSet(:,1),refSet(:,2),'.k'), title('refSet')
    xlabel('RT (minutes)'),ylabel('MZ (m/z units)'),axis([0 limitsPlot(1) 0 limitsPlot(2)]) , hold on, grid on
    subplot(2,2,2), plot(targetSet(:,1),targetSet(:,2),'.r'), title('targetSet')
    xlabel('RT (minutes)'),ylabel('MZ (m/z units)'),axis([0 limitsPlot(1) 0 limitsPlot(2)]) , hold on, grid on
    subplot(2,2,3), plot(ref_MatchSet(:,1),ref_MatchSet(:,2),'.k'), title('refMatchSet')
    xlabel('RT (minutes)'),ylabel('MZ (m/z units)'),axis([0 limitsPlot(1) 0 limitsPlot(2)]) , hold on, grid on
    subplot(2,2,4), plot(target_MatchSet(:,1),target_MatchSet(:,2),'.r'), title('targetMatchSet')
    xlabel('RT (minutes)'),ylabel('MZ (m/z units)'),axis([0 limitsPlot(1) 0 limitsPlot(2)]) , hold on, grid on
    
end