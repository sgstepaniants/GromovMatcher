function [penaltyScores] = M2S_defineW_getScores(refSet,targetSet,adjResiduals_X,W,plotOrNot)


if nargin == 3
    W = [1,1,0];
    plotOrNot = 1;
elseif nargin == 4
    plotOrNot = 1;
end
    
%% ADJUST THE WEIGHTS OF EACH DIMENSION
% The penalised scores RTMZdist_medNeigh_to_TRpair_scaled are the 
% weighted sum of squares of the residuals. For a weight of [1,1,1] the
% penalisation score for the percentile value is 1.
% The weight definition also allows to delete one of the dimensions by
% setting the weight of that dimension to zero


% Residuals_X_i = RTMZFIdist_medNeigh_to_TRpair;
% Median_X = nanmedian(Residuals_X_i);
% Residuals_X_centered = Residuals_X_i - Median_X;
% absResid_atPercentile = prctile(abs(Residuals_X_centered),residPercentile);% absolute residual value
% The penaltyScores were previously named RTMZdist_medNeigh_to_TRpair_scaled

% Transform FImed TO LOG10 
refSet(:,3) = log10(refSet(:,3));
targetSet(:,3) = log10(targetSet(:,3));

% Wi = W;
Median_X = nanmedian(adjResiduals_X);
% if nansum(W) == 0 % give RT, MZ, FI equal value for the most extreme value in each of the three dimensions
%     %NOTE: if one of the dimensions is not desired, use NaN, e.g. [0,0,NaN]
% %     Wi = W;    
%     W = (nanmax(nanmax( (abs(adjResiduals_X(:,~isnan(Wi))) ))) *ones(1,3)) ./ nanmax(abs(adjResiduals_X)); 
%     W(isnan(Wi)) = 0;  
% end

penaltyScores = NaN(size(adjResiduals_X,1),1);% THESE ARE THE FINAL PENALTY SCORES!!
for f = 1:size(adjResiduals_X,1)
        penaltyScores(f,1) = sqrt(adjResiduals_X(f,:) * (repmat(W,3,1).*eye(3)) * (adjResiduals_X(f,:))');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLOTS ******************************************************
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if plotOrNot == 1
    load ('M2ScolorScheme.mat')
    M2S_figureH(0.8,0.45);
    set(gcf, 'Name',['Weighted residuals used to calculate penalisation scores. The weights are: ',num2str(W)])
    xlabel_all = {'RT of reference feature','MZ of reference feature','FI of reference feature'};
    ylabel_all = {'RT weighted residuals','MZ weighted residuals','FI weighted residuals'};
    ylim_all=[];
    for a=1:3
%         if ~isnan(Wi(a))
        subplot(1,3,a),plot(refSet(:,a),W(a)*adjResiduals_X(:,a),'.k'), axis tight, hold on, ylim_all = [ylim_all;ylim]; grid on
        plot(xlim',[Median_X(a);Median_X(a)],':r');
        plot(xlim',W(a)*[1;1],'-r'),plot(xlim',W(a)*[-1;-1],'-r');
        xlabel(xlabel_all{a}); ylabel(ylabel_all{a});
%         else
%             subplot(1,3,a),title('NOT USED')
%         end
    end
    ylim_all_max = nanmax(ylim_all(:,2)); ylim_all_min = nanmin(ylim_all(:,1));
    for a=1:3; subplot(1,3,a); ylim([ylim_all_min,ylim_all_max]);end
    
    

    % Figure RTdist, MZdist, FI colored by scores
    M2S_figureH(0.95,0.45);
    set(gcf,  'Name','Distances in each domain colored by penalisation scores')
    subplot(1,3,1)
    scatter(refSet(:,1),targetSet(:,1)-refSet(:,1),15*ones(size(refSet,1),1),penaltyScores,'filled'),colorbar, hold on
    xlabel('RT reference (minutes)'), ylabel('RTdist(Minutes)'), grid on, axis tight
    subplot(1,3,2)
    scatter(refSet(:,2),targetSet(:,2)-refSet(:,2),15*ones(size(refSet,1),1),penaltyScores,'filled'),colorbar, hold on
    xlabel('MZ reference (Minutes)'), ylabel('MZdist (Dalton)'), grid on, axis tight
    subplot(1,3,3)
    scatter(refSet(:,3),targetSet(:,3),15*ones(size(refSet,1),1),penaltyScores,'filled'),colorbar, hold on
    xlabel('FI reference (Log10)'), ylabel('FI target (Log10)'), grid on, axis tight
    
    
    colormap(M2Scolormap);
end
