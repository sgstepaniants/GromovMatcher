% [adjResiduals_X,residPercentile] = M2S_adjustResiduals(refSet,targetSet,Residuals_X,residPercentile)
% Function M2S_adjustResiduals
% This function is part of M2S toolbox to match metabolomics features across untargeted datasets.
% 
% Standardise the residuals in a similar fashion to dividing by stdev.
%
% - The residuals are in different units and need to be standardised in 
% order to be combined to calculate the penalisation scores for each match.
% This is done in a similar way to z-scoring (divide by stdev), by dividing
% by the MAD of the residuals, or by a specific value of residuals in each dimension.
% After this division, the value of the residual at the defined point is 1
% so it is easy to visualise the relevance of each of the dimension's residuals (RT/MZ/log10FI).
%
% There are several ways to call this function:
% 1. Using the default "mad" method to find the residual. Different value is found for each dimension.
% [adjResiduals_X,residPercentile] = M2S_adjustResiduals(refSet,targetSet,Residuals_X);
%
% 2. Defining the factor to divide the residuals by (residPercentile) 
% opt.adjustResiduals.residPercentile = NaN % Same as the default (double MAD)
% opt.adjustResiduals.residPercentile = 0 % Percentile defined by estimating number of poor matches. Same percentile for all dimensions.
% opt.adjustResiduals.residPercentile = 95;% Percentile value (e.g. 95) defined by user, can be between ]0,100] 
% opt.adjustResiduals.residPercentile = [0.05,0.004,1.2]; % A residual value user-defined by inspecting figure "Residuals for RT adn MZ" 
%
% [adjResiduals_X,residPercentile] = M2S_adjustResiduals(refSet,targetSet,Residuals_X,opt.adjustResiduals.residPercentile)
%

% *** Rui Climaco Pinto ***
% *** Imperial College London, 2021 ***

function [adjResiduals_X,residPercentile] = M2S_adjustResiduals(refSet,targetSet,Residuals_X,residPercentile)

fprintf('\n Started function M2D_adjustResiduals\n ')

% If default settings:
if nargin <= 3
    residPercentile = NaN;
end

% Transform FImed to Log10FI:
refSet(:,3) = log10(refSet(:,3));
targetSet(:,3) = log10(targetSet(:,3));


%% Calculate the value of the residuals at the defined percentile

%%  A - (Default) determination of residuals percentile separately in each dimension using MAD
% Needs only the Residuals_X

if sum (isnan(residPercentile))>0 % IF there is at least one NaN in residPercentile
    disp('Method choice: residPercentile automatically defined using a threshold point')
    percentileThresh = NaN(1,3); residPercentile=NaN(1,3);
    [S_RT,percentileThresh(1,1)] = M2S_threshSelection((Residuals_X(:,1)),'mad',3,0);
    residPercentile(1,1)= 100 *percentileThresh(1,1);
    [S_MZ,percentileThresh(1,2)] = M2S_threshSelection((Residuals_X(:,2)),'mad',3,0);
    residPercentile(1,2)= 100 *percentileThresh(1,2);
    [S_logFI,percentileThresh(1,3)] = M2S_threshSelection((Residuals_X(:,3)),'mad',3,0);
    residPercentile(1,3)= 100 *percentileThresh(1,3);   
    absResid_atPercentile = [S_RT.S_limit, S_MZ.S_limit,S_logFI.S_limit];% absolute residual value
    % Normalize residuals by dividing by the residual value at specified percentile (similar to z-scoring)
    adjResiduals_X = Residuals_X ./ repmat(absResid_atPercentile,size(Residuals_X,1),1);

%% IF there are no NaN in residPercentile
elseif sum (isnan(residPercentile))==0 
    
    %% B - User-defined residual values for each dimension (which will become 1)
    if length(residPercentile)==3 % IF all values in residPercentile are defined
        disp('Method choice: values selected from plot of residuals used to standardise residuals')
        % NOTE: the values of y are defined by the user. It helps to look
        % at the residuals plot. There are no calculations in this case
    
    %% C - Input is a single percentile value for all dimensions
    % IF input is single value > 0 and < 100 then it is a percentile
    elseif (length(residPercentile) == 1 && residPercentile >0 && residPercentile <= 100) 
        disp('Method choice: residPercentile user defined, same for 3 dimensions')
        %residPercentile = repmat(residPercentile,1,3);
  
    else 
        disp(' NOTE: residPercentile was not well defined')
        disp(' It can be NaN, zero, a number in ]0,1] or a row vector with 3 numbers')
    end
    
    
          
    %%  IF (C) is chosen, user-defined residual value is the input:
    if (sum (isnan(residPercentile')))==0 && length(residPercentile)==1
        absResid_atPercentile = [prctile(abs(Residuals_X(:,1)),residPercentile(1,1)),...
                                prctile(abs(Residuals_X(:,2)),residPercentile(1,1)),...
                                prctile(abs(Residuals_X(:,3)),residPercentile(1,1))];% absolute residual value
    % OTHERWISE B WAS CHOSEN
    else 
        
        absResid_atPercentile = residPercentile;
    end
    % Normalize residuals by dividing by the residual value at specified percentile (similar to z-scoring)
    adjResiduals_X = Residuals_X ./ repmat(absResid_atPercentile,size(Residuals_X,1),1);
end


% Account for Inf
for a=1:size(adjResiduals_X,2)
    tempVec = adjResiduals_X(~isinf(adjResiduals_X(:,a)),a);
    adjResiduals_X(isinf(adjResiduals_X(:,a)),a) = nanmax(tempVec);
end



%% PLOTS
if sum (isnan(residPercentile))==0 && length(residPercentile)==3 % This IF does not perform correctly ** INCORRECT!! **
    M2S_figureH(0.8,0.8);    set(gcf, 'Name',['Residuals values  ',num2str(residPercentile)]);
else
    M2S_figureH(0.8,0.8);    set(gcf,'Name',['Residuals equal at percentile  ',num2str(residPercentile)]);
end
a=1; subplot(2,3,a),plot(refSet(:,a),Residuals_X(:,a),'.k','MarkerSize',8), axis tight, hold on, xlim1 = xlim; grid on
plot(xlim',[0;0],'-k')
plot(xlim',[absResid_atPercentile(a);absResid_atPercentile(a)],'-r'),plot(xlim',[-absResid_atPercentile(a);-absResid_atPercentile(a)],'-r')
xlabel('RT of reference feature'), ylabel('RT centered residuals') 
a=2; subplot(2,3,a),plot(refSet(:,a),Residuals_X(:,a),'.k','MarkerSize',8), axis tight, hold on, xlim1 = xlim; grid on
plot(xlim',[0;0],'-k')
plot(xlim',[absResid_atPercentile(a);absResid_atPercentile(a)],'-r'),plot(xlim',[-absResid_atPercentile(a);-absResid_atPercentile(a)],'-r')
xlabel('MZ of reference feature'), ylabel('MZ centered residuals')
a=3; subplot(2,3,a),plot(refSet(:,a),Residuals_X(:,a),'.k','MarkerSize',8), axis tight, hold on, xlim1 = xlim; grid on
plot(xlim',[0;0],'-k')
plot(xlim',[absResid_atPercentile(a);absResid_atPercentile(a)],'-r'),plot(xlim',[-absResid_atPercentile(a);-absResid_atPercentile(a)],'-r')
xlabel('log10FI of reference feature'), ylabel('log10FI centered residuals')


absResidIsOne_atPercentileDefined= [1,1,1];%prctile(abs(adjResiduals_X),residPercentile); % all values are equal to 1 after dividing by the percentile (similar to dividing by stdev)
% find min and max for the plot in the three plots
[minForPlot,~] = nanmin(nanmin(adjResiduals_X)); 
[maxForPlot,~] = nanmax(nanmax(adjResiduals_X)); 

a=1; subplot(2,3,3+a), plot(refSet(:,a), adjResiduals_X(:,a),'.k','MarkerSize',8), axis tight, hold on, xlim1 = xlim; grid on
plot(xlim1',[0;0],'-k')
plot(xlim1',[absResidIsOne_atPercentileDefined(a);absResidIsOne_atPercentileDefined(a)],'-r')
plot(xlim1',[-absResidIsOne_atPercentileDefined(a);-absResidIsOne_atPercentileDefined(a)],'-r')
xlabel('RT of reference feature'), ylabel('RT standardized residuals'); ylim([minForPlot,maxForPlot])
a=2; subplot(2,3,3+a), plot(refSet(:,a), adjResiduals_X(:,a),'.k','MarkerSize',8), axis tight, hold on, xlim1 = xlim; grid on
plot(xlim1',[0;0],'-k')
plot(xlim1',[absResidIsOne_atPercentileDefined(a);absResidIsOne_atPercentileDefined(a)],'-r'),
plot(xlim1',[-absResidIsOne_atPercentileDefined(a);-absResidIsOne_atPercentileDefined(a)],'-r')
xlabel('MZ of reference feature'), ylabel('MZ standardized residuals'); ylim([minForPlot,maxForPlot])
a=3; subplot(2,3,3+a), plot(refSet(:,a), adjResiduals_X(:,a),'.k','MarkerSize',8), axis tight, hold on, xlim1 = xlim; grid on
plot(xlim1',[0;0],'-k')
plot(xlim1',[absResidIsOne_atPercentileDefined(a);absResidIsOne_atPercentileDefined(a)],'-r')
plot(xlim1',[-absResidIsOne_atPercentileDefined(a);-absResidIsOne_atPercentileDefined(a)],'-r')
xlabel('log10FI of reference feature'), ylabel('log10FI standardized residuals'); ylim([minForPlot,maxForPlot])
