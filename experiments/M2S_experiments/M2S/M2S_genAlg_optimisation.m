%% This function starts at at a central point, then increases thresholds equally in both directions.
% A population contains a number of N subjects (e.g. 10).
% Each subject is composed of 4 parameters (which define an acceptance area 
% for matches in the RTdist vs RT and in the MZdist vs MZ dimensions).
% Parameters are [RTdist centre; MZdist centre; RTdist height; MZdist height]
% The "height" parameter is the distance from the centre of the region to 
% each of the acceptance threshold flat lines above and below the centre.
% This is applied both to RTdist and MZdist.
% The costFunction is maximised for:
% Cost = nSingleMatches+nMultMatches-extraFeaturesInMultMatchClusters_SSQ
%
% Uses:
% [genAlg_Res,optBest] = M2S_genAlg_optimisation(refSet, targetSet, [],[],[],1)
% [genAlg_Res,optBest] = M2S_genAlg_optimisation(refSet, targetSet, RTdist, MZdist, opt, plotOrNot)

function [genAlg_Res,optBest] = M2S_genAlg_optimisation(refSet, targetSet, RTdist, MZdist, opt, plotOrNot)

% rng(101)

%% Definitions
useSlope = 1; % Diagonal lines will be calculated as thresholds 

% nSeeds are used as a basis for the definition of the initial population
def_opt.nSeedsRT = 3;
def_opt.nSeedsMZ = 5;

% This divides the RTdist or MZdist regions in a number of sub-regions
def_opt.numberIncrements = 21;

def_opt.n_initial_increments = 3; % defines the number of initial areas above and below the seed points in RT and MZ

% The settings that stop the iterations:
def_opt.max_genNr = 100;% max number of iterations allowed. 
% If it is reached, the algorithm did not find a minumum and stops.
def_opt.max_bestOneCount = ceil(0.1*def_opt.max_genNr);% successive number of generations the same individual is chosen. 
% If it is reached, it found a minimum and stops.

% NOTE: populationSize = 2 + opt.nCrossover_descendents + opt.nMutation_descendents;
def_opt.nCrossover_descendents = 3;
def_opt.nMutation_descendents = 3;
def_opt.mutationProb=0.5; %Probability of mutation in each position (]0:1])

% The maximisation type defines the COST function to select the best model
% It sets the multiple match penalisation of each individual in a
% generation, obtaining the nAdjustedMatches value.
% nAdjustedMatches = nSingleMatches - nExtraFeaturesInMultMatchClusters_SSQ + nMultMatch.
% MultMatch can take the value 0 (SingleM), 0.5 (HalfMultiM) or 1 (MultiM) per multiple match.
% The strength of multiple match penalisation increases as
% MultiM<HalfMultiM<SingleM, being the 'SingleM' the most penalised for
% the occurrence of multiple matches
def_opt.maximisationType = 'Other'; % {'Other','onlySingle','SingleM','HalfMultiM','MultiM'}


% [newOpt] = M2S_addOptions(def_opt,opt)

% Default definitions:

flag_slopeCalculated = 0; % diagonal lines have not been calculated
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plotOrNot=0; %{0,1,2} Plot nothing/Plot only main/Plot all.

if (isempty(RTdist) && isempty(MZdist))
    RTdist = targetSet(:,1) - refSet(:,1);
    MZdist = targetSet(:,2) - refSet(:,2);
    opt = def_opt;
    %plotOrNot = 1;
end    

if isempty(opt) == 1
    opt = def_opt;    
end

if nargin ~= 6
    error('Need to have 6 inputs')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

populationSize = 2 + opt.nCrossover_descendents + opt.nMutation_descendents;

%% Create a unique label for each of the features in both sets

[refMZRT_str_matched] = M2S_createLabelMZRT('ref',refSet(:,2),refSet(:,1));
[targetMZRT_str_matched] = M2S_createLabelMZRT('target',targetSet(:,2),targetSet(:,1));
% The graph is more efficient if it uses numerical nodes (below)
% refMZRT_str_matched_idx = (1:length(refMZRT_str_matched))';
% targetMZRT_str_matched_idx = max(refMZRT_str_matched_idx) + (1:length(targetMZRT_str_matched))';

%% Define the seed points for the centre of each subject in RT and MZ (centre of acceptance regions)
% These are the start point centre of each subject (region) in RTdist and MZdist.
% They are defined as the median of groups of values across the RT (or MZ)

% opt.nSeedsRT = 4;
RT_regions_boundaries = linspace(min(refSet(:,1)),max(refSet(:,1)),opt.nSeedsRT+1)';
RT_regions01 = zeros(length(refSet(:,1)),opt.nSeedsRT) == 1;
RTdist_seeds = NaN(opt.nSeedsRT,1);
RT_seeds_regionCentre = NaN(opt.nSeedsRT,1);
for b=1:length(RT_regions_boundaries)-1
    RT_regions01(:,b) = (refSet(:,1)>RT_regions_boundaries(b) & refSet(:,1)<=RT_regions_boundaries(b+1));
    RTdist_seeds(b,1) = median(RTdist(RT_regions01(:,b)));
    RT_seeds_regionCentre(b,1) =  median(refSet(RT_regions01(:,b),1));
end

% opt.nSeedsMZ = 5;
MZ_regions_boundaries = linspace(min(refSet(:,2)),max(refSet(:,2)),opt.nSeedsMZ+1)';
MZ_regions01 = zeros(length(refSet(:,2)),opt.nSeedsMZ) == 1;
MZdist_seeds = NaN(opt.nSeedsMZ,1);
MZ_seeds_regionCentre = NaN(opt.nSeedsMZ,1);
for b=1:length(MZ_regions_boundaries)-1
    MZ_regions01(:,b) = (refSet(:,2)>MZ_regions_boundaries(b) & refSet(:,2)<=MZ_regions_boundaries(b+1));
    MZdist_seeds(b,1) = median(MZdist(MZ_regions01(:,b)));
    MZ_seeds_regionCentre(b,1) =  median(refSet(MZ_regions01(:,b),2));
end

%% Define the number of increments 
% The entire RTdist space from "min RTdist" to "max RTdist" is divided in 
% "numberIncrements-1" regions. The same for MZdist.

% numberIncrements = 21;
incrementFraction = 1/opt.numberIncrements;% "height" of each region
RTdist_increment = 0.5 * range(RTdist)*incrementFraction;
MZdist_increment = 0.5 * range(MZdist)*incrementFraction;

% Plot the seeds
if plotOrNot >=1
    M2S_figureH(0.6,0.35);
    set(gcf,'Name','Seed points for threshold search')
    subplot(1,2,1), plot(refSet(:,1),RTdist,'.k');
    hold on, plot(RT_seeds_regionCentre,RTdist_seeds,'.r','markersize',35);
    xlabel('RT'); ylabel('RTdist'), grid on, axis tight; M2S_plotaxes('-k',[NaN,0])
    subplot(1,2,2), plot(refSet(:,2),MZdist,'.k');
    hold on, plot(MZ_seeds_regionCentre,MZdist_seeds,'.r','markersize',35);
    xlabel('MZ'); ylabel('MZdist'), grid on, axis tight; M2S_plotaxes('-k',[NaN,0])
    
    % disp('MAYBE PLOT THE AREAS?')
    MZ_seeds_regionCentre(b,1)
    %plotaxes('-r',
end

% Plot one of the seeds with three areas around it

% Plot the seeds
if plotOrNot >=1000
    seedNr = 2
    M2S_figureH(0.6,0.35);
    set(gcf,'Name','Example seed points and three height distances')
    subplot(1,2,1), plot(refSet(:,1),RTdist,'.k');
    hold on, plot(RT_seeds_regionCentre(seedNr),RTdist_seeds(seedNr),'.r','markersize',35);
    plotaxes('-r',[NaN,RTdist_seeds(seedNr,1)+1*RTdist_increment])
    plotaxes('-r',[NaN,RTdist_seeds(seedNr,1)+3*RTdist_increment])
    plotaxes('-r',[NaN,RTdist_seeds(seedNr,1)+5*RTdist_increment])
    plotaxes('-r',[NaN,RTdist_seeds(seedNr,1)-1*RTdist_increment])
    plotaxes('-r',[NaN,RTdist_seeds(seedNr,1)-3*RTdist_increment])
    plotaxes('-r',[NaN,RTdist_seeds(seedNr,1)-5*RTdist_increment])
    
    xlabel('RT'); ylabel('RTdist'), grid on, axis tight; M2S_plotaxes('-k',[NaN,0])
    subplot(1,2,2), plot(refSet(:,2),MZdist,'.k');
    hold on, plot(MZ_seeds_regionCentre(seedNr),MZdist_seeds(seedNr),'.r','markersize',35);
    plotaxes('-r',[NaN,MZdist_seeds(seedNr,1)+1*MZdist_increment])
    plotaxes('-r',[NaN,MZdist_seeds(seedNr,1)+3*MZdist_increment])
    plotaxes('-r',[NaN,MZdist_seeds(seedNr,1)+5*MZdist_increment])
    plotaxes('-r',[NaN,MZdist_seeds(seedNr,1)-1*MZdist_increment])
    plotaxes('-r',[NaN,MZdist_seeds(seedNr,1)-3*MZdist_increment])
    plotaxes('-r',[NaN,MZdist_seeds(seedNr,1)-5*MZdist_increment])
    xlabel('MZ'); ylabel('MZdist'), grid on, axis tight; M2S_plotaxes('-k',[NaN,0])
    
    % disp('MAYBE PLOT THE AREAS?')
    
    
end


%% Define the initial population
% - The population is a matrix (Nx4)
% - Variables are [RTdist centre; MZdist centre; RTdist height; MZdist height; RTdist_slope' MZdist_slope]
% - The subject's centres are randomly sampled from the seeds.
% - The height for each subject is defined as "opt.n_initial_increments" (which
% is multiplied by the RTdist_increment or MZdist_increment)
% populationSize = 10;
%% Start on the first seed point and increase equally the increment

% popState --> [RTdist centre, MZdist centre, RTdist height, MZdist height]
initState_RTseeds_idx = floor(opt.nSeedsRT*rand(populationSize,1))+1;
initState_MZseeds_idx = floor(opt.nSeedsMZ*rand(populationSize,1))+1;

% opt.n_initial_increments = 3; % used to define the size of the initial areas above and below the seed points in RT and MZ

% These are the initial population settings
popState = [RTdist_seeds(initState_RTseeds_idx), MZdist_seeds(initState_MZseeds_idx),...
    opt.n_initial_increments*RTdist_increment*ones(populationSize,1),opt.n_initial_increments*MZdist_increment*ones(populationSize,1),zeros(populationSize,1),zeros(populationSize,1)];


%% The maximum number of single and multiple matches is:

Gmax =graph(refMZRT_str_matched,targetMZRT_str_matched);% String nodes
Gmax.Nodes.CCnr = (conncomp(Gmax))';% the component each feature belongs to
G_max.nTotalClusters = max(Gmax.Nodes.CCnr); % number of clusters
nFeaturesInCC_max = tabulate(Gmax.Nodes.CCnr); % number of features in each cluster
nExtraFeaturesInCC_max = nFeaturesInCC_max(:,2)-2;
nClusters_withNnodes_max = tabulate(nFeaturesInCC_max(:,2));
% G_max.nSingleMatches = nClusters_withNnodes_max(nClusters_withNnodes_max(:,1)==2,2);
G_max.nMultMatchClusters = sum(nClusters_withNnodes_max(nClusters_withNnodes_max(:,1)>2,2));
%G_max.nFeaturesInSingleMatches = 2*Gmax.nSingleMatches(i,1);
G_max.nFeaturesInMultMatchClusters = sum(nClusters_withNnodes_max(nClusters_withNnodes_max(:,1)>2,1) .* nClusters_withNnodes_max(nClusters_withNnodes_max(:,1)>2,2));
G_max.inverse_nFeaturesInMultMatchClusters_minus1_SSQ = sum(1./((nFeaturesInCC_max(:,2)-1)).^2);
G_max.nExtraFeaturesInMultMatchClusters = sum(nExtraFeaturesInCC_max);
G_max.nExtraFeaturesInMultMatchClusters_SSQ = nExtraFeaturesInCC_max'*nExtraFeaturesInCC_max;

Gmax.Nodes.CCnr = (conncomp(Gmax))';
Gmax.Edges.CCnr = Gmax.Nodes.CCnr(M2S_find_idxInReference(Gmax.Edges.EndNodes(:,1),Gmax.Nodes.Name));
G_max.nEdges = tabulate(Gmax.Edges.CCnr); 

G_max.nEdges = G_max.nEdges(:,1:2);
% Find number of clusters with N nodes or N edges
% CC.freq_clustersWithNnodes = tabulate(CC.nNodes(:,2));
G_max.freq_clustersWithNedges = tabulate(G_max.nEdges(:,2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Max number of single matches is min number of features in ref or target
maxN_singleMatches = min([length(unique(refMZRT_str_matched));length(unique(refMZRT_str_matched))]);
maxN_extraFeatures = G_max.nExtraFeaturesInMultMatchClusters;
%maxN_extraFeatures = G_max.nExtraFeaturesInMultMatchClusters_SSQ;

% This is not used at the moment, but should be the best
temp_freq_clustersWithNedges = G_max.freq_clustersWithNedges;
temp_freq_clustersWithNedges(temp_freq_clustersWithNedges(:,1)==1,:)=[];
maxN_extraMatches = sum((temp_freq_clustersWithNedges(:,1)-1) .*temp_freq_clustersWithNedges(:,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Start the generations (iterations)

bestOne=0;% the best of the subjects found in this iteration
bestOneCount = 0;% the number of successive times this subject was chosen
rtmz_hits_idx=cell(populationSize,1);% idx of matches inside each subject's area
bestIdx_inGen = [];

if plotOrNot == 2
    fig1000 = figure('Visible','off')
    M2S_figureH(0.8,0.5,gcf);
%     set(gcf,'Visible','on')
%     figure(1000) ; set(gcf,'Position',[229 42 1699 954],'visible',false); movegui(gcf,'center')
elseif plotOrNot >0
    figure1001 = figure; 
    M2S_figureH(0.7,0.5,figure1001);
    %set(gcf,'Position',[ 299 295 1324 415]); movegui(gcf,'center')
end

%% For each generation:
genAlg_Res = table;
genNr = 1;

while bestOneCount < opt.max_bestOneCount && genNr <= opt.max_genNr
    %% For each individual    
    for c1=1:populationSize
         RTcoeffs = [popState(c1,1);popState(c1,5)];
         RTdist_pred = [ones(size(refSet,1),1),refSet(:,1)] * RTcoeffs;
         RTrtmz_hits_idx = find(abs(RTdist-RTdist_pred) < popState(c1,3));
         MZcoeffs = [popState(c1,2);popState(c1,6)];
         MZdist_pred = [ones(size(refSet,1),1),refSet(:,2)] * MZcoeffs;
         MZrtmz_hits_idx = find(abs(MZdist-MZdist_pred) < popState(c1,4));
         rtmz_hits_idx{c1,1} = intersect(RTrtmz_hits_idx,MZrtmz_hits_idx);
    end
    
    %% For each individual 
    % Count number of matches, single, multiple, etc
    G=struct;
    for c2=1:populationSize
        if ~isempty(rtmz_hits_idx{c2,1})
            %% The graph is more efficient if it uses numerical nodes
            Gtemp =graph(refMZRT_str_matched(rtmz_hits_idx{c2,1}),targetMZRT_str_matched(rtmz_hits_idx{c2,1}));% String nodes
            Gtemp.Nodes.CCnr = (conncomp(Gtemp))';
            G.nTotalClusters(c2,1) = max(Gtemp.Nodes.CCnr); % number of clusters
            nFeaturesInCC = tabulate(Gtemp.Nodes.CCnr);
            nExtraFeaturesInCC = nFeaturesInCC(:,2)-2;
            nClusters_withNnodes = tabulate(nFeaturesInCC(:,2));
            G.nSingleMatches(c2,1) = nClusters_withNnodes(nClusters_withNnodes(:,1)==2,2);
            G.nMultMatchClusters(c2,1) = sum(nClusters_withNnodes(nClusters_withNnodes(:,1)>2,2));
            G.nFeaturesInSingleMatches(c2,1) = 2*G.nSingleMatches(c2,1);
            G.nFeaturesInMultMatchClusters(c2,1) = sum(nClusters_withNnodes(nClusters_withNnodes(:,1)>2,1) .* nClusters_withNnodes(nClusters_withNnodes(:,1)>2,2));
            G.inverse_nFeaturesInMultMatchClusters_minus1_SSQ(c2,1) = sum(1./((nFeaturesInCC(:,2)-1)).^2);
            G.nExtraFeaturesInMultMatchClusters(c2,1) = sum(nExtraFeaturesInCC);
            G.nExtraFeaturesInMultMatchClusters_SSQ(c2,1) = nExtraFeaturesInCC'*nExtraFeaturesInCC;
            
            % Find edges properties
            Gtemp.Edges.CCnr = Gtemp.Nodes.CCnr(M2S_find_idxInReference(Gtemp.Edges.EndNodes(:,1),Gtemp.Nodes.Name));
            CC.nEdges = tabulate(Gtemp.Edges.CCnr); 
            G.nExtraMatchesInMultMatchClusters(c2,1) = sum(CC.nEdges(:,2)-1);
            G.nExtraMatchesInMultMatchClusters_SSQ(c2,1) = (CC.nEdges(:,2)-1)'*(CC.nEdges(:,2)-1);
%             CC.nEdges = CC.nEdges(:,1:2);
%             CC.freq_clustersWithNedges = tabulate(CC.nEdges(:,2));
%             temp_freq_clustersWithNedges = CC.freq_clustersWithNedges;
%             temp_freq_clustersWithNedges(temp_freq_clustersWithNedges(:,1)==1,:)=[];
%             G.nExtraMatchesInMultMatchClusters(c2,1) = sum((temp_freq_clustersWithNedges(:,1)-1) .*temp_freq_clustersWithNedges(:,2));
%             G.nExtraMatchesInMultMatchClusters_SSQ(c2,1)
            
            
        else
            G.nTotalClusters(c2,1) = 0;
            G.nSingleMatches(c2,1) = 0;
            G.nMultMatchClusters(c2,1) = 0;
            G.nFeaturesInSingleMatches(c2,1) = 0;
            G.nFeaturesInMultMatchClusters(c2,1) = 0;
            G.inverse_nFeaturesInMultMatchClusters_minus1_SSQ(c2,1) = 0;
            G.nExtraFeaturesInMultMatchClusters(c2,1)=0;
            G.nExtraFeaturesInMultMatchClusters_SSQ(c2,1)=0;
            
            G.nExtraMatchesInMultMatchClusters(c2,1) = 0;
            G.nExtraMatchesInMultMatchClusters_SSQ(c2,1) = 0;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% This is the cost function we want to maximise (the higher the better)
    % - It is the sum of the singleMatches with multipleMatchClusters,
    % penalised by extra features in multipleMatchClusters (>3).
    % 
    % - The number of extra features in a cluster (with min 2 features) is n-2.
    % It is 0 for single matches (2 features); It is 1 for a cluster with 3
    % features; It is 3 for a cluster with 5 features, etc.
    % - nExtraFeaturesInMultMatchClusters_SSQ is the sum of the squared number of extra features in each cluster
    % (not the square of the sum of extra features).
    % It is 1 for a cluster with 3 features; But it is 4 for a cluster with 5 features (3^2), etc.
    %
    % Possibilities, from less aggressive to more aggressive:
        % G.nAdjustedMatches = G.nSingleMatches + 0.5* G.nMultMatchClusters - G.nExtraFeaturesInMultMatchClusters; % A
        % G.nAdjustedMatches = G.nSingleMatches - (G.nMultMatchClusters - G.nExtraFeaturesInMultMatchClusters).^2; % B 
        % G.nAdjustedMatches = G.nSingleMatches + 0.5*G.nMultMatchClusters - G.nExtraFeaturesInMultMatchClusters_SSQ; % C
        % G.nAdjustedMatches = G.nSingleMatches - G.nExtraFeaturesInMultMatchClusters; % D more tight
        % G.nAdjustedMatches = G.nSingleMatches - (0.5*G.nMultMatchClusters - G.nExtraFeaturesInMultMatchClusters).^2; % E more tight
        % G.nAdjustedMatches = G.nSingleMatches - G.nExtraFeaturesInMultMatchClusters.^2; % F more tight
        % G.nAdjustedMatches = G.nSingleMatches - G.nExtraFeaturesInMultMatchClusters_SSQ.^2; % G more tight
    if strcmp(opt.maximisationType,'A')
        % Maximise single matches, half of clusters, penalise extra features
        G.nAdjustedMatches = G.nSingleMatches + 0.5* G.nMultMatchClusters - G.nExtraFeaturesInMultMatchClusters;
    elseif strcmp(opt.maximisationType,'B')
        % Maximise single matches, penalise large multiple match clusters:
        G.nAdjustedMatches = G.nSingleMatches + (G.nMultMatchClusters - G.nExtraFeaturesInMultMatchClusters).^2;
    elseif strcmp(opt.maximisationType,'C')
        % Maximise both single and multiple match clusters, penalise large clusters
        G.nAdjustedMatches = G.nSingleMatches + 0.5*G.nMultMatchClusters - G.nExtraFeaturesInMultMatchClusters_SSQ;
    elseif strcmp(opt.maximisationType,'D')
        % Maximise both single and multiple match clusters, penalise large clusters
        G.nAdjustedMatches = G.nSingleMatches - G.nExtraFeaturesInMultMatchClusters;
    elseif strcmp(opt.maximisationType,'E')
        % Maximise single matches, penalise penalise large clusters
        G.nAdjustedMatches = G.nSingleMatches - (0.5*G.nMultMatchClusters - G.nExtraFeaturesInMultMatchClusters).^2;
    elseif strcmp(opt.maximisationType,'F')
        G.nAdjustedMatches = G.nSingleMatches - G.nExtraFeaturesInMultMatchClusters.^2;
    elseif strcmp(opt.maximisationType,'G') 
        % Strongest penalisation
        G.nAdjustedMatches = G.nSingleMatches - G.nExtraFeaturesInMultMatchClusters_SSQ.^2;
        % Using ROC-type of strategy:
        %{
        x = (1/maxN_extraFeatures*G.nExtraFeaturesInMultMatchClusters);
        y = (1/maxN_singleMatches*G.nSingleMatches);
        G.nAdjustedMatches = (1-x).*(y);
        % Plot the ROC-type points
         figure(1234), plot(x,y,'o'),
         text(x,y,num2str(G.nAdjustedMatches))
         axis([0 1 0 1]), grid on
         xlabel('percentage of extra features'); ylabel('percentage of single matches')
        %}
        
        tempT = table((1:length(G.nSingleMatches))',G.nSingleMatches, G.nMultMatchClusters, G.nExtraMatchesInMultMatchClusters,G.nAdjustedMatches,...
            'VariableNames',{'idx','nSingleMatches','nMultMatchClusters','nExtraMatchesInMultMatchClusters','nAdjustedMatches'});
        sortrows(tempT,'nAdjustedMatches','descend')
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Divide the adjusted matches by the RTdist x MZdist area
    % Although calculated, it is not used at the moment! Seems to overpenalise multMatches
    %{
    standardizedRangeRT = 1/range(RTdist)*popState(:,3);
    standardizedRangeMZ = 1/range(MZdist)*popState(:,4);
    G.standardizedRTMZarea = standardizedRangeRT .* standardizedRangeMZ;
    G.nAdjustedMatches_perArea = 1./G.standardizedRTMZarea .* G.nAdjustedMatches ;
    %}
    
    %% Table with all info on the whole population results
    
    G=struct2table(G);

    %% Plot all individuals in the population!
    if plotOrNot == 2
        figure(fig1000); clf
        %figure(1000); clf
        set(gcf,'Name',['Generation number: ',num2str(genNr)],'visible',true); movegui(gcf,'center')
        [DimRow,DimCol] = M2S_subplotDim(populationSize);
        for c3=1:populationSize
            subplot(populationSize/2,4,(c3-1)*2+1), plot(refSet(:,1),RTdist,'.k')
            hold on, plot(refSet(rtmz_hits_idx{c3,1},1),RTdist(rtmz_hits_idx{c3,1}),'.r'), grid on, axis tight
            M2S_plotaxes('b-',[NaN,popState(c3,1)+popState(c3,3)]);M2S_plotaxes('b-',[NaN,popState(c3,1)-popState(c3,3)])
            title([num2str(c3),'RT:  singleMatches=',num2str(G.nSingleMatches(c3,1)), '  multMatches=',num2str(G.nMultMatchClusters(c3,1))],'fontsize',8)
            subplot(populationSize/2,4,c3*2), plot(refSet(:,2),MZdist,'.k')
            hold on, plot(refSet(rtmz_hits_idx{c3,1},2),MZdist(rtmz_hits_idx{c3,1}),'.r'), grid on, axis tight
            M2S_plotaxes('b-',[NaN,popState(c3,2)+popState(c3,4)]);M2S_plotaxes('b-',[NaN,popState(c3,2)-popState(c3,4)])
            title([num2str(c3),'MZ:  extraFeaturesSSQ=',num2str(G.nExtraFeaturesInMultMatchClusters_SSQ(c3,1)), '  adjMatches=',num2str(G.nAdjustedMatches(c3,1)), '  adjMatchesPerArea=',num2str(round(G.nAdjustedMatches_perArea(c3,1)))],'fontsize',8);
        end
        drawnow
    end
    
    %% Choose the next parents
    
    % Best two subjects are the ones with highest nAdjustedMatches.
    % Alternatively, the 2nd best could be chosen by nAdjustedMatches_perArea
    %[sortedBest , sortedBest_idx] = sort(G.nAdjustedMatches_perArea,'descend');
    
    % Sort the nAdjustedMatches to find the best ones
    [sortedBest , sortedBest_idx] = sort(G.nAdjustedMatches,'descend');
    bestIdx_inGen = [bestIdx_inGen;sortedBest_idx(1)];
    if genNr>1 
        if sortedBest(1) > genAlg_Res.nAdjustedMatches(genNr-1) %&& isequal(sortedBest_idx(1:2) , [1;2]) 
            bestOne = sortedBest_idx(1);% change for the new best
            bestOneCount=1;% restart counting consecutive wins
            % delete entries equal to the best of this generation
            sameAs1_idx = find(sortedBest == sortedBest(1));
            sameAs1_idx(1) = [];
            sortedBest(sameAs1_idx) = [];
            sortedBest_idx(sameAs1_idx) = [];
            % but if all are equal, then the second is equal to the first
            if length(sameAs1_idx) == populationSize-1 
                sortedBest(2,1) = sortedBest(1,1);
                sortedBest_idx(2,1) = sortedBest_idx(1,1);
            end
            
        else % there was no increase in quality of the cost function
            bestOne=1;% If index 1
            bestOneCount = bestOneCount+1;% add another successive victory
        end        
                
%             % If the best now was also the best of previous generation:
%             if sortedBest_idx(1) == 1
%                 bestOne=1;% If index 1
%                 bestOneCount = bestOneCount+1;% add another successive victory
%             else % If the best is different from the best of previous generation
%                 bestOne = sortedBest_idx(1);% change for the new best
%                 bestOneCount=1;% restart counting consecutive wins
%             end
%             % delete entries equal to the best of this generation
%             sameAs1_idx = find(sortedBest == sortedBest(1));
%             sameAs1_idx(1) = [];
%             sortedBest(sameAs1_idx) = [];
%             sortedBest_idx(sameAs1_idx) = [];
%             % but if all are equal, then the second is equal to the first
%             if length(sameAs1_idx) == populationSize-1 
%                 sortedBest(2,1) = sortedBest(1,1);
%                 sortedBest_idx(2,1) = sortedBest_idx(1,1);
%             end
    
    else % if it is the first generation
        bestOne = sortedBest_idx(1);
        bestOneCount=1;
    end
    G_best = G(sortedBest_idx(1:2),:);
       
    
    %% define new population
    newParents = popState(sortedBest_idx(1:2),:);
    newParents_table = array2table(newParents,'VariableNames',{'RTintercept','MZintercept','RTheight','MZheight','RTslope','MZslope'});
    
    % Plot the best one (the best parent)!
    if plotOrNot == 1
        figure(figure1001); clf
        set(gcf,'Name',['Best: Putative total number of matches = ',num2str(round(G_best.nTotalClusters(1)))]); %movegui(gcf,'center')
            p=1;
            p_idx=sortedBest_idx(p);            
            RTcoeffs = [popState(p_idx,1);popState(p_idx,5)];
            RTdist_pred = [ones(size(refSet,1),1),refSet(:,1)] * RTcoeffs;
            MZcoeffs = [popState(p_idx,2);popState(p_idx,6)];
            MZdist_pred = [ones(size(refSet,1),1),refSet(:,2)] * MZcoeffs;
            
            % RT
            subplot(1,2,(p-1)*2+1), 
            plot(refSet(:,1),RTdist,'.k'), title(['RT  centre: ',num2str(newParents(p,1)),'  height: ',num2str(newParents(p,3))])
            hold on, 
            plot(refSet(rtmz_hits_idx{p_idx,1},1),RTdist(rtmz_hits_idx{p_idx,1}),'.r'), axis tight, grid on
            plot(refSet(:,1),RTdist_pred+popState(p_idx,3),'.r')
            plot(refSet(:,1),RTdist_pred-popState(p_idx,3),'.r')
            xlabel('RT'); ylabel('RTdist'), grid on, axis tight; ylim([min(RTdist),max(RTdist)])
            % MZ
            subplot(1,2,p*2), plot(refSet(:,2),MZdist,'.k'), title(['MZ centre: ',num2str(newParents(p,2)),'  height: ',num2str(newParents(p,4))])
            hold on, plot(refSet(rtmz_hits_idx{p_idx,1},2),MZdist(rtmz_hits_idx{p_idx,1}),'.r'), axis tight, grid on
            plot(refSet(:,2),MZdist_pred+popState(p_idx,4),'.r')
            plot(refSet(:,2),MZdist_pred-popState(p_idx,4),'.r')
            %M2S_plotaxes('-k',[NaN,0]); M2S_plotaxes('r-',[NaN,popState(p_idx,2)+popState(p_idx,4)]);M2S_plotaxes('r-',[NaN,popState(p_idx,2)-popState(p_idx,4)])
            xlabel('MZ'); ylabel('MZdist'), grid on, axis tight; ylim([min(MZdist),max(MZdist)]) 
        drawnow
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% TO CREATE DIAGONAL LINES AFTER FINDING HORIZONTAL ONES:
    %% THIS IS TO BE DONE ONLY ONCE!!
    if (bestOneCount == opt.max_bestOneCount || genNr == opt.max_genNr) && flag_slopeCalculated == 0  
        if useSlope == 1
            flag_slopeCalculated = 1;
            bestOneCount = 0;
            %genNr = 1;
            for a=1:2
                idx = rtmz_hits_idx{sortedBest_idx(a),1};
                RTmdlr = fitlm(refSet(idx,1),RTdist(idx),'RobustOpts','on');
                if RTmdlr.Coefficients.pValue(2) < 0.01
                    disp('Define linear regression in RT')
                    newParents(a,1) = RTmdlr.Coefficients.Estimate(1);
                    newParents(a,5) = RTmdlr.Coefficients.Estimate(2);
                else
                    disp('DID NOT define linear regression in RT') 
                end
                MZmdlr = fitlm(refSet(idx,2),MZdist(idx),'RobustOpts','on');
                if MZmdlr.Coefficients.pValue(2) < 0.01
                    disp('Define linear regression in MZ')
                    newParents(a,2) = MZmdlr.Coefficients.Estimate(1);
                    newParents(a,6) = MZmdlr.Coefficients.Estimate(2);
                else
                    disp('DID NOT define linear regression in MZ') 
                end
            end
        
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Create descendents
    
    %% descendents by crossover
    % These are subjects formed only by mixing values from the two parents.
    % 1.define the number of crossover descendents
%     opt.nCrossover_descendents = 4;
    % 2.Create a matrix equal to the first parent to serve as base
    descendentsCrossover = repmat(newParents(1,:),opt.nCrossover_descendents,1);
    % 3. Create a matrix equal to the second parent
    secondParentMatrix = repmat(newParents(2,:),opt.nCrossover_descendents,1);
    % 4. Create a mask filter with 50% change for any of the parents
    crossover_mask = rand(opt.nCrossover_descendents,size(descendentsCrossover,2))<0.5;
    % 5. Mix both parents using the random mask 
    descendentsCrossover(crossover_mask) = secondParentMatrix(crossover_mask);
    
    %% descendents by crossover and then extra change of height
    % 'RTintercept','MZintercept','RTheight','MZheight','RTslope','MZslope'
    %1. These descendents are started as equal to the crossover ones
    descendentsCrossoverExtra = descendentsCrossover;
    % 2. But then 
    crossoverExtra_mask = rand(opt.nCrossover_descendents,2)<0.5;
    crossoverExtra_maskValsToAdd = 2* def_opt.n_initial_increments * (rand(opt.nCrossover_descendents,2)-0.5).* [repmat(RTdist_increment,opt.nCrossover_descendents,1) , repmat(MZdist_increment,opt.nCrossover_descendents,1)];
    descendentsCrossoverExtra(:,3:4) = descendentsCrossoverExtra(:,3:4)+crossoverExtra_mask.*crossoverExtra_maskValsToAdd;
    
    %% descendents by mutation
    % These are subjects formed first by crossover as above and then adding
    % totally random variation
    % 1.define the number of mutant descendents
%     opt.nMutation_descendents = 4;
    % 2.Create a matrix equal to the first parent to serve as base
    MutantCrossover = repmat(newParents(1,:),opt.nMutation_descendents,1);
    % 3. Create a matrix equal to the second parent
    secondParentMatrixM = repmat(newParents(2,:),opt.nMutation_descendents,1);
    % 4. Create a mask filter with 50% change for any of the parents
    Mutant_mask1 = rand(opt.nMutation_descendents,size(MutantCrossover,2))<0.5;
    % 5. Mix both parents using the random mask 
    MutantCrossover(Mutant_mask1) = secondParentMatrixM(Mutant_mask1);
 
    % xState = RTdist_seed, MZdist_seed,RTdist_thresh, MZdist_thresh
    % 6. Define randomly for each mutant subject the seed for its centre in RTdist and MZdist 
    Mutant_initState_RTseeds_idx = floor(opt.nSeedsRT*rand(opt.nMutation_descendents,1))+1;
    Mutant_initState_MZseeds_idx = floor(opt.nSeedsMZ*rand(opt.nMutation_descendents,1))+1;
    % 7. Define a random value to define the areas of each mutant subject 
    Mutant_randomRTdist_multFactor = ceil(opt.numberIncrements*rand(opt.nMutation_descendents,1));
    Mutant_randomMZdist_multFactor = ceil(opt.numberIncrements*rand(opt.nMutation_descendents,1));
%     Mutant_random_popState = [RTdist_seeds(Mutant_initState_RTseeds_idx), MZdist_seeds(Mutant_initState_MZseeds_idx),...
%         RTdist_increment*ceil(numberIncrements*rand(opt.nMutation_descendents,1)),  MZdist_increment*ceil(numberIncrements*rand(opt.nMutation_descendents,1))];
% NOTE: below the idx 5 and 6 are always the slope of the best parent
    Mutant_random_popState = [RTdist_seeds(Mutant_initState_RTseeds_idx), MZdist_seeds(Mutant_initState_MZseeds_idx),...
        RTdist_increment*Mutant_randomRTdist_multFactor,  MZdist_increment*Mutant_randomMZdist_multFactor,...
        repmat(newParents(1,5),length(Mutant_initState_RTseeds_idx),1),repmat(newParents(1,6),length(Mutant_initState_RTseeds_idx),1)];
    % 8. Create a mask filter with 50% probability of mutation at each position
    
    Mutant_mask = rand(opt.nMutation_descendents,size(MutantCrossover,2))<opt.mutationProb;
    
    % 9. The descendents are mutants from their parents base genome mix:
    descendentsMutation = MutantCrossover;
    % 9. The descendents are totally random mutants
    % descendentsMutation = Mutant_random_popState;
    % 10. Apply the mutations to the crossover matrix using the mask
    descendentsMutation(Mutant_mask) = Mutant_random_popState(Mutant_mask);
    
    %% The new population is:
    popState = [newParents;descendentsCrossoverExtra;descendentsMutation];
    
    %% Collect the output
    % Get it from: G_best, newParents_table  
    tempTable = [array2table([genNr,bestOneCount],'VariableNames',{'GenerationNr','nSuccessiveBest'}),newParents_table(1,:),G_best(1,:)];
    genAlg_Res = [genAlg_Res;tempTable];
   
    fprintf(' Generation:%d RepeatedBest: %d nTotal:%d  nSingle:%d  nMultiple:%d  nAdjMatches:%d\n', genNr,bestOneCount,G_best.nSingleMatches(1)+G_best.nMultMatchClusters(1),G_best.nSingleMatches(1),G_best.nMultMatchClusters(1),G_best.nAdjustedMatches(1))
    genNr = genNr+1;
end

% Adjust the generation count by deleting 1
genNr = genNr-1;

%% Collect the output
% Get it from: G_best, newParents_table  

% Setting for the matching with the just defined thresholds:
optBest.FIadjustMethod = 'median'; % {'median','regression'}
optBest.multThresh.RT_intercept = [genAlg_Res.RTintercept(end) - genAlg_Res.RTheight(end),genAlg_Res.RTintercept(end)+ genAlg_Res.RTheight(end)];
optBest.multThresh.RT_slope = [genAlg_Res.RTslope(end), genAlg_Res.RTslope(end)];
optBest.multThresh.MZ_intercept = [genAlg_Res.MZintercept(end)- genAlg_Res.MZheight(end),genAlg_Res.MZintercept(end)+ genAlg_Res.MZheight(end)];
optBest.multThresh.MZ_slope = [genAlg_Res.MZslope(end), genAlg_Res.MZslope(end)];
optBest.multThresh.log10FI_intercept = [-1000 1000];
optBest.multThresh.log10FI_slope = [0 0];      


%disp(bestIdx_inGen)
            
      