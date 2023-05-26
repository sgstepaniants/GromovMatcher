function [refDataMatched,refVarInfoMatched,targetDataMatched,targetVarInfoMatched] = M2S_applyMatchingResults(filenameRef,filenameTarget,refFeatures_idx,targetFeatures_idx);


% Load data
refData = readmatrix(filenameRef,'Sheet','Data');
refVarInfo = readtable(filenameRef,'Sheet','VarInfo');

targetData = readmatrix(filenameTarget,'Sheet','Data');
targetVarInfo = readtable(filenameTarget,'Sheet','VarInfo');

% Select the matched data and features
refDataMatched = refData(:,refFeatures_idx);
refVarInfoMatched = refVarInfo(refFeatures_idx,:);

targetDataMatched = targetData(:,targetFeatures_idx);
targetVarInfoMatched = targetVarInfo(targetFeatures_idx,:);

% Save results
refIdx = strfind(filenameRef,'.');
writematrix(refDataMatched,[filenameRef(1:refIdx-1),'_matched.xlsx'],'Sheet','Data')
writetable(refVarInfoMatched,[filenameRef(1:refIdx-1),'_matched.xlsx'],'Sheet','VarInfo')

targetIdx = strfind(filenameTarget,'.');
writematrix(targetDataMatched,[filenameTarget(1:targetIdx-1),'_matched.xlsx'],'Sheet','Data')
writetable(targetVarInfoMatched,[filenameTarget(1:targetIdx-1),'_matched.xlsx'],'Sheet','VarInfo')