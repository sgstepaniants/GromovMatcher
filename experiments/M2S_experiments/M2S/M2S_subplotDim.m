function [DimRow,DimCol] = M2S_subplotDim(n)
% This function finds the best subplot settings (rows vs cols) for a number
% of plots indicated by n.
% The possible plots are NxN  N-1xN or N-2vN, which are the ones more
% advised for a rectangular screen
% [DimRow,DimCol] = subplotDim(n)

% evaluate number of blank plots 

% Rui Pinto 2010

sqrt_n = ceil(sqrt(n));

NN = sqrt_n^2 - n;
Nminus1_N = (sqrt_n-1)*(sqrt_n) - n;
Nminus2_N = (sqrt_n-2)*(sqrt_n) - n;
Nminus1_Nplus1 = (sqrt_n-1)*(sqrt_n+1) - n;
Nminus1_Nplus2 = (sqrt_n-1)*(sqrt_n+2) - n;
Nminus2_Nplus1 = (sqrt_n-2)*(sqrt_n+1) - n;
Nminus2_Nplus2 = (sqrt_n-2)*(sqrt_n+2) - n;


res = [NN,Nminus1_N,Nminus2_N,Nminus1_Nplus1,Nminus1_Nplus2,Nminus2_Nplus1,Nminus2_Nplus2];
res(res<0) = NaN;

[~,minIdx] = nanmin(res);

if minIdx == 1
    Dim = [sqrt_n,sqrt_n];
elseif minIdx == 2
    Dim = [sqrt_n-1,sqrt_n];
    elseif minIdx == 3
    Dim = [sqrt_n-2,sqrt_n];
    elseif minIdx == 4
    Dim = [(sqrt_n-1),(sqrt_n+1)];
    elseif minIdx == 5
    Dim = [(sqrt_n-1),(sqrt_n+2)];
    elseif minIdx == 6
    Dim = [(sqrt_n-2),(sqrt_n+1)];
    elseif minIdx == 7
    Dim = [(sqrt_n-2),(sqrt_n+2)];
end

DimRow = Dim(1);
DimCol = Dim(2);


