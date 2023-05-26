function [chi2_crit,chi2_prob,chi2_df,r_chi2] = M2S_gofTestChi2(x,df,nbins,plotOrNot)

% df = 1
%r = chi2rnd(df,1000,1); % for test

% binning of real values (A)
[A] = histcounts(x,nbins); %,'normalization','probability');

% Create and bin random values (B)
Btemp_all=zeros(1,nbins);
r_chi2 = chi2rnd(df,length(x),10);
for iterat = 1:10
[Btemp] = histcounts(r_chi2(:,iterat),nbins); %,'normalization','probability');
Btemp_all = Btemp_all+Btemp;
end
B = round((Btemp_all/10));
B(B==0) = 1;
%% COMPARISON OF TWO SETS OF VALUES A and B would be like this:
%{
Total = A+B;
Proportion_inA = A./Total;
TotalA = sum(A);
TotalB = sum(B);
TotalTotal = sum(Total);

table_AB = table(A,B,Total,Proportion_inA);

%% Calculation of the chi2 test on numbers from table_AB
ExpectedNumbers_A = Total * (TotalA/TotalTotal);
ExpectedNumbers_B = Total * (TotalB/TotalTotal); 
% Total_ExpectedNumbers_A = sum(ExpectedNumbers_A)

OminusE_A = A-ExpectedNumbers_A;
OminusE_B = B-ExpectedNumbers_B;

sq_OminusE_divE_A = (OminusE_A.^2)./ExpectedNumbers_A;
sq_OminusE_divE_B = (OminusE_B.^2)./ExpectedNumbers_B;

Total_sq_OminusE_divE_A = sum(sq_OminusE_divE_A);
Total_sq_OminusE_divE_B = sum(sq_OminusE_divE_B);

chi2_crit = Total_sq_OminusE_divE_A + Total_sq_OminusE_divE_B;

%}

%% COMPARISON OF A SET OF VALUES WITH CHI2

%% Distribution of samples in bins for the two sets (A real; B random chi2)
% https://www.bmj.com/about-bmj/resources-readers/publications/statistics-square-one/8-chi-squared-tests

OminusE = A-B;
sq_OminusE_divE = (OminusE.^2)./B;
chi2_crit = sum(sq_OminusE_divE);

chi2_df = nbins-1;
chi2_prob = chi2cdf(chi2_crit,chi2_df,'upper');
% chi2_prob = chi2cdf(chi2_crit,chi2_df);

% NOTE: If e.g. chi2_prob < 0.05 the two curves are different. The smaller
% the chi2_prob, the more different the two curves.

if plotOrNot == 1
    figure('Position',[13         487        1301         420])
    subplot(1,2,1),plot(A,'o-k')
    hold on, plot(B,'o-r'), grid on, axis tight
    title(['Comparison of values with chi2 with df = ',num2str(df)])
    legend({'observed','expected'},'Location','northeast')
    subplot(1,2,2), bar(A-B)
    title('Difference between observed and expected'), axis tight, grid on
end