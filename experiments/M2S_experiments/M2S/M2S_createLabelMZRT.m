function [MZRTstring, repeatedStrings] = M2S_createLabelMZRT(stringForLabel,varargin)
%% M2S_createLabelMZRT
% create a label for each entry, e.g., {'SLPOS_1250.1234_5.1234'} 
% (MS data m/z and RT in minutes) 
%
% [MZRT_str] = M2S_createLabelMZRT('LipidPosMode',m_z,RetentionTime)
%
% INPUT:
% MS DATA: 
% Input two separate columns (m/z values and retention time)
% Given a label, and the values for the two columns:
% [MZRT_str] = M2S_createLabelMZRT('LipidPositiveMode',m_z,RetentionTime)
% NOTE: THE SECOND STRING ARGUMENT IS DIVIDED BY 60 IN CASE IT IS > 150,
% to transform seconds to minutes.
% NOTE: the numerical data is truncated (nor rounded) at the 4th digit
% after the decimal point.
% NOTE: The function works also for single column of values, eg in NMR ppm.
% NMR DATA: 
% 1. Given a label, and the NMR ppm values
% [X.VarInfo.MZRT_str] = M2S_createLabelMZRT('CPMG',X.VarInfo.ppm)
% 
% NOTE: for whole-string labels, use the following code instead of this function:
% X.VarInfo.MZRT_str = strcat("BILISA","_",stringArray)
%
% OUTPUT:
% MZRT_str: all labels as a cell
%
% M2S toolbox to match LCMS metabolomics features of untargeted datasets.
% *** Rui Climaco Pinto ***
% *** Imperial College London, 2021 ***

% If there is only one argument, create only one string
arg1 = varargin{1};
arg1_str=cell(length(arg1),1); 
for a=1:length(arg1)
    
    if isnan(arg1(a))
        arg1_str{a,1} = 'NaN';
    else
        temp_arg1=num2str(floor(10000*arg1(a)));
        if arg1(a)<0.1
            temp_arg1=num2str(floor(100000*arg1(a)));
            arg1_str{a,1} = ['0.0',temp_arg1(end-3:end-1)];
        elseif arg1(a)<1
            arg1_str{a,1} = ['0.',temp_arg1(end-3:end)];
        else
            arg1_str{a,1} = [temp_arg1(1:end-4),'.',temp_arg1(end-3:end)];
        end
    end
end


% if there are two arguments, create a second string    
if length(varargin) == 2   
    arg2 = varargin{2};
    
    %% force RT in minutes
    if nanmax(arg2)>150
        arg2=1/60*arg2;
    end
    
    arg2_str=cell(length(arg2),1); 
    for a=1:length(arg2)
        
        if isnan(arg2(a))
            arg2_str{a,1} = 'NaN';
        else
            temp_arg2=num2str(floor(10000*arg2(a)));
            if arg2(a)<0.0001
                arg2_str{a,1} = '0.0001';
            elseif arg2(a)<0.001      
                temp_arg2=num2str(floor(100000*arg2(a)));
                arg2_str{a,1} = ['0.000',temp_arg2(end:end-1)];
            elseif arg2(a)<0.01      
                temp_arg2=num2str(floor(100000*arg2(a)));
                arg2_str{a,1} = ['0.00',temp_arg2(end-2:end-1)];
            elseif arg2(a)<0.1
                temp_arg2=num2str(floor(100000*arg2(a)));
                arg2_str{a,1} = ['0.0',temp_arg2(end-3:end-1)];
            elseif arg2(a)<1
                arg2_str{a,1} = ['0.',temp_arg2(end-3:end)];
            else
                arg2_str{a,1} = [temp_arg2(1:end-4),'.',temp_arg2(end-3:end)];
            end
        end
    end
end
   
% Create the final label
MZRTstring=cell(length(arg1),1);  
if length(varargin) == 1
    for a=1:length(arg1)
        MZRTstring{a,1}=[stringForLabel,'_', arg1_str{a,1}]; 
    end
elseif length(varargin) == 2
    for a=1:length(arg1)
        MZRTstring{a,1}=[stringForLabel,'_', arg1_str{a,1},'_', arg2_str{a,1}]; 
    end
end

if length(unique(MZRTstring)) ~= length(MZRTstring)
    disp('** WARNING: there are some repeated strings in the output **')
    repeatedStrings = tabulate(MZRTstring);
    repeatedStrings = table(string(repeatedStrings(:,1)),str2double(string(repeatedStrings(:,2))),str2double(string(repeatedStrings(:,3))),'VariableNames',{'MZRT_str','Count','Percent'});
    repeatedStrings = sortrows(repeatedStrings,2,'descend');
    repeatedStrings(repeatedStrings.Count == 1,:)=[];
    %disp(repeatedStrings)
    disp('** WARNING: the above are repeated strings in the output **')
else
    disp('** GOOD NEWS: there are no repeated strings in the output **')
end
