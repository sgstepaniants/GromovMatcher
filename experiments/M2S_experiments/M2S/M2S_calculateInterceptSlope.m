% [interceptSlope] = M2S_calculateInterceptSlope(startPoint,endPoint)
% This function is part of M2S toolbox to match metabolomics features across untargeted datasets.
% 
% 
% Function to calculate a slope between two points
%
% INPUT:
% Two ways to use:
% 1- By clicking on two points in a plot (no input):
% RTMZFI_slope = M2S_calculateInterceptSlope()
% 2- By inserting the values of the two points ([x1,y1],[x2,y2]) as below:
% RTMZFI_slope = M2S_calculateInterceptSlope([0,0.5],[12,-4.5]);
%
% OUTPUT:
% Row vector interceptSlope (= [intercept, slope])
%
%   Rui Climaco Pinto, 2021
%   Imperial College London

function [interceptSlope] = M2S_calculateInterceptSlope(startPoint,endPoint)

interceptSlope = [NaN,NaN];

if nargin == 0
    fprintf('\nFunction to calculate intercept and slope between two points in a plot.\n')
    fprintf('The last figure to be clicked is the currently active figure (gcf).\n')
    fprintf('Choose 2 points in the plot, using mouse left-click.\n')
    fprintf('To cancel click Ctrl+C\n')
    %f = msgbox('Click on the desired figure','Info','icon','help');
    fig = ancestor(gca,'figure');
    hold on
    counterX = 1;
    while counterX<=2
    % click on plot to choose an area, and find indices of points inside
        [xchosen(counterX,1),ychosen(counterX,1)] = ginput(1);
        if counterX == 1
            hold on, plot(xchosen(1),ychosen(1),'or') ;
        end
        counterX=counterX+1;
    end
    plot([xchosen(1);xchosen(2)],[ychosen(1);ychosen(2)],'-r') ;
    plot(xchosen(2),ychosen(2),'or')
    
    startPoint = [xchosen(1),ychosen(1)];
    endPoint = [xchosen(2),ychosen(2)];
end

interceptSlope(1,2) = (endPoint(2)-startPoint(2))./(endPoint(1)-startPoint(1));
interceptSlope(1,1) = startPoint(2) - startPoint(1) * interceptSlope(1,2);