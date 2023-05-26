%% Function to plot lines on x=0 and y=0 to make it more visible
% Uses:
% M2S_plotaxes() - plots at [0,0]
% M2S_plotaxes('r') - plots red line at [0,0]
% M2S_plotaxes(':r') - plots red dotted line at [0,0]
% M2S_plotaxes('k',[1,3]) - plots black line at [1,3]
% M2S_plotaxes('-k',[50,NaN]) - plots black vertical line at x=50 only
% M2S_plotaxes('-k',[NaN,50]) - - plots black horizontal line at y=50 only
% NOTE: xy_vals contains the values that a vertical line should cross X and 
% a horizontal line should cross Y

function M2S_plotaxes(line_color,xy_vals)

x_lim = xlim;
y_lim = ylim;

gca;
hold on

if nargin == 0
    line_color = 'k';
    x_vals = [0,0]';
    y_vals = [0,0]';
    xy_vals = [0,0]
elseif nargin == 1
    x_vals = [0,0]';
    y_vals = [0,0]';
    xy_vals = [0,0]
elseif nargin == 2
    x_vals = [xy_vals(1),xy_vals(1)]';
    y_vals = [xy_vals(2),xy_vals(2)]';
    
    % if the xy_vals given are lower or higher than the current plot limits
    if xy_vals(1) < x_lim(1)
        x_lim(1) = xy_vals(1);
    end
    if xy_vals(1) > x_lim(2)
        x_lim(2) = xy_vals(1);
    end
    if xy_vals(2) < y_lim(1)
        y_lim(1) = xy_vals(2);
    end
    if xy_vals(2) > y_lim(2)
        y_lim(2) = xy_vals(2);
    end
    
elseif nargin > 2
    disp('Too many arguments for plotaxes function')

end

if nargin <= 2
    if prod(ylim) <= 0
    plot(x_vals,y_lim',line_color)
    end

    if prod(xlim) <=0
    plot(x_lim',y_vals,line_color)
    end
    
    if isnan(xy_vals(1))
        plot(xlim',y_vals,line_color)
    end
    if isnan(xy_vals(2))
        plot(x_vals,ylim',line_color)
    end
    
end
%axis tight