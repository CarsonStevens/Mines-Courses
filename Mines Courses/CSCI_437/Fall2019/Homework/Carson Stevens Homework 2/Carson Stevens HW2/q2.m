% Author: Carson Stevens
% CSCI 437 HW2.2

% Reading in text sample
text = imread("textsample.tif");

% Specify and crop template
quadrant_width = 8;
quadrant_height = 25;
a = imcrop(text, [550, 754, quadrant_width, quadrant_height]);

% Threshold for a correlation of template
threshold = 0.81;

% Apply regionalmax to img and then compute correlation img/matrix
C = imregionalmax(text, 8);
C = normxcorr2(a, C);

% Iterate over correlation and if the correlation of the template is above
% threshold, draw rectangle
count=[0,0];
for row=1:size(C, 2)
    for col=1:size(C, 1)
        
        if C(col,row) >= threshold
            
            % append row and col to count
            count = [count; row col];
            
            % Draw rectangle over a template match
            text = insertShape(text, 'Rectangle', [row-quadrant_width-2, col-quadrant_height-2, quadrant_width+2, quadrant_height+4], 'LineWidth', 3, 'Color', 'r');
            
            % Line to try to skip multiple matches on template and same a
            row = row + quadrant_width + 6;
        end
        
    end
end

% Show results count shows x,y coords [(0,0) initializer subtracted from printed total]
imshow(text)
sprintf("There are %d a's", size(count,1)-1)
count
