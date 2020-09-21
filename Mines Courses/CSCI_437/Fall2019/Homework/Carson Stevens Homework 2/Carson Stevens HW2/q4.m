% Author: Carson Stevens
% CSCI 437 HW2.4

% Read in video
movieObj = VideoReader('fiveCCC.avi'); % open file

% Create video writer to store result
video = VideoWriter('fiveCCCoutput.avi'); %create the video object
open(video)

% Same video stats
get(movieObj) % display all info
nFrames =movieObj.NumberOfFrames;
width =movieObj.Width; % get image width
height =movieObj.Height; % get image height

% Declare threshold for distance between centroids of blobs
threshold = 1

% Iterate through frames of video
for i=1:nFrames
    RGB = read(movieObj,i); % get one RGB image
    I = rgb2gray(RGB); %Convert to grayscale
    I2 = im2bw(I, 0.7); %Covert to BW
    
    % Image Process black blobs
    S = strel('disk', 3); %Apply opening
    bw = imopen(I2, S);
    S = strel('disk', 1); %Apply eroding
    bw = imerode(I2, S);
    bw = imcomplement(bw); % Take compliment to distinguish from other centroid
    
    % Image Process white blobs
    ww = im2bw(I, 0.6);
    bw = bwlabel(bw,8);
    ww = bwlabel(ww,8);
    
    % Define regions
    black_blobs = regionprops(bw);
    white_blobs = regionprops(ww);
    
    % Iterate through all blob combos and see if centroids are less than
    % threshold distance
    for black_index=1: size(black_blobs, 1)
        for white_index=1: size(white_blobs,1)
            
            % Get Centroids from current blobs
            bc = black_blobs(black_index).Centroid;
            wc = white_blobs(white_index).Centroid;
            
            % Get distance between centroids
            dist_matrix = [bc(1), bc(2); wc(1), wc(2)];
            dist = pdist(dist_matrix,'euclidean');
            
            if dist <= threshold
                % Draw rectangle over centroid match
                for k=4:8
                    
                    
            end
        end
    end 
    
    % Insert Frame number to image
    RGB = insertText(RGB,[width-110, 0],"Frame: "+num2str(i),'FontSize',18,'BoxColor', 'black','BoxOpacity',0.4,'TextColor','white');
    
    % Write result frame to output
    writeVideo(video, RGB)
end

% Close output video
close(video)
