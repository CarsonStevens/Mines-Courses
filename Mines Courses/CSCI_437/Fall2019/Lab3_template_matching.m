VideoReader.getFileFormats()
VideoWriter.getProfiles()
movieObj = VideoReader('building.avi'); % open file
get(movieObj) % display all info
nFrames =movieObj.NumberOfFrames;
width =movieObj.Width; % get image width
height =movieObj.Height; % get image height

%{
for i=1:nFrames
     RGB = read(movieObj,i); % get one RGB image
     I = rgb2gray(RGB); %Convert to grayscale
     imshow(I,[]);
     drawnow; % Force drawing of image now
end
%}
RGB = read(movieObj,1); % get one RGB image
I = rgb2gray(RGB); %Convert to grayscale
imshow(I,[]);
drawnow; % Force drawing of image now

[x,y] = ginput(1); % 1 = one click
x = round(x)
y = round(y)
quadrant_side = 9;
rectangle('Position', [x-9,y-9,19,19], 'EdgeColor', 'r')
temp_img = I(y-quadrant_side:y+quadrant_side, x-quadrant_side:x+quadrant_side);
imshow(temp_img)

vidObj = VideoWriter("mymovie.avi"); %create avi file
open(vidObj);

for i=2:nFrames
    RGB = read(movieObj,i); % get one RGB image
    I = rgb2gray(RGB); %Convert to grayscale
    imshow(I,[])
    C = normxcorr2(temp_img, I);
    max_img_val = max(C(:));
    [x,y] = find(C==max_img_val);
    x = x - quadrant_side
    y = y - quadrant_side
    
    rectangle('Position', [y-quadrant_side x-quadrant_side 19 19], 'EdgeColor', 'r')
    drawnow; % Force drawing of image now
    newFrameOut = getframe;
    writeVideo(vidObj,newFrameOut);
end
close(vidObj); % all done, close file

