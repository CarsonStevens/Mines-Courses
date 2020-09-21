img = imread("test000.jpg")
imshow(img)
[x,y] = ginput(1); % 1 = one click
x = round(x)
y = round(y)
quadrant_width = 9
quadrant_height = 9
rectangle('Position', [x-9,y-9,19,19])
template = [[x+quadrant_width,x-quadrant_width],[y+quadrant_width,y-quadrant_width]]

temp_img = img(y-quadrant_height:y+quadrant_height, x-quadrant_width:x+quadrant_width)
imshow(temp_img)
img2 = imread("test012.jpg")
imshow(img2)
C = normxcorr2(temp_img, img2)
imshow(C)

max_img_val = max(C(:))
[x,y] = find(C==max_img_val)
x = x - quadrant_width
y = y - quadrant_height
fprintf('Correlation score = %f at (x,y) = (%d,%d)\n', max_img_val,x, y);
rectangle('Position', [x-quadrant_width y-quadrant_width 19 19])