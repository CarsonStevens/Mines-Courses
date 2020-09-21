cameraman_original = imread("cameraman.tif");
sigma = 3;
width = 6 * sigma;
Gx = fspecial('gaussian', [1,width], sigma);
Gy = fspecial('gaussian', [width,1], sigma);
G = fspecial('gaussian', width, sigma);
tic
cameraman_G = imfilter(cameraman_original, G);
timerG = toc
imshow(cameraman_G)
tic
cameraman_Gxy = imfilter(imfilter(cameraman_original, Gy), Gx);
timerGxy = toc
imshow(cameraman_Gxy)
cameraman_G(27,1)
cameraman_Gxy(27,1)
difference = 0;
for row=1:size(cameraman_G, 2)
    for col=1:size(cameraman_G, 1)
        if abs(cameraman_G(col,row)-cameraman_Gxy(col,row)) > difference
            difference = (cameraman_G(col,row)-cameraman_Gxy(col,row));
            row
            col
        end
    end
end
sprintf("The difference was %d", difference)
sprintf("G took %d time to compute", timerG)
sprintf("Gxy took %d time to compute", timerGxy)