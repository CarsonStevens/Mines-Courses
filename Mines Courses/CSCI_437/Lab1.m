%Matrices
my_matrix=[1 2 3; 4 5 6; 7 8 9]
my_matrix=[1,2,3; 4,5,6; 7,8,9]
(my_matrix(1,2))
(1:10)
(my_matrix(3,:))
(my_matrix(:,3))

% Math and Variables
a=5
b=6
% displays variables and values
whos
% clears variables (locally?)
clear all
c=7
whos

% [1,2,3]'*[4,5,6]'
[1,2,3]'*[4,5,6]
[1,2,3]'.*[4,5,6]'

% 1. Create a vector A, consisting of the first 5 odd numbers
A=[1,3,5,7,9]
% 2. Create a vector B, consisting of the first 5 even numbers
B=[2,4,6,8,10]
% 3. Find the inner (dot) product of A and B
dot(A,B)
% 4. Find C, the outer product of A and B
C=A'.*B
% 4a. What is the size of C ?
size(C)
% 5. Compute the sum of all the elements of C
sum(sum(C))
% 6. Compute the trace of C
trace(C)
% 7. Compute the determinant of C
det(C)
% 8. Square all the elements of C
C.^2
% 9. Use M=magic(n) to create a magic square of size n, where n should be the same size as C.
M=magic(size(C))
% 10. Compute the matrix product of M and C
M*C
% 11. Compute the point-by-point product of M and C
M.*C
% 12. Compute the product of M and (column) vector A
M*A'
% 13. Use range notation to create a vector containing the numbers between 0 and 10, and store the results in variable X.
X = 0:10
% 14. Use plot(X,ln(X)) and make sure you understand what you are seeing.
plot(X,log(X))
% 15. For the following exercises, use the online MATLAB documentation to figure out how to...
% https://www.mathworks.com/help/index.html
% 16. Use “rand” to create a random matrix M
M = rand(3,3)
% 17. Take the inverse of M
inv(M)
% 18. Multiply M by 255, then convert it to uint8
M=uint8(M*255)
% 19. Clip values of M between 50 to 200
M(50>M & M<200)


% 1. You can use imread to read in an image:
[X,cmap]=imread('corn.tif')
% 2. 
imshow(X)
imshow(X,cmap)
imshow(X,[])
imshow(X,[]),figure
imtool(X)
imtool(X,cmap)
% 3.
disp(X)
%4. 
D=im2double(X)