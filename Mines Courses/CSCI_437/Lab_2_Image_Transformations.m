% Consider a rotation about the X axis of 1.1 radians, followed by a 
% rotation about the Y axis of -0.5 radians, followed by a rotation about 
% the Z axis by 0.1 radians (this order of rotations is called the “XYZ 
% fixed angles” convention).

% 1.1 Give the 3x3 rotation matrix corresponding to the rotations above.
Rz = [cos(0.1) -sin(0.1) 0;
      sin(0.1) cos(0.1) 0;
      0 0 1]
Ry = [1 0 0;
      0 cos(-0.5) -sin(-0.5);
      0 sin(-0.5) cos(-0.5)]
Rx = [cos(1.1) 0 sin(1.1) ;
      0 1 0;
      -sin(1.1) 0 cos(1.1)]

R = Rz * Ry * Rx

% 1.2 Since the rotation matrix is “orthonormal” matrix (i.e., a square matrix 
% whose rows and columns are orthogonal unit vectors), its inverse is equal 
% to its transpose. Show this.
R'
inv(R)
if isequal(R',inv(R))
  disp("Because orthonormal, inverse = transpose")
end

% 1.3
R_opposite = Rx * Ry * Rz

if not(isequal(R_opposite, R))
    disp("They aren't equal")
end

% 2.1 Camera to World
p1 = [6.8158 -35.1954 43.0640 1]'
p2 = [7.8493 -36.1723 43.7815 1]'
p3 = [9.9579 -25.2799 40.1151 1]'
p4 = [8.8219 -38.3767 46.6153 1]'
p5 = [9.5890 -28.8402 42.2858 1]'
p6 = [10.8082 -48.8146 56.1475 1]'
p7 = [13.2690 -58.0988 59.1422 1]'
translation = [10;-25;40]
H = [R translation;
     0 0 0 1]

% 2.2 World to Camera
Hwc = inv(H)

% 2.3 Intrinsic Camera
K = [400 0 256/2; 0 400 170/2; 0 0 1]

% 2.4
width = 256
height = 170
blank_image = zeros(height, width)
imshow(blank_image)
P = [p1,p2,p3,p4,p5,p6,p7]
P_camera = zeros(3,7)

K*(Hwc(1:3))
