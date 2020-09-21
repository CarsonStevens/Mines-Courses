% Author: Carson Stevens
% CSCI 437 HW1 Question 3

% focal length in px
f = 600;

% img size = 640px x 480px
img = zeros(480,640);
img_height = size(img, 2);
img_width = size(img, 1);

% cube coordinates in world:
cube_coords_world = [0 0 0 1;
                     1 0 0 1;
                     1 1 0 1;
                     0 1 0 1;
                     0 0 1 1;
                     1 0 1 1;
                     1 1 1 1;
                     0 1 1 1;]';

% camera position to mount translation matrix
t_c_m = [0;-1.5;0];

% mount position to vehicle translation matrix
t_m_v = [0;-3;0];

% translation of vehicle to world
t_v_w = [6;-8;1];

% 0 degree Rotation matrix for camera to mount
Rz_c_m = [cosd(0) -sind(0) 0; sind(0) cosd(0) 0; 0 0 1;]
Ry_c_m = [cosd(0) 0 sind(0); 0 1 0; -sind(0) 0 cosd(0);]
Rx_c_m = [1 0 0; 0 cosd(0) -sind(0); 0 sind(0) cosd(0);]

% Rotation matrix of camera to mount (identity matrix)
R_c_m = Rz_c_m*Ry_c_m*Rx_c_m

% Homogenous camera to mount
H_c_m = [R_c_m t_c_m; 0 0 0 1]

% Homogenous mount to camera
H_m_c = inv(H_c_m)

% -120 degree Rotation matrix around x axis for mount to vehicle
%Rz_m_v = [cosd(0) -sind(0) 0; sind(0) cosd(0) 0; 0 0 1;]
%Ry_m_v = [cosd(0) 0 sind(0); 0 1 0; -sind(0) 0 cosd(0);]
Rx_m_v = [1 0 0; 0 cosd(-120) -sind(-120); 0 sind(-120) cosd(-120);]

% Rotation matrix of camera to world
R_m_v = Rx_m_v

% Homogenous mount to vehicle
H_m_v = [R_m_v t_m_v; 0 0 0 1;]
H_v_m = inv(H_m_v)
  
% 30 degree Rotation matrix around z axis for vehicle to world
Rz_v_w = [cosd(30) -sind(30) 0; sind(30) cosd(30) 0; 0 0 1;]
Ry_v_w = [cosd(0) 0 sind(0); 0 1 0; -sind(0) 0 cosd(0);]
Rx_v_w = [1 0 0; 0 cosd(0) -sind(0); 0 sind(0) cosd(0);]

% Rotation matrix for vehicle to world
R_v_w = Rz_v_w * Ry_v_w * Rx_v_w

% Homogenous Vehicle to World
H_v_w = [R_v_w t_v_w; 0 0 0 1;]
% Homogenous World to Vehicle
H_w_v = inv(H_v_w)

% for 3D to 3D Transformation
% Homogeneous Camera to World
H_c_w = H_v_w * H_m_v * H_c_m
H_w_c = inv(H_c_w)

% K parameter
K = [f 0 (img_width/2); 0 f (img_height/2); 0 0 1]

%for 3D to 2D Transformation and Illistration
H_c_w2 = H_v_w * H_m_v * H_c_m
H_w_c2 = inv(H_c_w2)

Mext2 = H_w_c2(1:3,:);
p_wire = K * Mext2 * cube_coords_world

% Create Points for cube on image
for col=1:size(p_wire,2)
    p_wire(:,col) = p_wire(:,col) / p_wire(3,col);
    img(round(p_wire(2,col)), round(p_wire(1,col))) = 255;
end
imshow(img)
% wire part
[round(p_wire(2,:)); round(p_wire(1,:))]
line(p_wire(1,:),p_wire(2,:))

% for 3D to 3D Transformation and Illistration
% Mext parameter
Mext = H_w_c(1:3,:);

% World points to camera image
p_img = H_w_c * cube_coords_world(:,:)

%Plotting
verts = reshape(p_img(1:3,1:8)',[],3);
scatter3(p_img(1,:), p_img(2,:), p_img(3,:))
scatter3([0,0,0],[0,-1.5,-4.5],[0,0,0])
fac = [1 2 6 5;2 3 7 6;3 4 8 7;4 1 5 8;1 2 3 4;5 6 7 8];
patch('Vertices',verts,'Faces',fac,...
      'FaceVertexCData',hsv(6),'FaceColor','flat')
text(0,0,0,"Camera")
text(0,-1.5,0,"Mount")
text(0,-4.5,0,"Vehicle")
xlabel('x')
ylabel('y')
zlabel('z')
axis equal
disp("Projected Point Coordinates")
projected_coords = p_img(1:3,1:8)
