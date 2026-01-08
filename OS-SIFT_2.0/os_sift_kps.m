clear;close all;clc

%  opticla to sar Sift

%% read image
addpath 'Moje_dane\'
% gray image is required
% Kolejność EO SAR Gray Bad Log
im1_path='EO_URRC_SUB_035m_gray.png';  %optical image
%im1_path = '18APR24174447-P2AS-013202438020_01_P001L1-1-L2.tiff';
%im2_path='SAR_URRC_SUB_035m_gray.png';  %sar image
%im2_path = 'GF3_KAS_SL_015331_W95.9_N41.3_20190709_L1B_HH_L10004105495-1-L2.tiff';
%im2_path='SAR_URRC_SUB_035m_bad.png';  %sar image
im2_path='SAR_URRC_SUB_035m_log.png';  %sar image
display('initiation started')
image_1=imread(im1_path);
image_1_ = double(image_1(1000:2000,1000:2000,:));
image_2=imread(im2_path); 
image_2_ = double(image_2(1000:2000,1000:2000,:));
%image_3 =imread(im3_path); 
%image_3_ = double(image_3(1000:2000,1000:2000,:));
%image_4 =imread(im4_path); 
%image_4_ = double(image_4(1000:2000,1000:2000,:));
display('images readed')
image_1_=imadjust(im2double(image_1_));
image_2_=imadjust(im2double(image_2_));
%image_2=imadjust(im2double(image_3_));
%image_2=imadjust(im2double(image_4_));
display('Adjusted')
image_11=image_1_+0.001;%prevent denominator to be zero  
image_22=image_2_+0.001;
%image_33=image_3_+0.001;%prevent denominator to be zero  
%image_44=image_4_+0.001;
display('prevent denominator done')
%% Define parameters 
sigma=2;%the parameter of first scale
ratio=2^(1/3);%scale ratio
Mmax=8;%layer number
d=0.04;
d_SH_1=0.00001;%Harris function threshold  
d_SH_2=0.00001;%Harris function threshold  
change_form='affine';%it can be 'similarity','afine','perspective'
is_sift_or_log='GLOH-like';%Type of descriptor,it can be 'GLOH-like','SIFT'
is_keypoints_refine=false;% set to false if the number of keypoints is small
is_multi_region=false; % set to false for efficiency

[r1,c1]=size(image_11);
[r2,c2]=size(image_22);
%[r3,c3]=size(image_33);
%[r4,c4]=size(image_44);
display('parameters defined')
%% Create HARRIS function
[sar_harris_function_1,gradient_1,angle_1]=build_scale_opt(image_11,sigma,Mmax,ratio,d);
[sar_harris_function_2,gradient_2,angle_2]=build_scale_sar(image_22,sigma,Mmax,ratio,d);
%[sar_harris_function_3,gradient_3,angle_3]=build_scale_sar(image_33,sigma,Mmax,ratio,d);
%[sar_harris_function_4,gradient_3,angle_3]=build_scale_sar(image_44,sigma,Mmax,ratio,d);

%% Feature point detection
[GR_key_array_1]=find_scale_extreme(sar_harris_function_1,d_SH_1,sigma,ratio,gradient_1,angle_1);
[GR_key_array_2]=find_scale_extreme(sar_harris_function_2,d_SH_2,sigma,ratio,gradient_2,angle_2);
%[GR_key_array_1]=find_scale_extreme(sar_harris_function_1,d_SH_1,sigma,ratio,gradient_1,angle_1);
%[GR_key_array_2]=find_scale_extreme(sar_harris_function_2,d_SH_2,sigma,ratio,gradient_2,angle_2);
% save the strongest 5000 points
kp1res = sort(GR_key_array_1(:,6),'descend');
kp2res = sort(GR_key_array_2(:,6),'descend');
GR_key_array_11=GR_key_array_1(GR_key_array_1(:,6)>kp1res,:);
GR_key_array_22=GR_key_array_2(GR_key_array_2(:,6)>kp2res,:);
if is_keypoints_refine == true
%     [ GR_key_array_1 ] = RemovebyBorder( GR_key_array_1, c1,r1, 11 );
%     [ GR_key_array_2 ] = RemovebyBorder( GR_key_array_2, c2,r2, 11 );
    [ GR_key_array_1 ] = pointrefine(image_1_,GR_key_array_1,Mmax,sigma);
    [ GR_key_array_2 ] = pointrefine(image_2_,GR_key_array_2,Mmax,sigma);
end

display('Feature Points Detected')
%display(GR_key_array_1)
%display(GR_key_array_2)
writematrix(GR_key_array_1,"RIFT_EO_URRC_03m_gray_n.csv")
writematrix(GR_key_array_2,"RIFT_SAR_URRC_03m_log_n.csv")



