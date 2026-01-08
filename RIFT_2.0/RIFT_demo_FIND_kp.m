clc;clear;close all;
warning('off')

addpath sar-optical   % type of multi-modal data

str1='.\Moje_dane\SAR_URRC_SUB_035m_bad.png';   % image pair
str2='.\Moje_dane\SAR_URRC_SUB_035m_gray.png';
im1 = im2uint8(imread(str1));
im2 = im2uint8(imread(str2));

if size(im1,3)==1
    temp=im1;
    im1(:,:,1)=temp;
    im1(:,:,2)=temp;
    im1(:,:,3)=temp;
end

if size(im2,3)==1
    temp=im2;
    im2(:,:,1)=temp;
    im2(:,:,2)=temp;
    im2(:,:,3)=temp;
end

disp('RIFT feature detection and description')
% RIFT feature detection and description
[des_m1,des_m2] = RIFT_no_rotation_invariance(im1(1000:2000,1000:2000,:),im2(1000:2000,1000:2000,:),4,6,96);
writematrix(des_m1.kps,"RIFT_SAR_URRC_03m_bad_n.csv")
writematrix(des_m2.kps,"RIFT_SAR_URRC_03m_gray_n.csv")
disp('Done')
