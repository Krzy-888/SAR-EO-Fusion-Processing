clc;clear;close all;
warning('off')

addpath sar-optical   % type of multi-modal data
% Define parameters 
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

scales = ["10","1","035"];
parts = [100,1000,2857];
i = 1;
for s = scales
    % IMREAD
    name_EO = "EO_UIAA_SUB_"+ s +"m_gray.png";
    name_SAR = "SAR_UIAA_SUB_"+ s +"m_gray.png";
    
    im1_path = '.\UIAA_Multiscale\' + name_EO
    im2_path = '.\UIAA_Multiscale\' + name_SAR
    image_1=imread(im1_path);
    image_1_ = double(image_1(1:parts(i),1:parts(i),:));
    image_2=imread(im2_path); 
    image_2_ = double(image_2(1:parts(i),1:parts(i),:));

    image_1_=imadjust(im2double(image_1_));
    image_2_=imadjust(im2double(image_2_));
    display('Adjusted')
    image_11=image_1_+0.001;%prevent denominator to be zero  
    image_22=image_2_+0.001;
    display('prevent denominator done')


    [r1,c1]=size(image_11);
    [r2,c2]=size(image_22);

    % Create HARRIS function
    [sar_harris_function_1,gradient_1,angle_1]=build_scale_opt(image_11,sigma,Mmax,ratio,d);
    [sar_harris_function_2,gradient_2,angle_2]=build_scale_sar(image_22,sigma,Mmax,ratio,d);

    % Feature point detection
    [GR_key_array_1]=find_scale_extreme(sar_harris_function_1,d_SH_1,sigma,ratio,gradient_1,angle_1);
    [GR_key_array_2]=find_scale_extreme(sar_harris_function_2,d_SH_2,sigma,ratio,gradient_2,angle_2);

    kp1res = sort(GR_key_array_1(:,6),'descend');
    kp2res = sort(GR_key_array_2(:,6),'descend');
    GR_key_array_11=GR_key_array_1(GR_key_array_1(:,6)>kp1res,:);
    GR_key_array_22=GR_key_array_2(GR_key_array_2(:,6)>kp2res,:);
    if is_keypoints_refine == true
        [ GR_key_array_1 ] = pointrefine(image_1_,GR_key_array_1,Mmax,sigma);
        [ GR_key_array_2 ] = pointrefine(image_2_,GR_key_array_2,Mmax,sigma);
    end

    name_eo = typ + "_UIAA_SUB_"+ s +"m_gray.csv"
    name_sar = typ + "_UIAA_SUB_"+ s +"m_gray.csv"
    writematrix(GR_key_array_1,'.\UIAA_Multiscale\'+name_eo);
    writematrix(GR_key_array_2,'.\UIAA_Multiscale\'+name_sar);
    i = i + 1; 
end
disp('Done')
