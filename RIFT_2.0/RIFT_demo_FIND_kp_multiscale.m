clc;clear;close all;
warning('off')

addpath sar-optical   % type of multi-modal data

scales = ["10","1","035"];
parts = [100,1000,2857];
types = ["EO","SAR"];
for typ = types
    i = 1;
    for s = scales
        name = typ + "_UIAA_SUB_"+ s +"m_gray.png"
        im1 = im2uint8(imread('.\UIAA_Multiscale\' + name));

        if size(im1,3)==1
            temp=im1;
            im1(:,:,1)=temp;
            im1(:,:,2)=temp;
            im1(:,:,3)=temp;
        end

        disp('RIFT feature detection and description')
        [des_m1] = RIFT_no_rotation_invariance_For_One_IMG(im1(1:parts(i),1:parts(i),:),4,6,96);


        % RIFT feature detection and description
        name = typ + "_UIAA_SUB_"+ s +"m_gray.csv";

        writematrix(des_m1.kps,'.\UIAA_Multiscale\'+name);
        i = i + 1; 
    end
end
disp('Done')
