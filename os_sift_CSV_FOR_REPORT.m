
warning('off')
% data = ["URRC","UIAA","URWH","UDYE"];
data = ["UIAA"];
scales = ["10"];
norms = ["gray"];
grd = [10,1,0.35];

ilorazy = [grd(1)/0.35,grd(1)/0.35,grd(2)/0.35];

pkt_PNEO_list = [];
pkt_CAPELLA_list = [];


fid = fopen('report_OS_SIFT\EO_OS_SIFT_mach.csv','a');
% KONIECZNIE DO USTALENIA I WRZUCENIA POD for n = norms
for dateseses = data
    disp(dateseses);
    for s = scales
        for n = norms
            path_img_sar = "Norm\SAR_"+dateseses+"_SUB_"+s+"m_"+n+".png";
            rng(0,'twister')
            if s == "GM_035"
                path_img_eo = "Norm\EO_"+dateseses+"_SUB_"+"035"+"m_gray.png";
            else 
                path_img_eo = "Norm\EO_"+dateseses+"_SUB_"+s+"m_gray.png";
            end
            im1_path=path_img_eo;  %optical image
            im2_path=path_img_sar; 
            display('initiation started')
            image_1=imread(im1_path);
            image_2=imread(im2_path); 
            display('images readed')
            image_1=imadjust(im2double(image_1));
            image_2=imadjust(im2double(image_2));
            image_11=image_1+0.001;%prevent denominator to be zero  
            image_22=image_2+0.001;
            display('prevent denominator done')
            %% Define parameters 
            tic;
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
            %% Create HARRIS function
            [sar_harris_function_1,gradient_1,angle_1]=build_scale_opt(image_11,sigma,Mmax,ratio,d);
            [sar_harris_function_2,gradient_2,angle_2]=build_scale_sar(image_22,sigma,Mmax,ratio,d);
            [r1,c1]=size(image_11);
            [r2,c2]=size(image_22);
            %% Feature point detection
            [GR_key_array_1]=find_scale_extreme(sar_harris_function_1,d_SH_1,sigma,ratio,gradient_1,angle_1);
            [GR_key_array_2]=find_scale_extreme(sar_harris_function_2,d_SH_2,sigma,ratio,gradient_2,angle_2);
            kp1res = sort(GR_key_array_1(:,6),'descend');
            kp2res = sort(GR_key_array_2(:,6),'descend');
            GR_key_array_11=GR_key_array_1(GR_key_array_1(:,6)>kp1res,:);
            GR_key_array_22=GR_key_array_2(GR_key_array_2(:,6)>kp2res,:);
            if is_keypoints_refine == true
                [ GR_key_array_1 ] = pointrefine(image_1,GR_key_array_1,Mmax,sigma);
                [ GR_key_array_2 ] = pointrefine(image_2,GR_key_array_2,Mmax,sigma);
            end
            initial_OS_SIFT = toc;
            display('Feature Points Detected')
            %% Descriptor calculation
            tic;
            [descriptors_1,locs_1]=calc_descriptors_parallel(gradient_1,angle_1,GR_key_array_1);
            [descriptors_2,locs_2]=calc_descriptors_parallel(gradient_2,angle_2,GR_key_array_2);
            %% match & image fusion
            [solution,rmse,cor22,cor11]=CSC_1(image_2,image_1,descriptors_2,locs_2,descriptors_1,locs_1);
            Iinit_maching= toc;
            name_kp_mach_sar = "SAR_"+dateseses+"_SUB_"+s+"m_"+n+"_mach.csv";
            name_kp_mach_eo = "EO_"+dateseses+"_SUB_"+s+"m_"+n+"_mach.csv";
            name_kp_bfmach_sar = "SAR_"+dateseses+"_SUB_"+s+"m_"+n+"_before_mach.csv";
            writematrix(locs_2(:,1:2),'report_OS_SIFT\'+name_kp_bfmach_sar);
            writematrix(cor22,'report_OS_SIFT\'+name_kp_mach_sar);
            writematrix(cor11,'report_OS_SIFT\'+name_kp_mach_eo);
            total_time = initial_OS_SIFT + Iinit_maching;
            time_list = [total_time, RIFT_Iinit_process_point_detection_and_description, RIFT_Iinit_maching, Outlier_removal];
            name_mach_sar_time = "SAR_"+dateseses+"_SUB_"+s+"m_"+n+"_mach_time.csv";
            writematrix(time_list,'report_OS_SIFT\'+name_mach_sar_time);
        end
    end
end
fclose(fid);
disp('DONE')

