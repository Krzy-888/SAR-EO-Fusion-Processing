warning('off')
% data = ["URRC","UIAA","URWH","UDYE"];
data = ["URRC","UIAA","URWH","UDYE"];
% scales = ["10","1","035","GM_035"];
scales = ["10","1","035"];
norms = ["gray","log","bad"];
grd = [10,1,0.35];

ilorazy = [grd(1)/0.35,grd(1)/0.35,grd(2)/0.35];

pkt_PNEO_list = [];
pkt_CAPELLA_list = [];


fid = fopen('report_RIFT\EO_RIFT_mach.csv','a');
% KONIECZNIE DO USTALENIA I WRZUCENIA POD for n = norms
for d = data
    disp(d);
    for s = scales
        for n = norms
            rng(0)
            path_img_sar = "SAR_"+d+"_SUB_"+s+"m_"+n+".png";
            if s == "GM_035"
                path_img_eo = "EO_"+d+"_SUB_"+"035"+"m_gray.png";
            else 
                path_img_eo = "EO_"+d+"_SUB_"+s+"m_gray.png";
            end
            % fprintf(fid, '%s\n', d);
            disp(path_img_sar);
            disp(path_img_eo);
            header = path_img_sar + "->" + path_img_eo;
            
            %Imread
            im1 = im2uint8(imread("Norm\"+path_img_sar));
            im2 = im2uint8(imread("Norm\"+path_img_eo));

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
            im1_ = im1;
            im2_ = im2;
            tic;
            [des_m1,des_m2] = RIFT_no_rotation_invariance(im1_,im2_,4,6,96);
            RIFT_Iinit_process_point_detection_and_description = toc;
            disp('Point detection');
            disp(RIFT_Iinit_process_point_detection_and_description);
            
            tic;
            disp('nearest matching')
            % nearest matching
            [indexPairs,matchmetric] = matchFeatures(des_m1.des,des_m2.des,'MaxRatio',1,'MatchThreshold', 100);
            matchedPoints1 = des_m1.kps(indexPairs(:, 1), :);
            matchedPoints2 = des_m2.kps(indexPairs(:, 2), :);
            [matchedPoints2,IA]=unique(matchedPoints2,'rows');
            matchedPoints1=matchedPoints1(IA,:);

            name_kp_bfmach_sar = "SAR_"+d+"_SUB_"+s+"m_"+n+"_before_mach.csv";
            name_kp_bfmach_eo = "EO_"+d+"_SUB_"+s+"m_"+n+"_before_mach.csv";
            RIFT_Iinit_maching = toc;
            disp(RIFT_Iinit_maching);
            disp('outlier removal')
            % Saving orginal points
            writematrix(des_m1.kps,'report_RIFT\'+name_kp_bfmach_sar);
            writematrix(des_m2.kps,'report_RIFT\'+name_kp_bfmach_sar);
            % outlier removal
            tic;
            H=FSC(matchedPoints1,matchedPoints2,'affine',2);
            Y_=H*[matchedPoints1';ones(1,size(matchedPoints1,1))];
            Y_(1,:)=Y_(1,:)./Y_(3,:);
            Y_(2,:)=Y_(2,:)./Y_(3,:);
            E=sqrt(sum((Y_(1:2,:)-matchedPoints2').^2));
            inliersIndex=E<3;
            cleanedPoints1 = matchedPoints1(inliersIndex, :);
            cleanedPoints2 = matchedPoints2(inliersIndex, :);
            Outlier_removal = toc;
            disp(Outlier_removal);
            name_kp_mach_sar = "SAR_"+d+"_SUB_"+s+"m_"+n+"_mach.csv";
            name_kp_mach_eo = "EO_"+d+"_SUB_"+s+"m_"+n+"_mach.csv";
            % saving mached and cleaned points
            writematrix(cleanedPoints1,'report_RIFT\'+name_kp_mach_sar);
            writematrix(cleanedPoints2,'report_RIFT\'+name_kp_mach_eo);
            disp('outlier removal')
            % name_process_time = d+"_SUB_"+s+"m_"+n+"_time.csv";
            total_time = RIFT_Iinit_process_point_detection_and_description + RIFT_Iinit_maching + Outlier_removal;
            time_list = [total_time, RIFT_Iinit_process_point_detection_and_description, RIFT_Iinit_maching, Outlier_removal];
            name_mach_sar_time = "SAR_"+d+"_SUB_"+s+"m_"+n+"_mach_time.csv";
            writematrix(time_list,'report_RIFT\'+name_mach_sar_time);

        end
    end
end
fclose(fid);
display("DONE")