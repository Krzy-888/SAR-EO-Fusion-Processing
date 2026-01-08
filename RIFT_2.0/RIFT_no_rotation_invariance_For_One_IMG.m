% This is a samplest implementation of the proposed RIFT algorithm. In this implementation,...
% rotation invariance part and corner point detection are not included.

function [des_m1] = RIFT_no_rotation_invariance_For_One_IMG(im1,s,o,patch_size)

% m1 and m2 are the maximum moment maps;
% eo1{s,o} = convolution result for scale s and orientation o.
% The real part is the result of convolving with the even symmetric filter,
% the imaginary part is the result from convolution with the odd symmetric filter.

[m1,~,~,~,~,eo1,~] = phasecong3(im1,s,o,3,'mult',1.6,'sigmaOnf',0.75,'g', 3, 'k',1);

a=max(m1(:)); b=min(m1(:)); m1=(m1-b)/(a-b);

% FAST detector on the maximum moment maps to extract edge feature points.
m1_points = detectFASTFeatures(m1,'MinContrast',0.05);
m1_points=m1_points.selectStrongest(2000000);   %number of keypoints can be set by users

% RIFT descriptor
des_m1 = RIFT_descriptor_no_rotation_invariance(im1, m1_points.Location,eo1, patch_size, s,o);



