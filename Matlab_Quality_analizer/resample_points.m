function data_new = resample_points(data, default_grd, scales)
% RESAMPLEPOINTS
%
% data         - dane wejściowe (np. wektor lub macierz)
% default_grd  - wartość referencyjna
% scales       - wektor skal
%
% data_new 

    data_new = cell(1, numel(scales));

    for i = 1:numel(scales)
        scale = scales(i);
        denominator = scale / default_grd;
        data_new{i} = data / denominator;
    end
end