az_folder = "/data/chaoyi_he/Radiation_Pattern/path/processed/az/";
el_folder = "/data/chaoyi_he/Radiation_Pattern/path/processed/vt/";

z1_ = 0.004:0.002:0.02;
p_ = 0.001:0.0005:0.004;
w_ = 0.3:0.1:0.7;
d1_ = 0.002:0.0005:0.004;
ar_ = 0.02:0.005:0.04;
ch_ = 0.02:0.005:0.04;
[z1_grid, p_grid, w_grid, d1_grid, ar_grid, ch_grid] = ndgrid(z1_, p_, w_, d1_, ar_, ch_);
combinations = [z1_grid(:), p_grid(:), w_grid(:), d1_grid(:), ar_grid(:), ch_grid(:)];

for count = 1:size(combinations, 1)
    z1 = combinations(count, 1); 
    p = combinations(count, 2); 
    w = combinations(count, 3) * p;
    d1 = combinations(count, 4);
    ar = combinations(count, 5);
    ch = combinations(count, 6);

    if z1 <= p
        continue;
    end
    
    ant1 = hornConicalCorrugated('FirstCorrugateDistance',z1,'Pitch',p,'CorrugateWidth'...
                                 ,w,'CorrugateDepth',d1,'ConeHeight',ch,'ApertureRadius',ar);

    ant5=cassegrain;
    ant5.Exciter = ant1;
    ant5.Exciter.Tilt = 270;
    ant5.Exciter.TiltAxis = [1 0 0];
    az = 0:1:360;
    el = -180:1:180;
    patOpt = PatternPlotOptions;
    patOpt.MagnitudeScale = [-15 35];
    [fieldval, azimuth, elevation] = pattern(ant5,10e9,az,el,'patternOptions',patOpt);

    az_index = find(elevation == 0);
    el_index = find(azimuth == 0);

    az_amp = fieldval(az_index, :);     % patternAzimuth(ant5, 10e9)
    el_amp = fieldval(:, el_index);     % patternElevation(ant5, 10e9)
%     az(:, 1) = az(:, 1) * pi / 180;
%     el(:, 1) = el(:, 1) * pi / 180;
%     polarplot(az(:, 1), az(:, 2) - min(az(:, 2)));

    az = [azimuth', az_amp'];
    el = sortrows([[-90:-1:-180, 180:-1:-89]', [el_amp(1:length(-90:-1:-180)); el_amp(length(-90:-1:-180):end-1)]], 1);

    az_path = fullfile(az_folder, strcat('hornConicalCorrugated', ...
                                         "_z1_", num2str(z1), ...
                                         "_p_", num2str(p), ...
                                         "_w_", num2str(w), ...
                                         "_d1_", num2str(d1), ...
                                         "_ar_", num2str(ar), ...
                                         "_ch_", num2str(ch), ...
                                         ".csv"));
    el_path = fullfile(el_folder, strcat('hornConicalCorrugated', ...
                                         "_z1_", num2str(z1), ...
                                         "_p_", num2str(p), ...
                                         "_w_", num2str(w), ...
                                         "_d1_", num2str(d1), ...
                                         "_ar_", num2str(ar), ...
                                         "_ch_", num2str(ch), ...
                                         ".csv"));

    writematrix(az, az_path);
    writematrix(el, el_path);
end