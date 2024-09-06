az_folder = "/data/chaoyi_he/Radiation_Pattern/path/processed/az/";
el_folder = "/data/chaoyi_he/Radiation_Pattern/path/processed/vt/";

R_ = 0.005:0.002:0.02;
WH_ = 0.01:0.003:0.03;
CH_ = 0.01:0.003:0.03;
AR_ = 0.01:0.003:0.03;
[R_grid, WH_grid, CH_grid, AR_grid] = ndgrid(R_, WH_, CH_, AR_);
combinations = [R_grid(:), WH_grid(:), CH_grid(:), AR_grid(:)];

for count = 1:size(combinations, 1)
    R = combinations(count, 1); 
    WH = combinations(count, 2); 
    CH = combinations(count, 3);
    AR = combinations(count, 4);

    FW = min(0.002, WH / 8);
    FO = min(0.01, WH / 2 - 0.003);
    FH = min(0.0075, R - 0.0015);
    if AR <= R
        continue;
    end
    ant1 = hornConical('Radius',R,'WaveguideHeight',WH,'ConeHeight',CH,'ApertureRadius',AR, ...
                       'FeedWidth',FW,'FeedOffset',FO,'FeedHeight',FH);

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

    az_path = fullfile(az_folder, strcat('HornConical', ...
                                         "_R_", num2str(R), ...
                                         "_WH_", num2str(WH), ...
                                         "_CH_", num2str(CH), ...
                                         "_AR_", num2str(AR), ...
                                         ".csv"));
    el_path = fullfile(el_folder, strcat('HornConical', ...
                                         "_R_", num2str(R), ...
                                         "_WH_", num2str(WH), ...
                                         "_CH_", num2str(CH), ...
                                         "_AR_", num2str(AR), ...
                                         ".csv"));

    writematrix(az, az_path);
    writematrix(el, el_path);
end