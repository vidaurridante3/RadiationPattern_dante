az_folder = "/data/chaoyi_he/Radiation_Pattern/path/processed/az/";
el_folder = "/data/chaoyi_he/Radiation_Pattern/path/processed/vt/";

FL_ = 0.01:0.005:0.04;
W_ = 0.01:0.005:0.04;
L_ = 0.01:0.005:0.04;
H_ = 0.01:0.005:0.04;
angleE = 12.2442;
angleH = 14.4712;
[FL_grid, W_grid, L_grid, H_grid] = ndgrid(FL_, W_, L_, H_);
combinations = [L_grid(:), W_grid(:), L_grid(:), H_grid(:)];

for count = 1:size(combinations, 1)
    FL = combinations(count, 1); 
    W = combinations(count, 2);  
    L = combinations(count, 3);
    H = combinations(count, 4);
    [FW, FH] = hornangle2size(W, H, FL, angleE, angleH);
    FO=[L / 2 - 0.002 0];
    ant1 = horn('FlareLength',FL,'FlareWidth',FW,'FlareHeight',FH,'Length',L,...
                 'Width',W,'Height',H,'FeedOffset',FO);

    az_path = fullfile(az_folder, strcat('Horn_', "FL_", num2str(FL), ...
                                         "_FW_", num2str(FW), ...
                                         "_FH_", num2str(FH), ...
                                         "_L_", num2str(L), ...
                                         "_W_", num2str(W), ...
                                         "_H_", num2str(H), ...
                                         ".csv"));
    el_path = fullfile(el_folder, strcat('Horn_', "FL_", num2str(FL), ...
                                         "_FW_", num2str(FW), ...
                                         "_FH_", num2str(FH), ...
                                         "_L_", num2str(L), ...
                                         "_W_", num2str(W), ...
                                         "_H_", num2str(H), ...
                                         ".csv"));
                                     
    if exist(az_path, 'file') == 2 && exist(el_path, 'file') == 2
        continue
    end
                                     
    ant5=cassegrain;
    ant5.Exciter = ant1;
    ant5.Exciter.Tilt = 270;
    ant5.Exciter.TiltAxis = [0 1 0];
    az = 0:1:360;
    el = -180:1:180;
    patOpt = PatternPlotOptions;
    patOpt.MagnitudeScale = [-15 35];
%     figure;
%     pattern(ant5,10e9,az,el,'patternOptions',patOpt)
    [fieldval, azimuth, elevation] = pattern(ant5,10e9,az,el,'patternOptions',patOpt);

    az_index = find(elevation == 0);
    el_index = find(azimuth == 0);

    az_amp = fieldval(az_index, :);
    el_amp = fieldval(:, el_index);

    az = [azimuth', az_amp'];
    el = sortrows([[-90:-1:-180, 180:-1:-89]', [el_amp(1:length(-90:-1:-180)); el_amp(length(-90:-1:-180):end-1)]], 1);

    writematrix(az, az_path);
    writematrix(el, el_path);
end