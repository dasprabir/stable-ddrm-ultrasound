%% Enhanced evaluate_with_metrics_auto.m 
clc; clear; close all;

%% 1. Load raw RF data
load('invivo.mat');
rf   = data;
rfn  = rf / max(abs(rf(:)));  % normalized RF

%% 2. Tikhonov restoration (Wiener filter)
load('psf_estim_vivo.mat');
H    = psf_estim_vivo; 
h    = H / sum(abs(H(:)));
[Mh, Nh] = size(H);
center   = round([Mh, Nh]/2);
D = fft2(circshift(padarray(h, [size(rf,1)-Mh, size(rf,2)-Nh], 'post'), 1-center));
SNR_val = 500;  % ~30 dB
VivoTK  = ifft2( fft2(rfn) .* conj(D) ./ (SNR_val^(-1) + conj(D).*D), 'symmetric');
VivoTK  = VivoTK / max(abs(VivoTK(:)));

%% 3. DDRM restoration (deep diffusion prior)
load('0_-1.mat', 'image');   % variable 'image'
image     = permute(image, [2,3,1]);
%grayimage = image(:,:,1);
grayimage = mean(image,3);

%% 4. ────── Envelope detection and log compression ──────
[M1,N1] = size(rf);
[M2,N2] = size(VivoTK);
[M3,N3] = size(grayimage);
M = min([M1,M2,M3]); N = min([N1,N2,N3]);

% Crop to same size
rf       = rf(1:M, 1:N);
VivoTK   = VivoTK(1:M,1:N);
grayimage= grayimage(1:M,1:N);

% Raw data processing
env_raw = abs(hilbert(rf));                 % Envelope
env_raw = env_raw / max(env_raw(:));        % Normalize
bmode_raw = 20 * log10(env_raw + eps);      % Log compression
bmode_raw = max(bmode_raw, -30);            % Clip to dynamic range

% Tikhonov processing  
env_tikh = abs(VivoTK);                     % Envelope (already complex)
env_tikh = env_tikh / max(env_tikh(:));     % Normalize
bmode_tikh = 20 * log10(env_tikh + eps);   % Log compression
bmode_tikh = max(bmode_tikh, -30);          % Clip to dynamic range

% DDRM processing
env_ddrm = abs(grayimage);         % Envelope
env_ddrm = env_ddrm / max(env_ddrm(:));     % Normalize
bmode_ddrm = 20 * log10(env_ddrm + eps);   % Log compression
bmode_ddrm = max(bmode_ddrm, -30);          % Clip to dynamic range


%% 6. Simplified B-mode Visualization (Only 3 Results + Shared Colorbar + Axial Resolution Labels)
g = figure('Name', 'B-mode Comparison', 'Position', [100, 100, 1200, 400]);

titles = {'B-mode Raw', 'Tikhonov', 'DDRM'};
images = {bmode_raw, bmode_tikh, bmode_ddrm};


% Plot all three B-mode results
for i = 1:3
    subplot(1, 3, i);
    imagesc(images{i});
    colormap gray;
    axis image on;
    caxis([-30, 0]);
    title(titles{i}, 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Lateral (px)', 'FontSize', 12);
    ylabel('Axial (px)', 'FontSize', 12);
    set(gca, 'FontSize', 12);
end


% Add unified colorbar manually on the right
cb = colorbar('Position', [0.93 0.15 0.015 0.7]);  % adjust for layout
ylabel(cb, 'Amplitude (dB)', 'FontSize', 12);
cb.FontSize = 12;

sgtitle('Ultrasound B-mode Comparison with Resolution','FontSize',16, 'FontWeight','bold');

%% 6. Simplified B-mode Visualization with Cropping and Shared Colorbar
g = figure('Name', 'B-mode Comparison (Cropped)', 'Position', [100, 100, 1200, 400]);

titles = {'B-mode Raw', 'Tikhonov', 'DDRM'};

% Define cropping indices (adjust as needed)
crop_rows = 1:401;    % Axial
crop_cols = 1:200;    % Lateral

% Crop all three B-mode images
bmode_raw_crop  = bmode_raw(crop_rows, crop_cols);
bmode_tikh_crop = bmode_tikh(crop_rows, crop_cols);
bmode_ddrm_crop = bmode_ddrm(crop_rows, crop_cols);

% Store cropped images in a cell array
images = {bmode_raw_crop, bmode_tikh_crop, bmode_ddrm_crop};

% Plot all three cropped B-mode results
for i = 1:3
    subplot(1, 3, i);
    imagesc(images{i});
    colormap gray;
    axis image on;
    caxis([-30, 0]);  % Adjust dB dynamic range
    title(titles{i}, 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Lateral (px)', 'FontSize', 12);
    ylabel('Axial (px)', 'FontSize', 12);
    set(gca, 'FontSize', 12);
end

% Add shared colorbar on the right
cb = colorbar('Position', [0.93 0.15 0.015 0.7]);
ylabel(cb, 'Amplitude (dB)', 'FontSize', 12, 'Rotation', 90);
cb.FontSize = 12;

% Super title
sgtitle('Ultrasound B-mode Comparison (Cropped)','FontSize',16, 'FontWeight','bold');
figure;
CR_raw  = ContrastRatio(env_raw);
figure;
CR_tikh = ContrastRatio(env_tikh);
figure;
CR_ddrm = ContrastRatio(env_ddrm);

fprintf('\nContrast Ratio (CR):\n');
fprintf('  Raw vivo      : %.2f dB\n', CR_raw);
fprintf('  Tikhonov        : %.2f dB\n', CR_tikh);
fprintf('  DDRM            : %.2f dB\n\n', CR_ddrm);

% Inputs: log envelope images (e.g., cropped)
input_ref = bmode_raw;
input_tikh = bmode_tikh;
input_ddrm = bmode_ddrm;

% Compute RG values
RG_tikh = resolution_gain(input_ref, input_tikh);
RG_ddrm = resolution_gain(input_ref, input_ddrm);

% Print results
fprintf('Resolution Gain (RG) relative to Raw B-mode:\n');
fprintf('  Tikhonov  : %.3f\n', RG_tikh);
fprintf('  DDRM      : %.3f\n', RG_ddrm);



