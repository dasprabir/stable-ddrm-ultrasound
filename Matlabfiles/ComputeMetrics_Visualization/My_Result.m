%% Enhanced evaluate_with_metrics_auto.m 
clc; clear; close all;

%% 1. Load raw RF data
load('newData.mat', 'matImage');
rf   = matImage;
rfn  = rf / max(abs(rf(:)));  % normalized RF

%% 2. Tikhonov restoration (Wiener filter)
load('PSF_crop.mat', 'cropped_psf');
H    = cropped_psf; 
h    = H / sum(abs(H(:)));
[Mh, Nh] = size(H);
center   = round([Mh, Nh]/2);
D = fft2(circshift(padarray(h, [size(rf,1)-Mh, size(rf,2)-Nh], 'post'), 1-center));
SNR_val = 100;  % ~30 dB
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
bmode_raw = max(bmode_raw, -50);            % Clip to dynamic range

% Tikhonov processing  
env_tikh = abs(VivoTK);                     % Envelope (already complex)
env_tikh = env_tikh / max(env_tikh(:));     % Normalize
bmode_tikh = 20 * log10(env_tikh + eps);   % Log compression
bmode_tikh = max(bmode_tikh, -50);          % Clip to dynamic range

% DDRM processing
env_ddrm = abs(grayimage);         % Envelope
env_ddrm = env_ddrm / max(env_ddrm(:));     % Normalize
bmode_ddrm = 20 * log10(env_ddrm + eps);   % Log compression
bmode_ddrm = max(bmode_ddrm, -50);          % Clip to dynamic range


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
    caxis([-50, 0]);
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


%% 6. Contrast Ratio
fprintf('\n--- CONTRAST RATIO (CR) ---\n');
figure;
fprintf('Raw CR:       %.4f\n', ContrastRatio(env_raw));
figure;
fprintf('Tikhonov CR:  %.4f\n', ContrastRatio(env_tikh));
figure;
fprintf('DDRM CR:  %.4f\n', ContrastRatio(env_ddrm));

%% 7. Resolution Gain on envelope
fprintf('\n--- RESOLUTION GAIN on Envelope ---\n');
RG_tikh = resolution_gain(env_raw,env_tikh);
fprintf('Tikhonov vs Raw: RG = %.3f\n', RG_tikh);
RG_ddrm = resolution_gain(env_raw, env_ddrm);
fprintf('DDRM vs Raw: RG = %.3f\n', RG_ddrm);


%% 7. Resolution Gain on Bmode
fprintf('\n--- RESOLUTION GAIN on bmode ---\n');
RG_tikh = resolution_gain(bmode_raw,bmode_tikh);
fprintf('Tikhonov vs Raw: RG = %.3f\n', RG_tikh);
RG_ddrm = resolution_gain(bmode_raw, bmode_ddrm);
fprintf('DDRM vs Raw: RG = %.3f\n', RG_ddrm);


%% Combined CR boxplots (Raw, Tikhonov, DDRM)

SZ = 8; SX = 8;

% --- Raw ---
[Nz, Nx] = size(env_raw);
Nz = floor(Nz/SZ)*SZ; Nx = floor(Nx/SX)*SX;
PWTD = env_raw(1:Nz, end-Nx+1:end);
N11 = SZ*ones(1,Nz/SZ); N22 = SX*ones(1,Nx/SX);
PWTD2 = reshape(mat2cell(PWTD,N11,N22),[],1);
patchs = nan(numel(PWTD2),1);
for i=1:numel(PWTD2)
    R = PWTD2{i};
    if any(R(:)), patchs(i) = mean(R(:).^2); end
end
[~,I] = min(patchs); Ref = patchs(I);
CR_raw = 10*log10(patchs./Ref); CR_raw = CR_raw(isfinite(CR_raw)&CR_raw>0);
med_raw = median(CR_raw);

% --- Tikhonov ---
[Nz, Nx] = size(env_tikh);
Nz = floor(Nz/SZ)*SZ; Nx = floor(Nx/SX)*SX;
PWTD = env_tikh(1:Nz, end-Nx+1:end);
N11 = SZ*ones(1,Nz/SZ); N22 = SX*ones(1,Nx/SX);
PWTD2 = reshape(mat2cell(PWTD,N11,N22),[],1);
patchs = nan(numel(PWTD2),1);
for i=1:numel(PWTD2)
    R = PWTD2{i};
    if any(R(:)), patchs(i) = mean(R(:).^2); end
end
[~,I] = min(patchs); Ref = patchs(I);
CR_tikh = 10*log10(patchs./Ref); CR_tikh = CR_tikh(isfinite(CR_tikh)&CR_tikh>0);
med_tikh = median(CR_tikh);

% --- DDRM ---
[Nz, Nx] = size(env_ddrm);
Nz = floor(Nz/SZ)*SZ; Nx = floor(Nx/SX)*SX;
PWTD = env_ddrm(1:Nz, end-Nx+1:end);
N11 = SZ*ones(1,Nz/SZ); N22 = SX*ones(1,Nx/SX);
PWTD2 = reshape(mat2cell(PWTD,N11,N22),[],1);
patchs = nan(numel(PWTD2),1);
for i=1:numel(PWTD2)
    R = PWTD2{i};
    if any(R(:)), patchs(i) = mean(R(:).^2); end
end
[~,I] = min(patchs); Ref = patchs(I);
CR_ddrm = 10*log10(patchs./Ref); CR_ddrm = CR_ddrm(isfinite(CR_ddrm)&CR_ddrm>0);
med_ddrm = median(CR_ddrm);

% --- Plot all in one figure ---
figure;
data = [CR_raw(:); CR_tikh(:); CR_ddrm(:)];
grp  = [repmat({'Raw'},numel(CR_raw),1); ...
        repmat({'Tikhonov'},numel(CR_tikh),1); ...
        repmat({'DDRM'},numel(CR_ddrm),1)];
boxplot(data, grp, 'Whisker', 2);
ylabel('CR [dB]'); title('Contrast Ratio across Methods');


%% RG on B-mode (windowed + central crop) — append-only
[H,W] = size(bmode_raw);

% 2D taper (outer product of 1D Hanns)
wz = hann(H); 
wx = hann(W); 
win = wz * wx.';   % HxW

% central crop (10% margin)
r = round(0.1*H); c = round(0.1*W);
CR = r+1 : H-r; 
CC = c+1 : W-c;

RG_bmode_tikh = resolution_gain(bmode_raw(CR,CC).*win(CR,CC), ...
                                bmode_tikh(CR,CC).*win(CR,CC));
RG_bmode_ddrm = resolution_gain(bmode_raw(CR,CC).*win(CR,CC), ...
                                bmode_ddrm(CR,CC).*win(CR,CC));

fprintf('\n--- RESOLUTION GAIN on B-mode (windowed & cropped) ---\n');
fprintf('Tikhonov vs Raw: RG = %.3f\n', RG_bmode_tikh);
fprintf('DDRM     vs Raw: RG = %.3f\n', RG_bmode_ddrm);



