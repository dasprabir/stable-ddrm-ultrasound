%% enhanced_evaluate_with_metrics_auto.m
% End-to-end pipeline + vector-PDF exports (single + multi-page)
clc; clear; close all;

%% ------------------------ 0) I/O settings ------------------------
% Input files (adjust if your filenames differ)
rf_file    = 'invivo.mat';             % contains variable `data` (RF)
psf_file   = 'psf_estim_vivo.mat';     % contains variable `psf_estim_vivo`
ddrm_file  = '0_-1.mat';               % contains variable `image` (H x W x 3 or similar)

% Output files
pdf_single_full     = 'Bmode_Comparison.pdf';
pdf_single_cropped  = 'Bmode_Comparison_Cropped.pdf';
pdf_report_all      = 'Ultrasound_Bmode_Report.pdf';

%% ------------------------ 1) Load data ---------------------------
try
    S = load(rf_file);
    if isfield(S,'data'); rf = S.data;
    elseif isfield(S,'rf'); rf = S.rf;
    else, error('RF file must contain variable `data` or `rf`.');
    end
catch ME
    error('Failed to load RF data: %s', ME.message);
end

rfn = rf / max(abs(rf(:)));  % normalize RF

try
    S = load(psf_file);
    if isfield(S,'psf_estim_vivo'); H = S.psf_estim_vivo;
    elseif isfield(S,'H'); H = S.H;
    else, error('PSF file must contain `psf_estim_vivo` or `H`.');
    end
catch ME
    error('Failed to load PSF: %s', ME.message);
end

try
    S = load(ddrm_file, 'image');   % variable 'image' expected
    image = S.image;
    if ndims(image)==3
        % assume (C,H,W) or (H,W,C) — adapt to your repo format
        if size(image,1) <= 4 && size(image,3) > 4
            % probably (C,H,W) -> (H,W,C)
            image = permute(image, [2,3,1]);
        end
        grayimage = mean(image,3);
    else
        grayimage = image;  % already single-channel
    end
catch ME
    warning('Failed to load DDRM image (using zeros placeholder): %s', ME.message);
    grayimage = zeros(size(rf));
end

%% ------------------------ 2) Tikhonov (Wiener-like) ---------------
% Normalize PSF and build frequency response (wrap-centered)
h = H / sum(abs(H(:)) + eps);
[Mh, Nh] = size(h);
[Mrf, Nrf] = size(rf);
center = round([Mh, Nh]/2);

% pad PSF to RF size, center align, then FFT
hp = padarray(h, [Mrf-Mh, Nrf-Nh], 'post');
hp = circshift(hp, 1-center);
D  = fft2(hp);

% Deconvolution (Wiener-like/Tikhonov): X = Y * conj(D) / (1/SNR + |D|^2)
SNR_val = 500;  % ~27 dB (adjust as needed)
VivoTK  = ifft2( fft2(rfn).*conj(D) ./ ( (SNR_val.^(-1)) + (conj(D).*D) ), 'symmetric');
VivoTK  = VivoTK / max(abs(VivoTK(:)) + eps);

%% ------------------------ 3) Size harmonization -------------------
[M1,N1] = size(rf);
[M2,N2] = size(VivoTK);
[M3,N3] = size(grayimage);
M = min([M1,M2,M3]); N = min([N1,N2,N3]);

rf        = rf(1:M, 1:N);
VivoTK    = VivoTK(1:M, 1:N);
grayimage = grayimage(1:M, 1:N);

%% ------------------------ 4) Envelope + Log Compression -----------
% Raw (RF -> envelope via analytic signal)
env_raw = abs(hilbert(rf));
env_raw = env_raw / max(env_raw(:) + eps);
bmode_raw = 20*log10(env_raw + eps);
bmode_raw = max(bmode_raw, -30);    % clamp to [-30,0] dB

% Tikhonov (already real-valued magnitude image)
env_tikh = abs(VivoTK);
env_tikh = env_tikh / max(env_tikh(:) + eps);
bmode_tikh = 20*log10(env_tikh + eps);
bmode_tikh = max(bmode_tikh, -30);

% DDRM restoration
env_ddrm = abs(grayimage);
env_ddrm = env_ddrm / max(env_ddrm(:) + eps);
bmode_ddrm = 20*log10(env_ddrm + eps);
bmode_ddrm = max(bmode_ddrm, -30);

%% ------------------------ 5) Full B-mode 3-panel -------------------
titles = {'B-mode Raw', 'Tikhonov', 'DDRM'};
imgs   = {bmode_raw, bmode_tikh, bmode_ddrm};

g1 = figure('Name','B-mode Comparison','Position',[100,100,1200,420]);
for i = 1:3
    subplot(1,3,i);
    imagesc(imgs{i}); colormap gray; axis image on;
    caxis([-30, 0]); set(gca,'FontSize',12);
    title(titles{i}, 'FontSize',14,'FontWeight','bold');
    xlabel('Lateral (px)'); ylabel('Axial (px)');
end
cb1 = colorbar('Position',[0.93 0.15 0.015 0.7]);
ylabel(cb1,'Amplitude (dB)','FontSize',12);
sgtitle('Ultrasound B-mode Comparison with Resolution','FontSize',16,'FontWeight','bold');

% Single-page vector PDF export (full)
exportgraphics(g1, pdf_single_full, 'ContentType','vector');

%% ------------------------ 6) Cropped B-mode 3-panel ----------------
% Adjust crop as needed
crop_rows = 1:min(401,M);
crop_cols = 1:min(200,N);

bmode_raw_c  = bmode_raw(crop_rows, crop_cols);
bmode_tikh_c = bmode_tikh(crop_rows, crop_cols);
bmode_ddrm_c = bmode_ddrm(crop_rows, crop_cols);

imgs_c = {bmode_raw_c, bmode_tikh_c, bmode_ddrm_c};

g2 = figure('Name','B-mode Comparison (Cropped)','Position',[100,100,1200,420]);
for i = 1:3
    subplot(1,3,i);
    imagesc(imgs_c{i}); colormap gray; axis image on;
    caxis([-30, 0]); set(gca,'FontSize',12);
    title(titles{i}, 'FontSize',14,'FontWeight','bold');
    xlabel('Lateral (px)'); ylabel('Axial (px)');
end
cb2 = colorbar('Position',[0.93 0.15 0.015 0.7]);
ylabel(cb2,'Amplitude (dB)','FontSize',12);
sgtitle('Ultrasound B-mode Comparison (Cropped)','FontSize',16,'FontWeight','bold');

% Single-page vector PDF export (cropped)
exportgraphics(g2, pdf_single_cropped, 'ContentType','vector');

%% ------------------------ 7) CR plots (capture handles) -----------
% These assume you have a function ContrastRatio(env) that PLOTS a figure
% and returns a numeric CR value. If it doesn't plot, wrap your own figure.
fCR = [];  % to collect figure handles if created
try
    figure; CR_raw  = ContrastRatio(env_raw);  fCR(end+1) = gcf; %#ok<SAGROW>
catch, warning('ContrastRatio failed for RAW.'); CR_raw = NaN; end

try
    figure; CR_tikh = ContrastRatio(env_tikh); fCR(end+1) = gcf; %#ok<SAGROW>
catch, warning('ContrastRatio failed for Tikhonov.'); CR_tikh = NaN; end

try
    figure; CR_ddrm = ContrastRatio(env_ddrm); fCR(end+1) = gcf; %#ok<SAGROW>
catch, warning('ContrastRatio failed for DDRM.'); CR_ddrm = NaN; end

fprintf('\nContrast Ratio (CR) [dB]:\n');
fprintf('  Raw       : %.2f dB\n', CR_raw);
fprintf('  Tikhonov  : %.2f dB\n', CR_tikh);
fprintf('  DDRM      : %.2f dB\n\n', CR_ddrm);

%% ------------------------ 8) Resolution Gain ----------------------
% Assumes resolution_gain(ref_bmode, test_bmode) returns a scalar.
try
    RG_tikh = resolution_gain(bmode_raw, bmode_tikh);
catch, warning('resolution_gain failed for Tikhonov.'); RG_tikh = NaN; end

try
    RG_ddrm = resolution_gain(bmode_raw, bmode_ddrm);
catch, warning('resolution_gain failed for DDRM.'); RG_ddrm = NaN; end

fprintf('Resolution Gain (RG) relative to Raw B-mode:\n');
fprintf('  Tikhonov  : %.3f\n', RG_tikh);
fprintf('  DDRM      : %.3f\n', RG_ddrm);

%% 9) Save all figures as individual vector PDFs, then try to merge



%% ------------------------ 10) (Optional) PNG copies ---------------
% If you also want high-DPI PNGs for slides (raster on purpose):
% exportgraphics(g1, 'Bmode_Comparison.png', 'Resolution', 300);
% exportgraphics(g2, 'Bmode_Comparison_Cropped.png', 'Resolution', 300);
