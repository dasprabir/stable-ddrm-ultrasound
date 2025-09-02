clc; clear; close all;

%% Load RF Data and PSF
fprintf('Loading data...\n');
try
    load('L12-50-50mm_caro_5MHz_fr_dte.mat');   
    rf = double(raw_data);                      
    fprintf('RF data loaded: %dx%d\n', size(rf,1), size(rf,2));
    
    load('psf_pw_est_carotid_fr.mat');          
    psf_full = double(psf_est);
    fprintf('Original PSF loaded: %dx%d\n', size(psf_full,1), size(psf_full,2));
    
catch ME
    fprintf('Error loading data: %s\n', ME.message);
    return;
end

%% Proper PSF Cropping (Using Your Method)
fprintf('\n=== PSF CROPPING ANALYSIS ===\n');

% Find actual peak location (not geometric center)
max_val = max(abs(psf_full(:)));
fprintf('Maximum absolute value of PSF: %.6f\n', max_val);
[max_val, linear_idx] = max(abs(psf_full(:)));
[row, col] = ind2sub(size(psf_full), linear_idx);
fprintf('Max value = %.6f at (row, col) = (%d, %d)\n', max_val, row, col);

% Crop around actual peak
center = row;  % Automatically use peak location
crop_size = 15;  % This gives 31x31 (center ± 15)
cropped_psf = psf_full(center-crop_size:center+crop_size, ...
                       center-crop_size:center+crop_size);

fprintf('PSF cropped from %dx%d to %dx%d around actual peak\n', ...
        size(psf_full,1), size(psf_full,2), size(cropped_psf,1), size(cropped_psf,2));

% Verify cropped PSF
max_cropped_val = max(abs(cropped_psf(:)));
fprintf('Maximum absolute value of Cropped PSF: %.6f\n', max_cropped_val);


%% Compare Original vs Cropped PSF
figure('Position', [50, 50, 1400, 500]);

% Original PSF
subplot(1,4,1);
imagesc(psf_full); colorbar; axis image;
title('Original PSF (53x53)');
hold on;
% Mark the peak
plot(col, row, 'r+', 'MarkerSize', 15, 'LineWidth', 3);

% Cropped PSF
subplot(1,4,2);
imagesc(cropped_psf); colorbar; axis image;
title(sprintf('Cropped PSF (%dx%d)', size(cropped_psf,1), size(cropped_psf,2)));

% 3D surface plot of cropped PSF
subplot(1,4,3);
surf(cropped_psf); shading interp;
title('Surface Plot of Cropped PSF');
xlabel('X'); ylabel('Y'); zlabel('Amplitude');

% PSF profiles comparison
subplot(1,4,4);
% Original PSF profile through peak
orig_profile = psf_full(row, :);
% Cropped PSF profile through center
crop_center = round(size(cropped_psf,1)/2);
crop_profile = cropped_psf(crop_center, :);

plot(1:length(orig_profile), orig_profile, 'b-', 'LineWidth', 2, 'DisplayName', 'Original 53x53');
hold on;
plot(1:length(crop_profile), crop_profile, 'r-', 'LineWidth', 2, 'DisplayName', sprintf('Cropped %dx%d', size(cropped_psf,1), size(cropped_psf,2)));
xlabel('Sample'); ylabel('Amplitude');
title('PSF Peak Profiles');
legend; grid on;

%% Deconvolution with Both PSF Sizes
fprintf('\n=== COMPARING DECONVOLUTION: Original vs Peak-Centered Cropped PSF ===\n');

% Data preparation
rfn = rf / max(abs(rf(:)));
lambda = 1e-3;  % Ultra-low regularization

%% Deconvolution with Original PSF (53x53)
fprintf('Deconvolution with original PSF (%dx%d)...\n', size(psf_full,1), size(psf_full,2));

% L2 normalization
H_orig = psf_full / sqrt(sum(abs(psf_full(:)).^2));

% BCCB matrix creation
[Mh_orig, Nh_orig] = size(H_orig);
center_orig = round([Mh_orig, Nh_orig] / 2);
D_orig = fft2(circshift(padarray(H_orig, [size(rf,1) - Mh_orig, size(rf,2) - Nh_orig], 'post'), 1 - center_orig));

% Apply deconvolution
RF_freq = fft2(rfn);
Wiener_filter_orig = conj(D_orig) ./ (abs(D_orig).^2 + lambda + eps);
VivoTK_original = real(ifft2(RF_freq .* Wiener_filter_orig));

%% Deconvolution with Peak-Centered Cropped PSF
fprintf('Deconvolution with peak-centered cropped PSF (%dx%d)...\n', size(cropped_psf,1), size(cropped_psf,2));

% L2 normalization
H_crop = cropped_psf / sqrt(sum(abs(cropped_psf(:)).^2));

% BCCB matrix creation
[Mh_crop, Nh_crop] = size(H_crop);
center_crop = round([Mh_crop, Nh_crop] / 2);
D_crop = fft2(circshift(padarray(H_crop, [size(rf,1) - Mh_crop, size(rf,2) - Nh_crop], 'post'), 1 - center_crop));

% Apply deconvolution
Wiener_filter_crop = conj(D_crop) ./ (abs(D_crop).^2 + lambda + eps);
VivoTK_cropped = real(ifft2(RF_freq .* Wiener_filter_crop));


%% Parameters for spatial scaling
pitch = 0.1;     % mm
c = 1540;        % m/s
fs = 40e6;       % Hz

Nx = size(rf, 2);
Nz = size(rf, 1);
x = ((0:Nx-1) - Nx/2) * pitch;
z = (0:Nz-1) * (c / (2 * fs)) * 1000;

%% Convert to B-mode for visualization
% Original
env_orig = abs(hilbert(rfn));
env_orig = env_orig / max(env_orig(:));
bmode_orig = 20 * log10(env_orig + eps);
bmode_orig = min(max(bmode_orig, -50), 0);

% Deconvolved with original PSF (53x53)
env_original = abs(hilbert(VivoTK_original));
env_original = env_original / max(env_original(:));
bmode_original = 20 * log10(env_original + eps);
bmode_original = min(max(bmode_original, -50), 0);

% Deconvolved with cropped PSF (30x30)
env_cropped = abs(hilbert(VivoTK_cropped));
env_cropped = env_cropped / max(env_cropped(:));
bmode_cropped = 20 * log10(env_cropped + eps);
bmode_cropped = min(max(bmode_cropped, -50), 0);

