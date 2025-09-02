clc;
clear;
close all;

%load('rf512.mat', 'data')
%load('exp_data2.mat')
%load('picmus_simu1.mat')
load('rf_image_invivo.mat', 'rf_image')

rf = rf_image;
rf = double(rf);
rfn = rf / max(abs(rf(:)));

% Display shape
[Nz, Nx] = size(rfn);
fprintf('Normalized RF shape: [%d × %d]\n', Nz, Nx);

% Estimate PSF using Oleg Michalovich method
ncf = 0.86;  % Normalized center frequency (adjust as needed)
psf_full = psf_est(rf, ncf);

% Display full estimated PSF
figure;
imagesc(psf_full);  axis image;
title('Full Estimated PSF');
colorbar;

max_val = max(abs(psf_full(:)));
fprintf('Maximum absolute value of PSF: %.6f\n', max_val);

[max_val, linear_idx] = max(abs(psf_full(:)));
[row, col] = ind2sub(size(psf_full), linear_idx);
fprintf('Max value = %.6f at (row, col) = (%d, %d)\n', max_val, row, col);

imagesc(psf_full);

axis image;
hold on;
plot(305, 305, 'r+', 'MarkerSize', 10, 'LineWidth', 2); % red cross at the peak
title('PSF with Center Marked');
colorbar;

center = 615;  % Automatically use peak location

crop_size = 15;
cropped_psf = psf_full(center-crop_size:center+crop_size, ...
                       center-crop_size:center+crop_size);

% Display cropped PSF
figure;
subplot(1,2,1);
imagesc(cropped_psf); axis image;
title('Cropped PSF (30x30)');
colorbar;

% 3D surface plot
subplot(1,2,2);
surf(cropped_psf); shading interp;
title('Surface Plot of Cropped PSF');


max_cropped_val = max(abs(cropped_psf(:)));
fprintf('Maximum absolute value of Cropped PSF: %.6f\n', max_cropped_val);

save('psf_besson_oleg.mat', 'cropped_psf');
fprintf('Cropped PSF saved to psf_besson_oleg.mat\n');

