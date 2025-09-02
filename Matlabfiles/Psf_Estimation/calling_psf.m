clc
close all
clear all


%load('carotid_cross.mat')
load('exp_data.mat')
% Extract the normalized RF matrix
rf = data;  % assuming the variable was saved as 'data'

% Display RF shape
[Nz, Nx] = size(rf);
fprintf('Normalized RF shape: [%d × %d]\n', Nz, Nx);

%psf = psf_est(rf', 0.86);
psf = psf_est(rf, 0.86);
%psf = psf';
imagesc(psf)

center = 257;
cropped_psf = psf(center-15:center+15, center-15:center+15);
figure
imagesc(cropped_psf)
colorbar
figure
surf(cropped_psf)

