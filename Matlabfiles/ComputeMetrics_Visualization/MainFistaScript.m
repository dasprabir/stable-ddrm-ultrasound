%% Simulation Data
%load('psf_GT_0.mat')
%load('KidneyRF_GT_MIR.mat')



load('PSF_crop.mat')
load('newData.mat', 'matImage');


maximum_iterations = 1e3; 
SNRdb=20 ; % define added noise level, in dB

%rf = single(data);
rf = single(matImage);
L = size(rf);
%F = generate_stationary_psf_operator(psf_ref, L);
F = generate_stationary_psf_operator(cropped_psf, L);
KidGTConv = F.forward(rf);
maxPSNR = 0;

sigma = sqrt(mean(KidGTConv(:).^2)*10^(-SNRdb/10));
KidGTConvNoise = KidGTConv + sigma * randn(size(KidGTConv)); 


for p = 1.3%[1 1.3 1.5 2]  
for lambda = 0.005%[0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10]

tic
res = lp_deconvolution(p, lambda, maximum_iterations, KidGTConvNoise, F);
toc
res2=real(res.x);
PSNR=US_ADM_calc_PSNR(rf,res2).PSNR;
SSIM=ssim(res2,rf);
if (PSNR>maxPSNR)
maxPSNR = PSNR;
end
fprintf('PSNR: %.2f SSIM: %.4f lambda: %f p: %.1f\n', PSNR, SSIM, lambda, p);

end
end


fprintf('maxPSNR: %.2f' , maxPSNR);
imagesc(abs((res2)))
colormap(gray)

%% Invivo Data
load('Estimated_PSF.mat')
load('InVivoData.mat')

maximum_iterations = 1e3; 

Mydata = single(InVivoData);
L = size(Mydata);
F = generate_stationary_psf_operator(Estimated_PSF, L);

for p = 1.3%[1 1.3 1.5 2]  
for lambda = 0.005%[0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10]

tic
res = lp_deconvolution(p, lambda, maximum_iterations, Mydata, F);
toc
res2=real(res.x);


end
end


imagesc(abs((res2)))
colormap(gray)
