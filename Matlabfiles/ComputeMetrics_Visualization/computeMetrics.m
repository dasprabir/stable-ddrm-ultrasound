function [TCR, CNR] = computeMetrics(x, roi_z, roi_b, roi_t, dBRange)
% Compute tissue-to-clutter ratio and contrast-to-noise ratio.
% Normalized enveloppe
%env = abs(hilbert(x));
%env = env / max(env(:));

% TCR
e_b = env(roi_b(1):roi_b(2), roi_z(1):roi_z(2)); mu_b = mean2(e_b); sigma_b = var(var(e_b'));
e_t = env(roi_t(1):roi_t(2), roi_z(1):roi_z(2)); mu_t = mean2(e_t); sigma_t = var(var(e_t'));
TCR  = 20*log10(mu_t/mu_b);

% compressed B-mode
bmode = 20*log10(env); 
bmode(bmode < -dBRange) = -dBRange; 

% CNR
bmode_b = bmode(roi_b(1):roi_b(2), roi_z(1):roi_z(2)); lin_b = 10.^(bmode_b/20); mu_b = mean(lin_b(:)); sigma_b = var(var(lin_b'));
bmode_t = bmode(roi_t(1):roi_t(2), roi_z(1):roi_z(2)); lin_t = 10.^(bmode_t/20); mu_t = mean(lin_t(:)); sigma_t = var(var(lin_t'));
CNR     = abs(mu_b - mu_t) / sqrt(sigma_b + sigma_t);
end

