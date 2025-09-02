%% === FUNCTION ===
function bmode = rf2bmode(rf, dyn_range)
    env = abs(hilbert(rf, 1));              % Envelope detection (axial)
    env = env / max(env(:));               % Normalize
    bmode = 20 * log10(env + 1e-6);        % Log compression
    bmode = max(bmode, -dyn_range);        % Clip to dynamic range
end