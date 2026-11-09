function G = design_pid(wc, pm, writeConfig)
%DESIGN_PID  PID design for the identified Tello plant with pidtune (Lab3 method), writes config/gains.json.
%   G = design_pid()                 % defaults: crossover wc = 0.35/L (rad/s), phase margin 60 deg, write json
%   G = design_pid(1.6, 60, true)    % explicit crossover / phase margin
%
%   The controller outputs a desired wall velocity v (cm/s); the Actuator inverts B (rc -> cm/s), so every axis
%   is the normalised plant  P(s) = e^{-L s} / (s (tau s + 1)).  C(s) = Kp + Ki/s + Kd s/(Tf s + 1), N = 1/Tf
%   (Simulink PID block 'Parallel' form, = ftrack/control.py PID).
%   'cross' gains drive the cross-track error of the path follower; 'point_axes' (wc x 0.8) the laser tracker.
%   Prints / plots the closed-loop step specs (rise time, settling time 2 %, overshoot) for the report.

here = fileparts(mfilename('fullpath'));
cfg = fullfile(here, '..', 'config');
Nmax = 15;       % derivative filter <= 15 rad/s: the camera loop runs at ~30 Hz (Nyquist ~94 rad/s), noise
kiRatio = 0;     % optional minimum integral (Ki >= kiRatio*wc*Kp); 0 = keep pidtune's Ki. 0.1 removes drift bias
                 % faster but adds a slow tail (2 % settling ~10 s in the linear step test)
Mj = jsondecode(fileread(fullfile(cfg, 'model.json')));
if nargin < 2 || isempty(pm), pm = 60; end
if nargin < 3, writeConfig = true; end
gj = jsondecode(fileread(fullfile(cfg, 'gains.json')));
axesList = ["x", "y"];
fig = figure('Color', 'w', 'Position', [60 60 1250 760]);
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact');
G = struct();
for k = 1:2
    ax = axesList(k);
    p = Mj.axes.(ax);
    P = tf(1, [p.tau 1 0], 'InputDelay', p.L);
    if nargin < 1 || isempty(wc), w = 0.35 / max(p.L, 0.05); else, w = wc; end
    w = min(w, 3.0);
    o = pidtuneOptions('PhaseMargin', pm, 'DesignFocus', 'reference-tracking');
    [C, info] = pidtune(P, 'PIDF', w, o);
    [Cp, infop] = pidtune(P, 'PIDF', 0.8 * w, o);
    C = practical(C, w, Nmax, kiRatio);  Cp = practical(Cp, 0.8 * w, Nmax, kiRatio);
    T = feedback(C * P, 1);
    [ys, ts_] = step(T, 0:0.005:60);
    S = stepinfo(ys, ts_, 1, 'SettlingTimeThreshold', 0.02);
    S5 = stepinfo(ys, ts_, 1, 'SettlingTimeThreshold', 0.05);
    S.SettlingTime5 = S5.SettlingTime;
    [Gm, Pm, ~, Wcp] = margin(C * P);
    G.(ax) = struct('C', C, 'Cpoint', Cp, 'wc', w, 'PM', info.PhaseMargin, 'step', S, 'GM_dB', 20 * log10(Gm), 'Wcp', Wcp);
    fprintf('axis %s: plant tau=%.3f L=%.3f | Kp=%.3f Ki=%.3f Kd=%.3f N=%.1f | wc=%.2f PM=%.1f GM=%.1f dB | tr=%.2f ts(2%%)=%.2f ts(5%%)=%.2f OS=%.1f%%\n', ...
        upper(ax), p.tau, p.L, C.Kp, C.Ki, C.Kd, 1 / max(C.Tf, 1e-6), Wcp, Pm, 20 * log10(Gm), S.RiseTime, S.SettlingTime, S.SettlingTime5, S.Overshoot);
    nexttile(tl, k);
    step(T, feedback(Cp * P, 1), 6); grid on;
    legend('cross-track (path)', 'point (laser)', 'Location', 'southeast');
    title(sprintf('axis %s closed-loop step: t_r=%.2f s, t_s=%.2f s, M_p=%.1f %%', upper(ax), S.RiseTime, S.SettlingTime, S.Overshoot));
    nexttile(tl, k + 2);
    margin(C * P); grid on;
    gj.cross.(ax) = struct('Kp', C.Kp, 'Ki', C.Ki, 'Kd', C.Kd, 'N', 1 / max(C.Tf, 1e-6));
    gj.point_axes.(ax) = struct('Kp', Cp.Kp, 'Ki', Cp.Ki, 'Kd', Cp.Kd, 'N', 1 / max(Cp.Tf, 1e-6));
end
exportgraphics(fig, fullfile(cfg, 'design_pid.png'), 'Resolution', 140);
if writeConfig
    gj.x_doc = sprintf('PIDF from matlab/design_pid.m (pidtune, PM %g deg) for P(s)=e^{-Ls}/(s(tau s+1)) of %s', pm, string(Mj.source));
    gj.source = "design_pid:" + string(Mj.source);
    fid = fopen(fullfile(cfg, 'gains.json'), 'w'); fprintf(fid, '%s', jsonencode(gj, 'PrettyPrint', true)); fclose(fid);
    fprintf('wrote %s\n', fullfile(cfg, 'gains.json'));
end
end

function C = practical(C, wc, Nmax, kiRatio)
% pidtune result -> implementable controller: limit the derivative filter, keep a minimum integral action
Tf = max(C.Tf, 1 / Nmax);
Ki = max(C.Ki, kiRatio * wc * C.Kp);
C = pid(C.Kp, Ki, C.Kd, Tf);
end
