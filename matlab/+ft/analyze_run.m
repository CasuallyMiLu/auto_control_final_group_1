function out = analyze_run(runDir)
%FT.ANALYZE_RUN  post-flight analysis of runs/<run>/log.csv + meta.json -> metrics.json, metrics.md, fig_*.png
%   Cross-track error e_n(t) of the TRACK phase (cm, + = left of the travel direction); the release from the
%   staging point is a step of size e0 = e_n(t_start):
%     rise time  t_r : |e_n| from 90 % to 10 % of |e0|
%     overshoot  M_p : largest excursion to the other side of the line (cm, % of |e0|), peak time t_p
%     settling   t_s : first time after which |e_n| stays within +-5 cm (also +-3 cm). Square: step window ends
%                      20 cm before the first corner; corners are reported separately (max |e_n| within +-25 cm)
%     steady state (t > t_s, rest of the lap): mean e_n (steady-state error), RMS, max, P95, % on the red line
%   Same definitions as ftrack/analysis.py.
opts = detectImportOptions(fullfile(runDir, 'log.csv'), 'TextType', 'string');
opts = setvartype(opts, {'phase', 'excite_axis'}, 'string');
T = readtable(fullfile(runDir, 'log.csv'), opts);
meta = jsondecode(fileread(fullfile(runDir, 'meta.json')));
ev = meta.events;
tTake = T.t(1); tLand = T.t(end);
for i = 1:numel(ev)
    if strcmp(ev(i).event, 'phase -> TAKEOFF'), tTake = ev(i).t; end
    if strcmp(ev(i).event, 'phase -> LAND'), tLand = ev(i).t; break; end
end
[~, rn] = fileparts(runDir);
out = struct('run', rn, 'kind', meta.kind, 'sim', meta.sim, 'land_reason', meta.land_reason, 'flight_time_s', tLand - tTake);
P = meta.scene.path_cm; closed = meta.scene.closed; L = meta.scene.length_cm;
trk = T(T.phase == "TRACK", :);
% drone-to-wall distance from the apparent size (pin-hole: apparent/real = L/(L-d))
bw = max(T.box_w, T.box_h); bw = bw(isfinite(bw));
if numel(bw) > 10
    r = median(bw) / meta.rig.drone_size_cm; Lc = meta.rig.camera_distance_cm;
    d = 0; if r > 1, d = Lc * (1 - 1 / r); end
    out.depth = struct('apparent_size_cm', median(bw), 'drone_wall_dist_cm', d, 'scale_to_drone_plane', (Lc - d) / Lc);
end
figs = {};
if height(trk) > 10
    tw = trk;
    if strcmp(meta.kind, 'square')
        cs = cornerS(P, L); s0 = trk.s(1);
        ahead = sort(mod(cs - s0, L)); first = ahead(find(ahead > 30, 1));
        if ~isempty(first)
            prog = mod(trk.s - s0, L);
            stop = find(prog > first - 20 & prog < first + 50, 1);
            if ~isempty(stop) && stop > 10, tw = trk(1:stop - 1, :); end
        end
    end
    m = stepMetrics(tw.t, tw.e_n);
    tt = trk.t - trk.t(1);
    e = trk.e_n; ss = e(tt >= m.steady_from_s & isfinite(e));
    if numel(ss) > 3
        m.ss_mean_cm = mean(ss); m.ss_mean_abs_cm = mean(abs(ss)); m.ss_rms_cm = sqrt(mean(ss .^ 2));
        m.ss_max_cm = max(abs(ss)); m.ss_p95_cm = prctile(abs(ss), 95);
        m.on_line_pct = 100 * mean(abs(ss) <= 10); m.on_white_pct = 100 * mean(abs(ss) <= 2);
    end
    m.track_time_s = tt(end);
    ds = abs(diff(trk.s)); ds(ds > 20 | isnan(ds)) = 0;
    m.mean_speed_cm_s = sum(ds) / max(tt(end), 1e-6);
    m.progress = max(trk.progress);
    if isfield(out, 'depth'), m.ss_rms_drone_plane_cm = m.ss_rms_cm * out.depth.scale_to_drone_plane; end
    out.track = m;
    if strcmp(meta.kind, 'square'), out.corners = cornerMetrics(trk, P, L); end
    figs{end + 1} = figTrack(runDir, trk, P, closed, m, meta);
end
las = T(T.phase == "LASER", :);
if height(las) > 10
    dd = hypot(las.laserX - las.X, las.laserY - las.Y); dd = dd(isfinite(dd));
    if ~isempty(dd)
        out.laser = struct('laser_dist_mean_cm', mean(dd), 'laser_dist_rms_cm', sqrt(mean(dd .^ 2)), ...
            'laser_phase_s', las.t(end) - las.t(1), 'laser_seen_pct', 100 * mean(isfinite(las.laserX)));
    end
    figTheme2(runDir, T, P, meta);
end
if any(T.phase == "EXCITE"), figSysid(runDir, T); end
figOverview(runDir, T, meta);
if strcmp(meta.kind, 'path'), out.race_time_s = out.flight_time_s; end
fid = fopen(fullfile(runDir, 'metrics.json'), 'w'); fprintf(fid, '%s', jsonencode(out, 'PrettyPrint', true)); fclose(fid);
md = metricsMd(out);
fid = fopen(fullfile(runDir, 'metrics.md'), 'w'); fprintf(fid, '%s', md); fclose(fid);
fprintf('%s', md);
end

% ===========================================================================================
function m = stepMetrics(t, e)
ok = isfinite(e); t = t(ok); e = e(ok); t = t - t(1);
band = 5; m = struct('e0_cm', e(1), 'band_cm', band);
e0 = e(1);
if abs(e0) >= 2 * band
    y = 1 - e / e0;
    i10 = find(y >= 0.1, 1); i90 = find(y >= 0.9, 1);
    m.rise_time_s = nan; if ~isempty(i10) && ~isempty(i90), m.rise_time_s = t(i90) - t(i10); end
    cr = find(sign(e) ~= sign(e0), 1);
    if ~isempty(cr)
        w = t >= t(cr) & t <= t(cr) + 6;
        z = -sign(e0) * e; z(~w) = -inf; [mx, k] = max(z);
        m.overshoot_cm = mx; m.overshoot_pct = 100 * mx / abs(e0); m.peak_time_s = t(k);
    else
        m.overshoot_cm = 0; m.overshoot_pct = 0; m.peak_time_s = nan;
    end
end
m.settling_time_s = settle(t, e, band); m.settling_time_3cm_s = settle(t, e, 3);
if isfinite(m.settling_time_s), m.steady_from_s = m.settling_time_s; else, m.steady_from_s = min(0.3 * t(end), 8); end
end

function ts = settle(t, e, b)
out = find(abs(e) > b);
if isempty(out), ts = 0; elseif out(end) == numel(e), ts = nan; else, ts = t(out(end) + 1); end
end

function cs = cornerS(P, L)
tan = diff([P; P(1, :)]); sl = vecnorm(tan, 2, 2); sv = [0; cumsum(sl)]; sv = sv(1:end - 1);
h = atan2(tan(:, 2), tan(:, 1)); dh = abs(mod(h - circshift(h, 1) + pi, 2 * pi) - pi);
cs = mod(sv(dh > deg2rad(45)), L);
end

function C = cornerMetrics(trk, P, L)
C = struct('corner_s_cm', {}, 'max_abs_e_cm', {});
for c = cornerS(P, L)'
    d = mod(trk.s - c + L / 2, L) - L / 2; w = abs(d) < 25;
    if any(w), C(end + 1) = struct('corner_s_cm', c, 'max_abs_e_cm', max(abs(trk.e_n(w)))); end %#ok<AGROW>
end
end

function f = figTrack(runDir, trk, P, closed, m, meta)
f = figure('Visible', 'off', 'Position', [0 0 760 760]); hold on; axis equal; grid on;
tg = gradient(P')'; if closed, tg = diff([P(end, :); P; P(1, :)]); tg = (tg(1:end - 1, :) + tg(2:end, :)) / 2; end
tg = tg ./ vecnorm(tg, 2, 2); nr = [-tg(:, 2), tg(:, 1)];
PP = P; if closed, PP = [P; P(1, :)]; end
for off = [10 -10]
    Q = P + off * nr; if closed, Q = [Q; Q(1, :)]; end
    plot(Q(:, 1), Q(:, 2), '-', 'Color', [0.78 0.06 0.18], 'LineWidth', 1);
end
plot(PP(:, 1), PP(:, 2), '-', 'Color', [0.35 0.35 0.35], 'LineWidth', 2);
tt = trk.t - trk.t(1); tr = tt < m.steady_from_s;
plot(trk.X(tr), trk.Y(tr), '--', 'Color', [1 0.5 0], 'LineWidth', 1.5);
plot(trk.X(~tr), trk.Y(~tr), '-', 'Color', [0.85 0.1 0.1], 'LineWidth', 1.5);
plot(trk.X(1), trk.Y(1), 'ko', 'MarkerFaceColor', 'k');
legend('red line edge +10', 'red line edge -10', 'ideal path (white line)', 'transient', 'steady state', 'start', 'Location', 'best');
xlabel('X (cm)'); ylabel('Y (cm)'); title(sprintf('%s: flight vs ideal path', meta.kind));
exportgraphics(f, fullfile(runDir, 'fig_trajectory.png'), 'Resolution', 130); close(f);
f = figure('Visible', 'off', 'Position', [0 0 1100 650]);
subplot(2, 1, 1); hold on; grid on;
patch([0 tt(end) tt(end) 0], [-5 -5 5 5], [0.6 0.9 0.6], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
plot(tt, trk.e_n, 'b-'); yline(0, 'k-');
if isfield(m, 'settling_time_s') && isfinite(m.settling_time_s), xline(m.settling_time_s, 'g--', sprintf('t_s=%.2f', m.settling_time_s)); end
if isfield(m, 'peak_time_s') && isfinite(m.peak_time_s), xline(m.peak_time_s, 'm--', sprintf('t_p=%.2f', m.peak_time_s)); end
ylabel('cross-track error e_n (cm)');
ttl = sprintf('e0=%.1f cm', m.e0_cm);
for k = {'rise_time_s', 'overshoot_pct', 'ss_mean_cm', 'ss_rms_cm'}
    if isfield(m, k{1}) && isfinite(m.(k{1})), ttl = [ttl sprintf(', %s=%.2f', k{1}, m.(k{1}))]; end %#ok<AGROW>
end
title(ttl, 'Interpreter', 'none');
subplot(2, 1, 2); plot(tt, trk.rc_a, tt, trk.rc_c); grid on; legend('rc a (left/right)', 'rc c (up/down)'); ylabel('rc');
xlabel('time since start of tracking (s)');
exportgraphics(f, fullfile(runDir, 'fig_error_time.png'), 'Resolution', 130); close(f);
end

function figTheme2(runDir, T, P, meta)
f = figure('Visible', 'off', 'Position', [0 0 760 760]); hold on; axis equal; grid on;
plot(P(:, 1), P(:, 2), '-', 'Color', [0.8 0.8 0.8], 'LineWidth', 8);
las = T(T.phase == "LASER", :);
plot(las.laserX, las.laserY, '.', 'Color', [0.1 0.7 0.1]);
plot(T.X, T.Y, 'r-', 'LineWidth', 1.2);
if ~isempty(meta.scene.obstacle_cm), O = meta.scene.obstacle_cm; fill(O(:, 1), O(:, 2), 'k', 'FaceAlpha', 0.7); end
for i = 1:size(meta.scene.blue_cm, 1), b = meta.scene.blue_cm(i, :); rectangle('Position', [b - 10 20 20], 'FaceColor', [0.2 0.5 0.9]); end
legend('track', 'laser dot', 'drone', 'Location', 'best'); xlabel('X (cm)'); ylabel('Y (cm)'); title('theme 2: path tracking + laser guidance');
exportgraphics(f, fullfile(runDir, 'fig_theme2.png'), 'Resolution', 130); close(f);
end

function figSysid(runDir, T)
ex = T(T.phase == "EXCITE", :);
f = figure('Visible', 'off', 'Position', [0 0 1100 600]);
subplot(2, 1, 1); plot(ex.t, ex.rc_a, '.', ex.t, ex.rc_c, '.'); grid on; legend('rc a', 'rc c'); ylabel('rc');
subplot(2, 1, 2); plot(ex.t, ex.X, '.', ex.t, ex.Y, '.'); grid on; legend('X', 'Y'); ylabel('cm'); xlabel('t (s)');
exportgraphics(f, fullfile(runDir, 'fig_sysid_data.png'), 'Resolution', 120); close(f);
end

function figOverview(runDir, T, meta)
f = figure('Visible', 'off', 'Position', [0 0 1100 700]);
subplot(3, 1, 1); plot(T.t, T.X, T.t, T.Y); grid on; ylabel('position (cm)'); legend('X', 'Y');
for i = 1:numel(meta.events)
    s = meta.events(i).event;
    if startsWith(s, 'phase ->'), xline(meta.events(i).t, ':', s(10:end), 'FontSize', 7); end
end
subplot(3, 1, 2); plot(T.t, T.rc_a, T.t, T.rc_c); grid on; ylabel('rc'); legend('a', 'c');
subplot(3, 1, 3); plot(T.t, T.vision_ms, T.t, T.loop_ms); grid on; ylabel('ms'); legend('vision', 'loop'); xlabel('t (s)');
exportgraphics(f, fullfile(runDir, 'fig_overview.png'), 'Resolution', 110); close(f);
end

function s = metricsMd(o)
s = sprintf('# %s\n\n- kind: %s', o.run, o.kind);
if o.sim, s = [s ' (SIM)']; end
s = [s sprintf('\n- landed because: %s\n- flight time (take-off cmd -> land cmd): %.1f s\n', o.land_reason, o.flight_time_s)];
if isfield(o, 'depth')
    d = o.depth;
    s = [s sprintf('- drone apparent size %.1f cm -> ~%.0f cm in front of the wall (wall -> drone plane x%.3f)\n', d.apparent_size_cm, d.drone_wall_dist_cm, d.scale_to_drone_plane)];
end
if isfield(o, 'track')
    m = o.track;
    s = [s sprintf('\n## Path tracking (cross-track error e_n)\n\n| spec | value |\n|---|---|\n')];
    K = {'e0_cm', 'cm'; 'rise_time_s', 's'; 'overshoot_cm', 'cm'; 'overshoot_pct', '%'; 'peak_time_s', 's'; ...
        'settling_time_s', 's (+-5 cm)'; 'settling_time_3cm_s', 's (+-3 cm)'; 'ss_mean_cm', 'cm (steady-state error, signed)'; ...
        'ss_mean_abs_cm', 'cm'; 'ss_rms_cm', 'cm'; 'ss_max_cm', 'cm'; 'ss_p95_cm', 'cm'; 'on_line_pct', '% on red line'; ...
        'on_white_pct', '% within +-2 cm'; 'track_time_s', 's'; 'mean_speed_cm_s', 'cm/s'; 'progress', ''};
    for i = 1:size(K, 1)
        if isfield(m, K{i, 1})
            v = m.(K{i, 1});
            if isfinite(v), s = [s sprintf('| %s | %.2f %s |\n', K{i, 1}, v, K{i, 2})]; else, s = [s sprintf('| %s | n/a |\n', K{i, 1})]; end %#ok<AGROW>
        end
    end
end
if isfield(o, 'corners') && ~isempty(o.corners)
    s = [s sprintf('\ncorners: ')];
    for c = reshape(o.corners, 1, []), s = [s sprintf('s=%.0f: max %.1f cm, ', c.corner_s_cm, c.max_abs_e_cm)]; end %#ok<AGROW>
    s = [s newline];
end
if isfield(o, 'laser')
    s = [s sprintf('\n## Laser guidance\n\n- mean distance laser-drone %.1f cm, laser phase %.1f s, laser seen %.0f %%\n', ...
        o.laser.laser_dist_mean_cm, o.laser.laser_phase_s, o.laser.laser_seen_pct)];
end
if isfield(o, 'race_time_s'), s = [s sprintf('\n**race time: %.1f s**\n', o.race_time_s)]; end
end
