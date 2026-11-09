function M = sysid_tello(runDir, writeConfig)
%SYSID_TELLO  System identification of the Tello wall-plane dynamics (Lab1 method on real flight data).
%   M = sysid_tello('..\runs\<stamp>_sysid_circle')          % identify + figures + model.mat
%   M = sysid_tello('..\runs\<stamp>_sysid_circle', true)    % ... and write ..\config\model.json
%
%   Data  : log.csv of sysid_flight.py, EXCITE phases (open-loop doublets + PRBS on rc a / rc c), camera
%           position X / Y in wall cm. Each block -> one iddata experiment (uniform Ts = 1/30 s, input held).
%   Models: (1) process model 'P1DI'  G(s) = Kp e^{-Td s} / (s (1 + Tp1 s))      (main, = ftrack/sysid.py)
%           (2) tfest, 2 poles, 0 zeros, estimated IO delay                       (Lab1-style comparison)
%   Split : training = PRBS + small-amplitude doublets, validation = large-amplitude doublets (never fitted).
%   Needs System Identification Toolbox + Control System Toolbox.

if nargin < 2, writeConfig = false; end
here = fileparts(mfilename('fullpath'));
opts = detectImportOptions(fullfile(runDir, 'log.csv'), 'TextType', 'string');
opts = setvartype(opts, {'phase', 'excite_axis'}, 'string');   % mostly empty columns: force text
T = readtable(fullfile(runDir, 'log.csv'), opts);
Ts = 1/30;
axesList = ["x", "y"];
M = struct();
fig = figure('Color', 'w', 'Position', [80 80 1200 820]);
tl = tiledlayout(fig, numel(axesList), 2, 'TileSpacing', 'compact');
for k = 1:numel(axesList)
    ax = axesList(k);
    [blocks, amp, isPrbs] = getBlocks(T, ax, Ts);
    if isempty(blocks), continue; end
    amps = unique(amp(~isPrbs));
    big = max(amps);
    tr = isPrbs | (amp ~= big);
    if numel(amps) < 2, tr = true(size(amp)); end
    va = ~tr;
    dTrain = merge(blocks{tr});
    % ---- (1) process model with integrator
    init = idproc('P1DI', 'Kp', 1, 'Tp1', 0.35, 'Td', 0.15);
    init.Structure.Kp.Minimum = 0.05;  init.Structure.Kp.Maximum = 5;
    init.Structure.Tp1.Minimum = 0.02; init.Structure.Tp1.Maximum = 3;
    init.Structure.Td.Minimum = 0;     init.Structure.Td.Maximum = 0.8;
    % disturbance model: hover drift / air flow is low-frequency noise; estimating it (ARMA on the output) keeps
    % it from biasing Kp, Tp1, Td (same role as the drift term in ftrack/sysid.py)
    opt = procestOptions('InitialCondition', 'estimate', 'DisturbanceModel', 'estimate', 'Display', 'off');
    sysP = procest(dTrain, init, opt);
    % ---- (2) Lab1-style transfer function (2 poles, 0 zeros, delay)
    tfo = tfestOptions('InitialCondition', 'estimate', 'Display', 'off');
    sysT = tfest(dTrain, 2, 0, NaN, tfo);
    % ---- fit on training / validation experiments
    [fitTrP, fitVaP] = fits(blocks, tr, va, sysP);
    [fitTrT, fitVaT] = fits(blocks, tr, va, sysT);
    Kv = sysP.Kp; tau = sysP.Tp1; L = sysP.Td;
    fprintf('axis %s: P1DI  Kv = %.3f cm/s per rc, tau = %.3f s, L = %.3f s | fit train %s | valid %s\n', ...
        upper(ax), Kv, tau, L, mat2str(round(fitTrP', 1)), mat2str(round(fitVaP', 1)));
    [nm, dn] = tfdata(sysT, 'v');
    fprintf('        tfest  num %s / den %s, delay %.3f s | fit train %s | valid %s\n', mat2str(nm, 4), mat2str(dn, 4), ...
        sysT.IODelay, mat2str(round(fitTrT', 1)), mat2str(round(fitVaT', 1)));
    M.(ax) = struct('Kv', Kv, 'tau', tau, 'L', L, 'sysP', sysP, 'sysT', sysT, ...
        'fitTrainP', fitTrP, 'fitValidP', fitVaP, 'fitTrainT', fitTrT, 'fitValidT', fitVaT, 'nTrain', nnz(tr), 'nValid', nnz(va));
    % ---- figures: validation comparison + model step/bode
    nexttile(tl);
    vi = find(va, 1); if isempty(vi), vi = 1; end
    compare(blocks{vi}, sysP, sysT);
    title(sprintf('axis %s validation (block %d, amp %g)', upper(ax), vi, amp(vi)));
    nexttile(tl);
    step(tf(Kv, [tau 1 0], 'InputDelay', L), 3); hold on;
    title(sprintf('axis %s: response of P1DI to a unit rc step (cm)', upper(ax)));
end
exportgraphics(fig, fullfile(runDir, 'matlab_sysid.png'), 'Resolution', 140);
save(fullfile(runDir, 'matlab_sysid.mat'), 'M');
if writeConfig && isfield(M, 'x') && isfield(M, 'y')
    J = struct();
    J.x_doc = 'identified by matlab/sysid_tello.m (procest P1DI)';
    J.source = "sysid:" + string(runDir);
    J.B = [M.x.Kv 0; 0 M.y.Kv];
    J.axes.x = struct('Kv', M.x.Kv, 'tau', M.x.tau, 'L', M.x.L);
    J.axes.y = struct('Kv', M.y.Kv, 'tau', M.y.tau, 'L', M.y.L);
    f = fullfile(here, '..', 'config', 'model.json');
    fid = fopen(f, 'w'); fprintf(fid, '%s', jsonencode(J, 'PrettyPrint', true)); fclose(fid);
    fprintf('wrote %s\n', f);
end
end

% ------------------------------------------------------------------------------------------------
function [blocks, amp, isPrbs] = getBlocks(T, ax, Ts)
ph = T.phase; exAx = string(T.excite_axis);
isEx = ph == "EXCITE" & exAx == ax;
t = T.t;
idx = find(isEx);
blocks = {}; amp = []; isPrbs = [];
if isempty(idx), return; end
cuts = [0; find(diff(t(idx)) > 0.3); numel(idx)];
if ax == "x", uCol = T.rc_a; yCol = T.X; else, uCol = T.rc_c; yCol = T.Y; end
for b = 1:numel(cuts) - 1
    ii = idx(cuts(b) + 1:cuts(b + 1));
    t0 = t(ii(1)) - 0.6; t1 = t(ii(end)) + 1.0;   % pre-roll (delay pipe) + tail, same as ftrack/sysid.py
    sel = find(t >= t0 & t <= t1);
    tt = t(sel); u = uCol(sel); y = yCol(sel);
    ok = isfinite(y);
    if nnz(ok) < 10, continue; end
    tg = (tt(1):Ts:tt(end))';
    ug = interp1(tt, u, tg, 'previous', 'extrap');
    yg = interp1(tt(ok), y(ok), tg, 'linear', 'extrap');
    yg = yg - yg(1);
    ue = uCol(ii);
    blocks{end + 1} = iddata(yg, ug, Ts, 'InputName', "rc_" + ax, 'OutputName', upper(ax) + "_cm"); %#ok<AGROW>
    amp(end + 1, 1) = max(abs(ue)); %#ok<AGROW>
    isPrbs(end + 1, 1) = sum(abs(diff(sign(ue))) > 0) > 6; %#ok<AGROW>
end
isPrbs = logical(isPrbs);
end

function [fTr, fVa] = fits(blocks, tr, va, sys)
fTr = []; fVa = [];
o = compareOptions('InitialCondition', 'e');
for i = 1:numel(blocks)
    [~, f] = compare(blocks{i}, sys, o);
    if tr(i), fTr(end + 1, 1) = f; else, fVa(end + 1, 1) = f; end %#ok<AGROW>
end
end
