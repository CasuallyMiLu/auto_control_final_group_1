function runDir = run_mission(mode, kind, opt)
%FT.RUN_MISSION  one program from start-up to landing (MATLAB main program).
%   mode 'theme1' | 'theme2' | 'sysid', kind 'circle' | 'square' | 'path'.
%   calibrate scene on the live frame -> preview (G = go) -> connect + battery -> take-off -> find the drone ->
%   orientation probe (short climb + sideways nudge) -> phases -> land -> log.csv, meta.json, analysis.
%   Every frame: vision -> wall position -> controller -> rc -> safety -> log -> dashboard.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));   % .../final
cfg = ft.load_cfg(root, opt);
rig = cfg.rig; G = cfg.gains; sf = rig.safety;
stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
name = mode; if ~strcmp(mode, 'theme2'), name = [mode '_' kind]; end
tagS = ''; if opt.sim, tagS = '_sim'; end
if ~isempty(opt.tag), tagS = [tagS '_' opt.tag]; end
runDir = fullfile(root, 'runs', [stamp '_' name '_matlab' tagS]);
mkdir(runDir);
events = struct('t', {}, 'event', {});
msg = ''; t = 0; phase = 'INIT'; phaseT0 = 0;
titles = struct('theme1', ['Theme 1 - ' kind ' tracking (MATLAB)'], 'theme2', 'Theme 2 - path + laser race (MATLAB)', ...
    'sysid', 'System identification flight (MATLAB)');

% ------------------------------------------------------------------------------------------- shared state
logF = -1; dash = []; tello = []; src = []; sc = []; landReason = ''; probeInfo = struct(); act = [];
fol = []; pt = []; stageXY = []; landXY = []; landR = 15; centre = []; prog = []; queue = cell(0, 2); cur = {};
nearSince = []; trackT0 = 0; holdXY = []; laserT0 = 0; laserLast = []; laserLastT = -1e9; inZoneSince = []; blkT0 = 0;
probeP0 = []; probeDir = []; probeLast = []; flightT0 = []; vis = []; las = []; lo = []; hi = [];
p = []; vEst = [0 0]; seen = false; lastSeen = -1e9; frameI = 0; trackErr = []; paused = false;
o = struct('uv', [], 'box', [], 'conf', 0, 'vel', [0 0], 'ms', 0); fr = []; full = [];
guard = ft.Guard();
cleanup = onCleanup(@() guard.fire()); %#ok<NASGU>
try
% ------------------------------------------------------------------------------------------- sources
if opt.sim
    bgFile = fullfile(root, 'assets', ['background_' kind '.jpg']);
    scTrue = ft.calibrate(imread(bgFile), kind, cfg, 'auto');
    simModel = cfg.model;
    if ~isempty(opt.simModel), simModel = jsondecode(fileread(opt.simModel)); end
    if opt.simFlipA, simModel.B(:, 1) = -simModel.B(:, 1); end
    drone = ft.SimTello(simModel, simStart(scTrue), opt.simWind);
    laserFn = [];
    if strcmp(mode, 'theme2'), laserFn = simLaser(scTrue, drone); end
    src = ft.CamSim(bgFile, scTrue, drone, laserFn, fullfile(root, 'assets'));
    tello = drone; guard.tello = drone;
else
    cam = rig.camera;
    useWebcam = strcmp(opt.camera, 'webcam') || (strcmp(opt.camera, 'auto') && ~strcmp(opt.vision, 'yolo') && exist('webcamlist', 'file') == 2);
    if useWebcam
        src = ft.CamWebcam(cam.index, cam.width, cam.height, kind, cfg, fullfile(runDir, 'camera_half.mp4'));
    else
        src = ft.CamPython(root, cam.index, cam.width, cam.height, cam.fps, fullfile(runDir, 'camera_raw.mp4'));
    end
end

% ------------------------------------------------------------------------------------------- calibration
upHint = opt.upHint; if isempty(upHint), upHint = rig.up_hint; end
lastErr = '';
for attempt = 1:30
    full = src.fullFrame();
    try
        sc = ft.calibrate(full, kind, cfg, upHint); break
    catch e
        lastErr = e.message; pause(0.3);
    end
end
if isempty(sc), error('scene calibration failed: %s', lastErr); end
imwrite(full, fullfile(runDir, 'scene_frame.jpg'));
logEvent(sprintf('scene %s: %.4f cm/px, path %.0f cm, up=[%.2f %.2f] (%s)', kind, sc.s_px, sc.path.L, sc.up, sc.info.up_source));
[lo, hi] = ft.bounds_cm(sc, sf.bounds_margin_cm);

% ------------------------------------------------------------------------------------------- vision
visionKind = opt.vision;
if strcmp(visionKind, 'auto'), if isa(src, 'ft.CamPython'), visionKind = 'yolo'; else, visionKind = 'bsm'; end, end
if strcmp(visionKind, 'yolo') && ~isa(src, 'ft.CamPython')
    logEvent('YOLO needs the Python camera; using MATLAB background subtraction'); visionKind = 'bsm';
end
fr0 = src.read();
if strcmp(visionKind, 'yolo')
    src.startYolo(sc, cfg, root, strcmp(mode, 'theme2'));
    vis = []; las = [];
else
    vis = ft.VisionBSM(sc, cfg, [size(fr0.img, 1) size(fr0.img, 2)]);
    las = []; if strcmp(mode, 'theme2'), las = ft.LaserMATLAB(sc, cfg, [size(fr0.img, 1) size(fr0.img, 2)]); end
end
logEvent(['vision: ' visionKind]);

% ------------------------------------------------------------------------------------------- controllers
act = ft.Actuator(cfg.model.B, G.actuator.rc_max, G.actuator.accel_max);
build();
cols = {'t', 'frame', 'phase', 'valid', 'u', 'v', 'X', 'Y', 'vX', 'vY', 's', 'e_n', 'refX', 'refY', 'v_ff', 'vdesX', 'vdesY', ...
    'rc_a', 'rc_b', 'rc_c', 'rc_d', 'laser_u', 'laser_v', 'laserX', 'laserY', 'box_w', 'box_h', 'conf', 'vision_ms', 'loop_ms', ...
    'bat', 'h', 'tof', 'vgx', 'vgy', 'vgz', 'pitch', 'roll', 'yaw', 'progress', 'excite_axis', 'excite_u'};
logF = fopen(fullfile(runDir, 'log.csv'), 'w');
fprintf(logF, '%s\n', strjoin(cols, ','));
rec = ''; if opt.record, rec = fullfile(runDir, 'dashboard.mp4'); end
dash = ft.Dashboard(sc, titles.(mode), opt.show, rec);

% ------------------------------------------------------------------------------------------- preview
fr = fr0;
if ~opt.yes && opt.show
    msg = 'check green path, start/direction; G = go, Q = quit';
    while true
        fr = src.read(); t = fr.t; o = see(fr);
        dash.render(fr, dstate(struct()));
        k = dash.takeKey();
        if strcmp(k, 'go'), break; end
        if strcmp(k, 'land') || ~isgraphics(dash.fig), logEvent('aborted at preview'); shutdown(); return; end
    end
end

% ------------------------------------------------------------------------------------------- connect + take-off
if ~opt.sim
    tello = ft.TelloUDP(rig.tello.ip); guard.tello = tello;
    if ~tello.connect()
        error('Tello did not answer ''command'' (Wi-Fi TELLO-xxxx? firewall allows MATLAB UDP 8889/8890? ryze object closed?)');
    end
    pause(0.5);
    bat = tello.battery();
    logEvent(sprintf('Tello connected, battery %.0f%%', bat));
    if bat < rig.tello.min_battery, error('battery %.0f%% < %d%%: change the battery', bat, rig.tello.min_battery); end
end
setPhase('TAKEOFF'); flightT0 = t;
tello.startCmd('takeoff');
acq = 0; tPrev = []; rateT = [];
landPending = false;

% ------------------------------------------------------------------------------------------- main loop
while true
    fr = src.read();
    if isnan(fr.t) || fr.t < 0, continue; end
    tl0 = tic; t = fr.t;
    if isempty(tPrev), dt = 1 / 30; else, dt = max(1e-3, t - tPrev); end
    tPrev = t; frameI = frameI + 1;
    o = see(fr);
    if ismethod(tello, 'pollState'), tello.pollState(); end
    lostFor = t - lastSeen;
    vDes = []; rc = [0 0 0 0]; info = struct();
    switch phase
        case 'TAKEOFF'
            rc = [];
            if tello.pollReply(sf.takeoff_timeout_s), setPhase('ACQUIRE'); end
        case 'ACQUIRE'
            if seen && inside(p), acq = acq + 1; else, acq = 0; end
            if acq >= 5
                if strcmp(opt.probe, 'off'), setPhase(firstPhase()); else, setPhase('PROBE_UP'); end
            elseif t - phaseT0 > sf.acquire_timeout_s
                land('drone not found in the image after take-off'); rc = [];
            end
        case {'PROBE_UP', 'PROBE_SIDE'}
            rc = probeStep();
        case {'LAND', 'DONE'}
            rc = [];
        otherwise
            if paused
                rc = [0 0 0 0]; msg = 'PAUSED (p resume, SPACE land)';
            elseif lostFor > sf.lost_land_s
                land(sprintf('drone lost for %.1f s', lostFor)); rc = [];
            elseif lostFor > sf.lost_hover_s || isempty(p)
                rc = [0 0 0 0]; act.reset(); msg = sprintf('drone not visible (%.1f s): hovering', lostFor);
            elseif ~inside(p)
                land(sprintf('left the safe area at X=%.0f, Y=%.0f cm', p)); rc = [];
            else
                pp = p + vEst * G.predict_s;
                [vDes, info] = control(pp, dt);
                if isfield(info, 'rc'), rc = info.rc;
                elseif ~isempty(vDes), [rc, vDes] = act.step(vDes, dt);
                end
            end
    end
    if ~isempty(flightT0) && ~any(strcmp(phase, {'LAND', 'DONE'})) && t - flightT0 > sf.max_flight_s
        land('max flight time'); rc = [];
    end
    if ~isempty(rc) && ~any(strcmp(phase, {'TAKEOFF', 'LAND', 'DONE'}))
        tello.rc(rc(1), rc(2), rc(3), rc(4));
    end
    if isempty(rc), rcSent = [0 0 0 0]; else, rcSent = rc; end
    rateT = [rateT(max(1, end - 28):end), t]; %#ok<AGROW>
    hz = 0; if numel(rateT) > 2, hz = (numel(rateT) - 1) / max(rateT(end) - rateT(1), 1e-6); end
    writeLog(info, vDes, rcSent, toc(tl0) * 1e3);
    if strcmp(phase, 'TRACK') && isfield(info, 'e_n'), trackErr(end + 1) = info.e_n; end %#ok<AGROW>
    if opt.show || opt.record, dash.render(fr, dstate(info, rcSent, hz)); end
    k = dash.takeKey();
    switch k
        case 'land', land('key');
        case 'emergency', logEvent('EMERGENCY STOP (ESC)'); tello.emergency(); setPhase('DONE');
        case 'pause', paused = ~paused; if paused, logEvent('pause'); else, logEvent('resume'); end
        case 'landkey', land('L key');
    end
    if opt.show && ~isgraphics(dash.fig) && ~any(strcmp(phase, {'LAND', 'DONE'})), land('window closed'); end
    if strcmp(phase, 'LAND') && tello.pollReply(8), setPhase('DONE'); end
    if strcmp(phase, 'DONE') && t - phaseT0 > 1.0, break; end
end
catch err
    fprintf(2, 'ERROR %s\n', err.message);
    shutdown();
    rethrow(err);
end
shutdown();
return

% =========================================================================================== nested functions
    function logEvent(s)
        fprintf('[%7.2f] %s\n', t, s);
        events(end + 1) = struct('t', round(t, 3), 'event', s);
        msg = s;
    end

    function setPhase(ph)
        phase = ph; phaseT0 = t;
        logEvent(['phase -> ' ph]);
        onPhase(ph);
    end

    function r = inside(x), r = all(x >= lo) && all(x <= hi); end

    function out = see(fr)
        if ~isempty(fr.meas), out = fr.meas; else, out = vis.step(fr.img, fr.t); end
        if ~isempty(out.uv)
            p = ft.to_cm(sc, out.uv); vEst = out.vel * sc.A'; lastSeen = fr.t; seen = true;
        else
            seen = false;
        end
    end

    function land(reason)
        if any(strcmp(phase, {'LAND', 'DONE'})), return; end
        landReason = reason;
        logEvent(['LAND: ' reason]);
        setPhase('LAND');
        tello.rc(0, 0, 0, 0);
        tello.startCmd('land');
    end

    % ------------------------------------------------------------------------- per-mode controllers
    function build()
        switch mode
            case 'theme1'
                fol = ft.PathFollower(sc.path, G, rig.closed_path.laps);
                pt = ft.PointTracker(G, 'point');
                P = sc.path.pts; c = 0.5 * (min(P) + max(P)); d = P(1, :) - c;
                stageXY = P(1, :) + opt.stage_cm * d / (norm(d) + 1e-9);
            case 'theme2'
                fol = ft.PathFollower(sc.path, G, 1);
                pt = ft.PointTracker(G, 'point');
                landXY = landingPoint(sc) + rig.theme2.land_offset_cm(:)';
                landR = rig.theme2.land_radius_cm;
            case 'sysid'
                pt = ft.PointTracker(G, 'point');
                P = sc.path.pts; centre = 0.5 * (min(P) + max(P)) + opt.holdOffset;
                if isempty(prog), prog = makeProgram(); end
        end
    end

    function ph = firstPhase()
        switch mode
            case 'theme1', if opt.stage_cm > 0, ph = 'STAGE'; else, ph = 'TRACK'; end
            case 'theme2', ph = 'TRACK';
            case 'sysid', ph = 'GOTO';
        end
    end

    function onPhase(ph)
        switch ph
            case {'STAGE', 'GOTO'}, pt.reset(); act.reset(); nearSince = [];
            case 'TRACK', fol.reset(); act.reset(); trackT0 = t;
            case 'LASER', pt.reset(); holdXY = p; laserT0 = t; laserLast = []; laserLastT = -1e9; inZoneSince = [];
            case 'EXCITE', blkT0 = t;
            case {'PROBE_UP', 'PROBE_SIDE'}, probeP0 = []; probeDir = [];
        end
    end

    function [v, info] = control(x, dt)
        v = [0 0]; info = struct();
        switch phase
            case {'STAGE', 'GOTO'}
                if strcmp(phase, 'STAGE'), tgt = stageXY; else, tgt = centre; end
                [v, ti] = pt.step(x, tgt, dt);
                info.ref = tgt; info.exy = tgt - x;
                ok = norm(x - tgt) < 5 + 1 * strcmp(phase, 'GOTO');   % position only: the velocity estimate is noisy
                if ok, if isempty(nearSince), nearSince = t; end, else, nearSince = []; end
                hold_ = 0.8 + 0.2 * strcmp(phase, 'GOTO');
                if strcmp(phase, 'STAGE')
                    if (~isempty(nearSince) && t - nearSince > hold_) || t - phaseT0 > 12, setPhase('TRACK'); end
                else
                    if ~isempty(nearSince) && t - nearSince > hold_
                        if isempty(queue), land('identification program finished');
                        else, cur = queue(1, :); queue(1, :) = []; setPhase('EXCITE'); end
                    elseif t - phaseT0 > 15
                        land('GOTO timeout');
                    end
                end
                if ~isempty(ti), end
            case 'TRACK'
                [v, info] = fol.step(x, dt);
                info.exy = info.ref - x;
                if fol.done
                    if strcmp(mode, 'theme1')
                        logEvent(sprintf('lap complete in %.1f s', t - trackT0)); land('lap complete');
                    else
                        logEvent(sprintf('reached B (path done) after %.1f s', t - trackT0)); setPhase('LASER');
                    end
                end
            case 'LASER'
                L = getf(o, 'laser');
                if isempty(fr.meas) && ~isempty(las), L = las.step(fr.img); end
                if ~isempty(L), laserLast = ft.to_cm(sc, L); laserLastT = t; info.laser = L; end
                if ~isempty(laserLast) && t - laserLastT <= rig.theme2.laser_lost_hold_s
                    tgt = laserLast; holdXY = [];
                else
                    if isempty(holdXY), holdXY = x; end
                    tgt = holdXY;
                end
                [v, ti] = pt.step(x, tgt, dt);
                info.ref = ti.ref; info.exy = ti.e;
                % landing trigger (rig.theme2.land_trigger): 'zone' = drone over the lower blue square (C);
                % 'laser_off' = operator switches the laser off (after using it); 'key' = only the L key
                trig = rig.theme2.land_trigger;
                if strcmp(trig, 'zone')
                    if norm(x - landXY) < landR
                        if isempty(inZoneSince), inZoneSince = t; end
                        if t - inZoneSince >= rig.theme2.land_hold_s
                            logEvent(sprintf('landing zone reached (laser phase %.1f s)', t - laserT0)); land('landing zone');
                        end
                    else
                        inZoneSince = [];
                    end
                elseif strcmp(trig, 'laser_off')
                    if ~isempty(laserLast) && t - laserLastT > rig.theme2.laser_off_s
                        logEvent(sprintf('laser switched off -> land (laser phase %.1f s)', t - laserT0)); land('laser off');
                    end
                end
            case 'EXCITE'
                ax = cur{1}; blk = prog.(ax){cur{2}};
                tb = t - blkT0; acc = 0; u = [];
                for j = 1:size(blk, 1)
                    acc = acc + blk(j, 1);
                    if tb < acc, u = blk(j, 2); break; end
                end
                dev = x - centre;
                if isempty(u) || max(abs(dev)) > opt.box
                    if ~isempty(u), logEvent(sprintf('block %s%d stopped: %.0f cm from the centre', ax, cur{2}, max(abs(dev)))); end
                    setPhase('GOTO'); info.rc = [0 0 0 0]; v = []; return
                end
                if strcmp(ax, 'x'), info.rc = [u 0 0 0]; else, info.rc = [0 0 u 0]; end
                info.excite_axis = ax; info.excite_u = u; info.ref = centre; v = [];
        end
    end

    function P = makeProgram()
        rs = RandStream('mt19937ar', 'Seed', opt.seed);
        P = struct();
        for axc = opt.axes
            blocks = {}; sgn = 1;
            for A = opt.amps
                for T = opt.durs
                    blocks{end + 1} = [T sgn * A; 1.2 0; T -sgn * A; 1.2 0]; sgn = -sgn; %#ok<AGROW>
                end
            end
            seq = zeros(0, 2); tt = 0; u = 30;
            while tt < 7
                d = 0.35 * randi(rs, 3); seq(end + 1, :) = [d u]; %#ok<AGROW>
                if rand(rs) < 0.7, u = -u; end
                tt = tt + d;
            end
            blocks{end + 1} = [seq; 1.2 0]; %#ok<AGROW>
            P.(axc{1}) = blocks;
            for j = 1:numel(blocks), queue(end + 1, :) = {axc{1}, j}; end %#ok<AGROW>
        end
    end

    % ------------------------------------------------------------------------- orientation probe
    function rc = probeStep()
        Pw = 40; T = 0.7; still = 0.6; tt = t - phaseT0;
        if isempty(probeP0) && seen, probeP0 = {o.uv, p}; end
        rc = [0 0 0 0];
        if strcmp(phase, 'PROBE_UP')
            if tt < T, rc = [0 0 Pw 0]; return; end
            if tt < T + still, return; end
            if seen && ~isempty(probeP0)
                duv = o.uv - probeP0{1};
                if norm(duv) > 8 / sc.s_px
                    upm = duv / norm(duv);
                    ang = acosd(max(-1, min(1, dot(upm, sc.up))));
                    probeInfo.up_measured_img = upm; probeInfo.up_angle_err_deg = ang;
                    if ang > 45
                        [sc, chg] = ft.set_up_vector(sc, upm);
                        if sc.path.closed
                            sa = rig.closed_path.start_angle_deg; if isstruct(sa), sa = sa.(kind); end
                            sc = ft.orient_closed(sc, rig.closed_path.direction, sa);
                        end
                        logEvent(sprintf('probe: up re-oriented by %d deg', chg));
                        [lo, hi] = ft.bounds_cm(sc, sf.bounds_margin_cm);
                        p = ft.to_cm(sc, o.uv); dash.setScene(sc); build();
                    else
                        logEvent(sprintf('probe: up confirmed (%.0f deg off)', ang));
                    end
                else
                    logEvent('probe: climb too small to judge, keeping the calibrated orientation');
                end
            end
            setPhase('PROBE_SIDE'); return
        end
        % PROBE_SIDE: push towards the track centre first (stays in view), then back
        if isempty(probeDir)
            P = sc.path.pts; cx = 0.5 * (min(P(:, 1)) + max(P(:, 1)));
            if ~isempty(p) && p(1) > cx, probeDir = -1; else, probeDir = 1; end
        end
        if tt < T, rc = [probeDir * Pw 0 0 0]; return; end
        if tt < T + still
            if seen, probeLast = p; end
            if tt < 0.4 + 0.05, probeLast = []; end
            return
        end
        if tt < 2 * T + still, rc = [-probeDir * Pw 0 0 0]; return; end
        if tt < 2 * T + 2 * still, return; end
        if isempty(probeP0)
            logEvent('probe: drone not seen at the start, keeping rc a sign');
        elseif isempty(probeLast)
            logEvent('probe: drone left the view during the inward push -> rc a reversed'); act.flipA(); probeInfo.side = 'lost->flipped';
        else
            d = (probeLast(1) - probeP0{2}(1)) * probeDir;
            probeInfo.side_dx_cm = d;
            if abs(d) <= 4, logEvent('probe: sideways motion too small to judge, keeping rc a sign');
            elseif d < 0, act.flipA(); logEvent('probe: rc a moves the drone the other way (drone faces the camera): sign flipped');
            else, logEvent('probe: rc a sign confirmed');
            end
        end
        setPhase(firstPhase());
    end

    % ------------------------------------------------------------------------- log / dashboard / shutdown
    function writeLog(info, vDes, rcS, loopMs)
        g = @(f) getf(info, f);
        rcS = round(rcS);
        uv = nanv(o.uv, 2); xy = [nan nan]; if seen, xy = p; end
        ref = nanv(g('ref'), 2); vd = nanv(vDes, 2);
        L = nanv(g('laser'), 2); Lxy = [nan nan]; if all(isfinite(L)), Lxy = ft.to_cm(sc, L); end
        bw = [nan nan]; if ~isempty(o.box), bw = [o.box(3) - o.box(1), o.box(4) - o.box(2)] * sc.s_px; end
        st = tello.state; sv = @(f) getf(st, f);
        ax = g('excite_axis'); if isempty(ax), ax = ''; end
        fprintf(logF, '%.4f,%d,%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,%.2f,%.2f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.4f,%s,%.1f\n', ...
            t, frameI, phase, seen, uv, xy, vEst, num(g('s')), num(g('e_n')), ref, num(g('v_ff')), vd, rcS, L, Lxy, bw, ...
            o.conf, o.ms, loopMs, num(sv('bat')), num(sv('h')), num(sv('tof')), num(sv('vgx')), num(sv('vgy')), num(sv('vgz')), ...
            num(sv('pitch')), num(sv('roll')), num(sv('yaw')), num(g('progress')), ax, num(g('excite_u')));
    end

    function st = dstate(info, rcS, hz)
        if nargin < 2, rcS = [0 0 0 0]; hz = 0; end
        e = trackErr; rmsE = nan; if ~isempty(e), rmsE = sqrt(mean(e .^ 2)); end
        xy = [nan nan]; if seen, xy = p; end
        exy = nanv(getf(info, 'exy'), 2);
        zone = []; if strcmp(mode, 'theme2') && strcmp(rig.theme2.land_trigger, 'zone'), zone = [landXY landR]; end
        bat = nan; if ~isempty(tello), bat = num(getf(tello.state, 'bat')); end
        st = struct('t', t, 'phase', phase, 'title', titles.(mode), 'uv', o.uv, 'box', o.box, 'ref', getf(info, 'ref'), ...
            'laser', getf(info, 'laser'), 'zone', zone, 'e_n', num(getf(info, 'e_n')), 'exy', exy, 'rc', rcS, 'xy', xy, ...
            'rms', rmsE, 'progress', num0(getf(info, 'progress')), 'bat', bat, 'hz', hz, 'vms', o.ms, 'msg', msg);
    end

    function shutdown()
        if guard.done, return; end
        try
            if ~isempty(tello) && tello.flying && ~strcmp(phase, 'DONE')
                logEvent('shutdown: landing'); tello.land();
            end
        catch err, disp(err.message);
        end
        if logF > 0, fclose(logF); logF = -1; end
        if ~isempty(dash), dash.close(); end
        if ~isempty(src), src.release(); end
        if ~isempty(tello), tello.close(); end
        guard.done = true;
        if ~isempty(sc) && exist(runDir, 'dir')
            meta = struct('name', name, 'kind', kind, 'sim', opt.sim, 'program', 'matlab', 'opt', opt, 'rig', rig, ...
                'model', cfg.model, 'gains', G, 'events', events, 'land_reason', landReason, 'probe', probeInfo, 'B_used', []);
            if ~isempty(act), meta.B_used = act.B; end
            meta.scene = struct('cm_per_px', sc.s_px, 'up_img', sc.up, 'origin_px', sc.origin, 'A', sc.A, ...
                'path_cm', round(sc.path.pts, 2), 'closed', sc.path.closed, 'length_cm', sc.path.L, ...
                'blue_cm', ft.to_cm(sc, sc.blue_px), 'obstacle_cm', [], 'info', sc.info);
            if ~isempty(sc.obstacle_px), meta.scene.obstacle_cm = ft.to_cm(sc, sc.obstacle_px); end
            fid = fopen(fullfile(runDir, 'meta.json'), 'w'); fprintf(fid, '%s', jsonencode(meta, 'PrettyPrint', true)); fclose(fid);
            try
                ft.analyze_run(runDir);
            catch err
                fprintf('analysis failed: %s\n', err.message);
            end
            fprintf('\nrun saved in %s\n', runDir);
        end
    end

    % ------------------------------------------------------------------------- simulation helpers
    function s0 = simStart(scT)
        P = scT.path.pts;
        switch mode
            case 'theme2', s0 = [P(1, 1), P(1, 2) - 80];
            otherwise, s0 = [0.8 * max(P(:, 1)), min(P(:, 2)) - 60];
        end
    end

    function f = simLaser(scT, dr)
        on = false; kk = 1; wp = [];
        f = @laserAt;
        function L = laserAt(~)
            L = [];
            B = scT.path.pts(end, :); d = dr.p;
            if isempty(wp)
                lp = landingPoint(scT);
                if ~isempty(scT.obstacle_px)
                    O = ft.to_cm(scT, scT.obstacle_px); xmin = min(O(:, 1)); ymin = min(O(:, 2)); ymax = max(O(:, 2));
                else
                    xmin = B(1) - 60; ymin = B(2) - 60; ymax = B(2) - 30;
                end
                wp = [xmin - 30, max(B(2), ymax + 15); xmin - 30, ymin - 25; lp];
            end
            if ~on
                if norm(d - B) < 25, on = true; else, return; end
            end
            if kk < size(wp, 1) && norm(d - wp(kk, :)) < 12, kk = kk + 1; end
            w = wp(kk, :); vv = w - d;
            if norm(vv) < 25, L = w; else, L = d + vv / norm(vv) * 25; end
        end
    end
end

% =========================================================================================== local helpers
function lp = landingPoint(sc)
B = sc.path.pts(end, :);
if isempty(sc.blue_px), lp = sc.path.pts(1, :) + [-60 0]; return; end
C = ft.to_cm(sc, sc.blue_px);
[~, i] = max(vecnorm(C - B, 2, 2));
lp = C(i, :);
end

function v = getf(s, f)
if isstruct(s) && isfield(s, f), v = s.(f); else, v = []; end
end

function x = num(v)
if isempty(v), x = nan; else, x = double(v(1)); end
end

function x = num0(v)
if isempty(v), x = 0; else, x = double(v(1)); end
end

function v = nanv(v, n)
if isempty(v), v = nan(1, n); end
v = double(v(:)');
end
