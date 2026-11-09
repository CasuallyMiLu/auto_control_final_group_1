classdef VisionBSM < handle
    %FT.VISIONBSM  drone position by background subtraction (MATLAB, Image Processing Toolbox).
    %   Works on the half-resolution frame. Background = median of the first frames (before take-off), then a slow
    %   running average outside the drone. Drone = blob 'darker than the background in any colour channel' (over
    %   the red band a black drone is darker in R although its grey level is similar), size gated, inside the work
    %   area. A constant-velocity Kalman filter picks / smooths the detection (gate 150 px).
    properties
        f = 0.5; mask; thr = 45; warm = 25; stack = {}; bg = []
        minA; maxA
        kx = []; kP = []; kt = 0; miss = 0; q = 2e7; r = 9; gate = 150; coast = 3
    end
    methods
        function o = VisionBSM(sc, cfg, halfSize)
            [lo, hi] = ft.bounds_cm(sc, cfg.rig.vision.work_margin_cm);
            poly = ft.to_px(sc, [lo(1) lo(2); hi(1) lo(2); hi(1) hi(2); lo(1) hi(2)]) * o.f;
            o.mask = poly2mask(poly(:, 1), poly(:, 2), halfSize(1), halfSize(2));
            cmpp = sc.s_px / o.f;
            o.minA = (4 / cmpp) ^ 2; o.maxA = (45 / cmpp) ^ 2;
        end
        function out = step(o, img, t)
            tic0 = tic;
            I = imgaussfilt(img, 0.8);
            dets = zeros(0, 7);   % [cx cy x0 y0 x1 y1 conf] in FULL-resolution px
            if isempty(o.bg)
                o.stack{end + 1} = I;
                if numel(o.stack) >= o.warm
                    o.bg = single(median(cat(4, o.stack{:}), 4)); o.stack = {};
                end
            else
                d = max(o.bg - single(I), [], 3);        % darker than background, any channel
                m = d > o.thr & o.mask;
                m = imclose(m, ones(5));
                st = regionprops(m, 'Area', 'Centroid', 'BoundingBox');
                st = st([st.Area] >= o.minA & [st.Area] <= o.maxA);
                for k = 1:numel(st)
                    b = st(k).BoundingBox; c = st(k).Centroid;
                    dets(end + 1, :) = [c / o.f, [b(1) b(2) b(1) + b(3) b(2) + b(4)] / o.f, min(1, st(k).Area / (o.minA * 6))]; %#ok<AGROW>
                end
                upd = true(size(m));
                for k = 1:size(dets, 1)
                    bb = round(dets(k, 3:6) * o.f);
                    upd(max(1, bb(2) - 10):min(end, bb(4) + 10), max(1, bb(1) - 10):min(end, bb(3) + 10)) = false;
                end
                a = 0.02;
                bgu = o.bg; Is = single(I);
                upd3 = repmat(upd, 1, 1, 3);
                bgu(upd3) = (1 - a) * bgu(upd3) + a * Is(upd3);
                o.bg = bgu;
            end
            out = o.track(dets, t);
            out.ms = toc(tic0) * 1e3;
        end
        function out = track(o, dets, t)
            out = struct('uv', [], 'box', [], 'conf', 0, 'vel', [0 0]);
            pred = [];
            if ~isempty(o.kx)
                dt = max(1e-3, t - o.kt);
                F = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1]; G = [dt ^ 2 / 2 0; 0 dt ^ 2 / 2; dt 0; 0 dt];
                o.kx = F * o.kx; o.kP = F * o.kP * F' + o.q * (G * G'); o.kt = t; pred = o.kx(1:2)';
            end
            ch = [];
            if ~isempty(dets)
                if ~isempty(pred) && o.miss <= o.coast
                    dd = vecnorm(dets(:, 1:2) - pred, 2, 2);
                    near = find(dd <= o.gate * (1 + 0.5 * o.miss));
                    if ~isempty(near), [~, j] = max(dets(near, 7)); ch = near(j);
                    else, far = find(dets(:, 7) >= 0.5); if ~isempty(far), [~, j] = max(dets(far, 7)); ch = far(j); end
                    end
                else
                    [~, ch] = max(dets(:, 7));
                end
            end
            if isempty(ch)
                if ~isempty(o.kx)
                    o.miss = o.miss + 1;
                    if o.miss <= o.coast, out.uv = o.kx(1:2)'; out.vel = o.kx(3:4)'; return; end
                    o.kx = [];
                end
                return
            end
            z = dets(ch, 1:2)';
            if isempty(o.kx) || o.miss > o.coast || norm(z' - pred) > 3 * o.gate
                o.kx = [z; 0; 0]; o.kP = diag([o.r o.r 1e4 1e4]); o.kt = t;
            else
                H = [1 0 0 0; 0 1 0 0]; R = o.r / max(dets(ch, 7), 0.1) * eye(2);
                K = o.kP * H' / (H * o.kP * H' + R);
                o.kx = o.kx + K * (z - H * o.kx); o.kP = (eye(4) - K * H) * o.kP;
            end
            o.miss = 0;
            out.uv = o.kx(1:2)'; out.vel = o.kx(3:4)'; out.box = dets(ch, 3:6); out.conf = dets(ch, 7);
        end
    end
end
