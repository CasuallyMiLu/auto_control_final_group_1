classdef CamSim < handle
    %FT.CAMSIM  renders the simulated drone (real sprite) on a real photo of the board, half resolution.
    %   Advances the SimTello by 1/fps per frame (deterministic, runs faster than real time).
    properties
        full; half; sc; drone; fps = 25; t = 0; laserFn = []; spr; alpha; noise = 2
    end
    methods
        function o = CamSim(bgFile, sc, drone, laserFn, assetsDir)
            o.full = imread(bgFile); o.half = imresize(o.full, 0.5);
            o.sc = sc; o.drone = drone; o.laserFn = laserFn;
            [s, ~, a] = imread(fullfile(assetsDir, 'drone_sprite.png'));
            meta = jsondecode(fileread(fullfile(assetsDir, 'drone_sprite.json')));
            k = meta.cm_per_px / sc.s_px * 0.5;
            o.spr = imresize(s, k);
            o.alpha = imresize(single(a) / 255, k);
        end
        function f = fullFrame(o), f = o.full; end
        function fr = read(o)
            dt = 1 / o.fps;
            o.drone.step(dt); o.t = o.t + dt;
            img = o.half;
            uv = ft.to_px(o.sc, o.drone.p) * 0.5;
            img = paste(img, o.spr, o.alpha, uv);
            if ~isempty(o.laserFn)
                L = o.laserFn(o.t);
                if ~isempty(L)
                    c = ft.to_px(o.sc, L) * 0.5;
                    img = insertShape(img, 'FilledCircle', [c 4], 'Color', [120 255 140], 'Opacity', 1);
                    img = insertShape(img, 'FilledCircle', [c 2], 'Color', [245 255 245], 'Opacity', 1);
                end
            end
            if o.noise > 0
                img = uint8(single(img) + o.noise * randn(size(img), 'single'));
            end
            fr = struct('img', img, 'scale', 0.5, 't', o.t, 'meas', []);
        end
        function release(~), end
    end
end

function img = paste(img, spr, a, c)
[h, w, ~] = size(spr);
x0 = round(c(1) - w / 2); y0 = round(c(2) - h / 2);
xa = max(1, x0); ya = max(1, y0); xb = min(size(img, 2), x0 + w - 1); yb = min(size(img, 1), y0 + h - 1);
if xa > xb || ya > yb, return; end
S = single(spr(ya - y0 + 1:yb - y0 + 1, xa - x0 + 1:xb - x0 + 1, :));
A = a(ya - y0 + 1:yb - y0 + 1, xa - x0 + 1:xb - x0 + 1);
img(ya:yb, xa:xb, :) = uint8(S .* A + single(img(ya:yb, xa:xb, :)) .* (1 - A));
end
