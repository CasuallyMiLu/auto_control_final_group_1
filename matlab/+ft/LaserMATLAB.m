classdef LaserMATLAB < handle
    %FT.LASERMATLAB  green laser dot on the half-resolution frame: very bright, green-dominant, small; inside the
    %   work area only (the dot also flares on the side wall). A saturated white core next to a green halo counts.
    properties
        f = 0.5; mask; maxA; last = []
    end
    methods
        function o = LaserMATLAB(sc, cfg, halfSize)
            [lo, hi] = ft.bounds_cm(sc, cfg.rig.vision.work_margin_cm);
            poly = ft.to_px(sc, [lo(1) lo(2); hi(1) lo(2); hi(1) hi(2); lo(1) hi(2)]) * o.f;
            o.mask = poly2mask(poly(:, 1), poly(:, 2), halfSize(1), halfSize(2));
            o.maxA = (15 / sc.s_px * o.f) ^ 2;
        end
        function uv = step(o, img)
            I = int16(img); R = I(:, :, 1); G = I(:, :, 2); B = I(:, :, 3);
            green = (G - max(R, B) > 40) & G > 170;
            core = G > 245 & R > 200 & B > 200;
            m = (green | (core & imdilate(green, ones(7)))) & o.mask;
            st = regionprops(m, 'Area', 'Centroid', 'PixelIdxList');
            uv = []; best = 0;
            for k = 1:numel(st)
                if st(k).Area < 2 || st(k).Area > o.maxA, continue; end
                ii = st(k).PixelIdxList;
                sc_ = st(k).Area * mean(double(G(ii)) - double(max(R(ii), B(ii))) + 0.3 * double(G(ii)));
                c = st(k).Centroid / o.f;
                if ~isempty(o.last), sc_ = sc_ / (1 + norm(c - o.last) / 400); end
                if sc_ > best, best = sc_; uv = c; end
            end
            if ~isempty(uv), o.last = uv; end
        end
    end
end
