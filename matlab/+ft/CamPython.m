classdef CamPython < handle
    %FT.CAMPYTHON  camera through Python/OpenCV (ftrack/matlab_bridge.py). mode 'frame': MATLAB gets the
    %   half-resolution frame and runs its own vision; mode 'yolo': Python runs YOLO26 (+ laser) and MATLAB gets the
    %   measurement vector plus a quarter-resolution preview.
    properties
        br; mode = 'frame'; wantLaser = false; index
    end
    methods
        function o = CamPython(root, index, w, h, fps, recordFile)
            ft.py_setup(root);
            o.br = py.importlib.import_module('ftrack.matlab_bridge');
            r = cell(o.br.open_camera(index, int32(w), int32(h), int32(fps), recordFile));
            o.index = double(r{1});
            fprintf('[camera] python/OpenCV camera %d: %dx%d\n', o.index, double(r{2}), double(r{3}));
        end
        function f = fullFrame(o), f = uint8(o.br.full_frame()); end
        function startYolo(o, sc, cfg, root, wantLaser)
            [lo, hi] = ft.bounds_cm(sc, cfg.rig.vision.work_margin_cm);
            poly = ft.to_px(sc, [lo(1) lo(2); hi(1) lo(2); hi(1) hi(2); lo(1) hi(2)]);
            md = cfg.rig.vision.model_dir;
            kind = char(o.br.setup_vision(py.numpy.array(poly(:)'), sc.s_px, 'yolo', md, wantLaser));
            o.mode = 'yolo'; o.wantLaser = wantLaser;
            fprintf('[vision] python detector: %s\n', kind);
        end
        function fr = read(o)
            if strcmp(o.mode, 'yolo')
                r = cell(o.br.detect(0.25, o.wantLaser));
                v = double(r{1}); img = uint8(r{2});
                m = struct('uv', [], 'box', [], 'conf', v(5), 'vel', v(10:11), 'ms', v(12), 'laser', []);
                if v(4) == 1, m.uv = v(2:3); end
                if all(isfinite(v(6:9))), m.box = v(6:9); end
                if all(isfinite(v(13:14))), m.laser = v(13:14); end
                fr = struct('img', img, 'scale', 0.25, 't', v(1), 'meas', m);
            else
                r = cell(o.br.half_frame());
                fr = struct('img', uint8(r{2}), 'scale', 0.5, 't', double(r{1}), 'meas', []);
            end
        end
        function release(o), try, o.br.close(); catch, end, end
    end
end
