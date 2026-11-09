classdef CamWebcam < handle
    %FT.CAMWEBCAM  MATLAB Support Package for USB Webcams. index 'auto' = first camera whose image contains a board.
    properties
        cam; t0; vw = []
    end
    methods
        function o = CamWebcam(index, w, h, kind, cfg, recordFile)
            names = webcamlist;
            if strcmp(string(index), "auto")
                idx = 1;
                for i = 1:numel(names)
                    try
                        c = webcam(i); img = snapshot(c); clear c
                        ft.calibrate(img, kind, cfg, 'auto'); idx = i; break
                    catch
                    end
                end
            else
                idx = double(index) + 1;
            end
            o.cam = webcam(idx);
            try, o.cam.Resolution = sprintf('%dx%d', w, h); catch, end
            o.t0 = tic;
            if nargin > 5 && ~isempty(recordFile)
                o.vw = VideoWriter(recordFile, 'MPEG-4'); o.vw.FrameRate = 20; open(o.vw);
            end
        end
        function f = fullFrame(o), f = snapshot(o.cam); end
        function fr = read(o)
            f = snapshot(o.cam); t = toc(o.t0);
            h = imresize(f, 0.5);
            if ~isempty(o.vw), writeVideo(o.vw, h); end
            fr = struct('img', h, 'scale', 0.5, 't', t, 'meas', []);
        end
        function release(o)
            if ~isempty(o.vw), close(o.vw); end
            clear o.cam
        end
    end
end
