classdef Dashboard < handle
    %FT.DASHBOARD  live display (電腦同步動態顯示控制狀態): upright camera view with path / reference / trail /
    %   laser / landing zone, rolling charts of cross-track error, position error and rc commands, status text.
    %   Keys: SPACE or q = land, p = pause/resume, ESC = emergency stop, g / Enter = go (preview).
    properties
        fig; axI; hImg; hPath; hTrail; hDrone; hRef; hLaser; hZone; hBox
        axE; axXY; axRC; hE; hEx; hEy; hA; hC; hTxt
        sc; k = 0; W = 1; H = 1; span = 20
        buf; trail = nan(600, 2); key = ''; vw = []
    end
    methods
        function o = Dashboard(sc, title_, show, recordFile)
            o.sc = sc;
            vis = 'on'; if ~show, vis = 'off'; end
            o.fig = figure('Name', title_, 'NumberTitle', 'off', 'Color', [0.1 0.1 0.1], 'Position', [40 60 1500 780], ...
                'Visible', vis, 'KeyPressFcn', @(~, e) o.onKey(e), 'MenuBar', 'none', 'ToolBar', 'none');
            o.axI = axes(o.fig, 'Position', [0.01 0.02 0.46 0.96]); hold(o.axI, 'on'); axis(o.axI, 'off');
            o.hImg = image(o.axI, zeros(10, 10, 3, 'uint8'));
            set(o.axI, 'YDir', 'reverse', 'DataAspectRatio', [1 1 1]);
            o.hPath = plot(o.axI, nan, nan, 'g-', 'LineWidth', 2);
            o.hTrail = plot(o.axI, nan, nan, 'r-', 'LineWidth', 2);
            o.hBox = plot(o.axI, nan, nan, 'g-', 'LineWidth', 1);
            o.hRef = plot(o.axI, nan, nan, 'mo', 'MarkerSize', 12, 'LineWidth', 2);
            o.hLaser = plot(o.axI, nan, nan, 'x', 'Color', [0.2 1 0.2], 'MarkerSize', 16, 'LineWidth', 3);
            o.hZone = plot(o.axI, nan, nan, '-', 'Color', [1 0.6 0], 'LineWidth', 2);
            o.hDrone = plot(o.axI, nan, nan, 'o', 'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k', 'MarkerSize', 9);
            o.axE = o.mkAxes([0.52 0.72 0.46 0.24], 'cross-track error e_n (cm), green = \pm5 cm', [-40 40]);
            patch(o.axE, [-1e5 1e5 1e5 -1e5], [-5 -5 5 5], [0.2 0.4 0.2], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
            o.hE = plot(o.axE, nan, nan, '-', 'Color', [1 0.8 0], 'LineWidth', 1.5);
            o.axXY = o.mkAxes([0.52 0.43 0.46 0.24], 'position error ex (blue) / ey (green) (cm)', [-60 60]);
            o.hEx = plot(o.axXY, nan, nan, '-', 'Color', [0.4 0.6 1], 'LineWidth', 1.2);
            o.hEy = plot(o.axXY, nan, nan, '-', 'Color', [0.4 1 0.4], 'LineWidth', 1.2);
            o.axRC = o.mkAxes([0.52 0.14 0.46 0.24], 'rc command a (blue, left/right) / c (green, up/down)', [-100 100]);
            o.hA = plot(o.axRC, nan, nan, '-', 'Color', [0.4 0.6 1], 'LineWidth', 1.2);
            o.hC = plot(o.axRC, nan, nan, '-', 'Color', [0.4 1 0.4], 'LineWidth', 1.2);
            o.hTxt = annotation(o.fig, 'textbox', [0.52 0.0 0.46 0.11], 'Color', 'w', 'EdgeColor', 'none', ...
                'FontName', 'Consolas', 'FontSize', 10, 'Interpreter', 'none', 'VerticalAlignment', 'top');
            o.buf = nan(1500, 6);
            o.setScene(sc);
            if nargin > 3 && ~isempty(recordFile)
                o.vw = VideoWriter(recordFile, 'MPEG-4'); o.vw.FrameRate = 15; open(o.vw);
            end
        end
        function ax = mkAxes(o, pos, ttl, yl)
            ax = axes(o.fig, 'Position', pos, 'Color', [0.15 0.15 0.15], 'XColor', [0.7 0.7 0.7], 'YColor', [0.7 0.7 0.7]);
            hold(ax, 'on'); grid(ax, 'on'); ylim(ax, yl); title(ax, ttl, 'Color', 'w', 'FontWeight', 'normal');
        end
        function setScene(o, sc)
            o.sc = sc;
            ang = atan2d(sc.up(2), sc.up(1));
            o.k = mod(round((ang + 90) / 90), 4);   % rot90(img, k): world up -> screen up
        end
        function onKey(o, e)
            switch e.Key
                case {'space', 'q'}, o.key = 'land';
                case 'escape', o.key = 'emergency';
                case 'p', o.key = 'pause';
                case {'g', 'return'}, o.key = 'go';
                case 'l', o.key = 'landkey';
            end
        end
        function k = takeKey(o), k = o.key; o.key = ''; end
        function q = rot(o, p, s)
            % full-res px -> displayed px (scaled by s, then rot90 k times counter-clockwise)
            q = (p - 0.5) * s + 0.5;
            w = o.W;
            h = o.H;
            for i = 1:o.k
                q = [q(:, 2), w - q(:, 1) + 1];
                t = w; w = h; h = t;
            end
        end
        function render(o, fr, st)
            if ~isgraphics(o.fig), return; end
            s = fr.scale; img = fr.img;
            o.H = size(img, 1); o.W = size(img, 2);
            R = rot90(img, o.k);
            o.hImg.CData = R;
            set(o.axI, 'XLim', [0.5 size(R, 2) + 0.5], 'YLim', [0.5 size(R, 1) + 0.5]);
            P = ft.to_px(o.sc, o.sc.path.pts); if o.sc.path.closed, P = [P; P(1, :)]; end
            q = o.rot(P, s); set(o.hPath, 'XData', q(:, 1), 'YData', q(:, 2));
            o.trail = [o.trail(2:end, :); nan nan];
            if ~isempty(st.uv), o.trail(end, :) = st.uv; end
            q = o.rot(o.trail, s); set(o.hTrail, 'XData', q(:, 1), 'YData', q(:, 2));
            o.setPt(o.hDrone, st.uv, s);
            if ~isempty(st.box)
                b = st.box; B = [b(1) b(2); b(3) b(2); b(3) b(4); b(1) b(4); b(1) b(2)];
                q = o.rot(B, s); set(o.hBox, 'XData', q(:, 1), 'YData', q(:, 2));
            else
                set(o.hBox, 'XData', nan, 'YData', nan);
            end
            if ~isempty(st.ref), o.setPt(o.hRef, ft.to_px(o.sc, st.ref), s); else, o.setPt(o.hRef, [], s); end
            o.setPt(o.hLaser, st.laser, s);
            if ~isempty(st.zone)
                a = linspace(0, 2 * pi, 40)'; Z = st.zone(1:2) + st.zone(3) * [cos(a) sin(a)];
                q = o.rot(ft.to_px(o.sc, Z), s); set(o.hZone, 'XData', q(:, 1), 'YData', q(:, 2));
            end
            o.buf = [o.buf(2:end, :); st.t, st.e_n, st.exy, st.rc(1), st.rc(3)];
            T = o.buf(:, 1);
            set(o.hE, 'XData', T, 'YData', o.buf(:, 2));
            set(o.hEx, 'XData', T, 'YData', o.buf(:, 3)); set(o.hEy, 'XData', T, 'YData', o.buf(:, 4));
            set(o.hA, 'XData', T, 'YData', o.buf(:, 5)); set(o.hC, 'XData', T, 'YData', o.buf(:, 6));
            for ax = [o.axE o.axXY o.axRC], xlim(ax, [st.t - o.span, st.t + 0.5]); end
            o.hTxt.String = sprintf(['%s   phase: %s   t = %6.2f s\npos X %7.1f  Y %7.1f cm   e_n %+6.1f cm   RMS %5.1f cm   progress %5.1f %%\n' ...
                'battery %3.0f %%   loop %4.1f Hz   vision %4.0f ms   %s\nkeys: SPACE/q land   p pause   ESC emergency stop'], ...
                st.title, st.phase, st.t, st.xy(1), st.xy(2), st.e_n, st.rms, 100 * st.progress, st.bat, st.hz, st.vms, st.msg);
            drawnow limitrate;
            if ~isempty(o.vw), writeVideo(o.vw, getframe(o.fig)); end
        end
        function setPt(o, h, p, s)
            if isempty(p), set(h, 'XData', nan, 'YData', nan); return; end
            q = o.rot(p, s); set(h, 'XData', q(1), 'YData', q(2));
        end
        function close(o)
            if ~isempty(o.vw), close(o.vw); end
            if isgraphics(o.fig), close(o.fig); end
        end
    end
end
