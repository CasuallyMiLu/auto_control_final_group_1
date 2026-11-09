classdef PointTracker < handle
    %FT.POINTTRACKER  per-axis PID towards a target point (laser dot / hover point); target low-pass filtered
    properties
        g; pid; tf = []; vMax
    end
    methods
        function o = PointTracker(gains, key)
            o.g = gains.(key);
            o.pid = {ft.PID(gains.point_axes.x, o.g.i_zone, o.g.i_max), ft.PID(gains.point_axes.y, o.g.i_zone, o.g.i_max)};
            o.vMax = o.g.v_max;
        end
        function reset(o), o.pid{1}.reset(); o.pid{2}.reset(); o.tf = []; end
        function [v, info] = step(o, x, target, dt)
            if isempty(o.tf), o.tf = target; else, o.tf = o.tf + (target - o.tf) * min(1, dt / max(o.g.target_tau, 1e-3)); end
            e = o.tf - x;
            if norm(e) < o.g.deadband, e = [0 0]; end
            v = [o.pid{1}.step(e(1), dt, o.vMax), o.pid{2}.step(e(2), dt, o.vMax)];
            if norm(v) > o.vMax, v = v * o.vMax / norm(v); end
            info = struct('ref', o.tf, 'e', e);
        end
    end
end
