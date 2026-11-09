classdef PathFollower < handle
    %FT.PATHFOLLOWER  v_des = v_ff(s) * t + u_n * n,  u_n = -PID(e_n)   (e_n: signed cross-track error, cm)
    %   v_ff from the speed profile (curvature + acceleration limits), reduced while far from the path, so the
    %   approach to the line is a clean step response of the cross-track loop. Gains blended between the X / Y
    %   designs by the direction of the correction (normal vector).
    properties
        p; g; pidX; pidY; pid; s = []; travel = 0; target; done = false
    end
    methods
        function o = PathFollower(path, gains, laps)
            if nargin < 3, laps = 1; end
            o.g = gains.path;
            iz = o.g.i_zone; im = o.g.i_max;
            o.pidX = ft.PID(gains.cross.x, iz, im); o.pidY = ft.PID(gains.cross.y, iz, im); o.pid = ft.PID(gains.cross.x, iz, im);
            o.p = ft.path_speed_profile(path, o.g.v_max, o.g.a_lat, o.g.a_long, o.g.v_min);
            if path.closed, o.target = laps * path.L; else, o.target = path.L; end
        end
        function reset(o), o.s = []; o.travel = 0; o.done = false; o.pid.reset(); end
        function [v, info] = step(o, x, dt)
            pr = ft.path_project(o.p, x, o.s, 25, o.g.search_fwd);
            if ~isempty(o.s)
                ds = pr.s - o.s;
                if o.p.closed, ds = mod(ds + o.p.L / 2, o.p.L) - o.p.L / 2; end
                o.travel = o.travel + max(ds, -5);
            end
            o.s = pr.s;
            e = pr.e_n; n = pr.nrm;
            w = n(2) ^ 2;
            gx = o.pidX; gy = o.pidY;
            o.pid.Kp = (1 - w) * gx.Kp + w * gy.Kp; o.pid.Ki = (1 - w) * gx.Ki + w * gy.Ki;
            o.pid.Kd = (1 - w) * gx.Kd + w * gy.Kd; o.pid.N = (1 - w) * gx.N + w * gy.N;
            un = -o.pid.step(e, dt);
            approach = min(max(1 - abs(e) / o.g.e_ff0, 0), 1);
            lead = o.g.ff_lead_cm;
            vff = ft.path_speed_at(o.p, o.s + lead) * approach;
            if ~o.p.closed
                rem = o.p.L - o.s;
                vff = min(vff, sqrt(2 * o.g.a_long * max(rem - 2, 0)));
            end
            [~, th] = ft.path_point(o.p, o.s + lead);
            v = vff * th + un * n;
            if o.travel >= o.target - (~o.p.closed) * 3 || (~o.p.closed && o.p.L - o.s < 3)
                o.done = true;
            end
            info = struct('s', o.s, 'e_n', e, 'ref', pr.q, 'v_ff', vff, 'progress', min(1, o.travel / max(o.target, 1e-6)));
        end
    end
end
