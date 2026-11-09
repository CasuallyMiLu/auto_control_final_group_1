classdef Actuator < handle
    %FT.ACTUATOR  desired wall velocity (cm/s) -> rc [a b c d]. B (2x2, cm/s per rc) maps [a c] -> [vX vY].
    properties
        B; Binv; rcMax = 60; accelMax = 200; vPrev = [0 0]
    end
    methods
        function o = Actuator(B, rcMax, accelMax)
            o.B = B; o.Binv = inv(B); o.rcMax = rcMax; o.accelMax = accelMax;
        end
        function reset(o), o.vPrev = [0 0]; end
        function flipA(o), o.B(:, 1) = -o.B(:, 1); o.Binv = inv(o.B); end
        function [rc, v] = step(o, vDes, dt)
            dv = vDes - o.vPrev; lim = o.accelMax * max(dt, 1e-3);
            if norm(dv) > lim, vDes = o.vPrev + dv * lim / norm(dv); end
            r = (o.Binv * vDes(:))';
            m = max(abs(r));
            if m > o.rcMax, r = r * o.rcMax / m; vDes = (o.B * r(:))'; end
            o.vPrev = vDes; v = vDes;
            rc = [r(1) 0 r(2) 0];
        end
    end
end
