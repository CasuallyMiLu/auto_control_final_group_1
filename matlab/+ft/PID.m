classdef PID < handle
    %FT.PID  C(s) = Kp + Ki/s + Kd*N*s/(s+N) (Simulink 'Parallel' form), backward Euler with the real dt.
    %   Anti-windup: no integration while saturated in the error's direction; integral only when |e| < iZone.
    properties
        Kp = 1; Ki = 0; Kd = 0; N = 10; iZone = inf; iMax = inf
        I = 0; D = 0; ePrev = []
    end
    methods
        function o = PID(g, iZone, iMax)
            o.Kp = g.Kp; o.Ki = g.Ki; o.Kd = g.Kd; o.N = g.N;
            if nargin > 1, o.iZone = iZone; end
            if nargin > 2, o.iMax = iMax; end
        end
        function reset(o), o.I = 0; o.D = 0; o.ePrev = []; end
        function u = step(o, e, dt, sat)
            if nargin < 4, sat = inf; end
            dt = min(max(dt, 1e-3), 0.2);
            if isempty(o.ePrev), o.ePrev = e; end
            o.D = (o.D + o.Kd * o.N * (e - o.ePrev)) / (1 + o.N * dt);
            o.ePrev = e;
            u = o.Kp * e + o.I + o.D;
            saturated = abs(u) >= sat && sign(u) == sign(e);
            if abs(e) < o.iZone && ~saturated
                o.I = min(max(o.I + o.Ki * e * dt, -o.iMax), o.iMax);
            end
        end
        function setGains(o, g), o.Kp = g.Kp; o.Ki = g.Ki; o.Kd = g.Kd; o.N = g.N; end
    end
end
