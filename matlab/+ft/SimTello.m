classdef SimTello < handle
    %FT.SIMTELLO  simulated drone with the TelloUDP interface. Wall-plane plant (same model the controller uses):
    %   v_ss = B [a; c] (cm/s), tau_i dv_i/dt = v_ss,i(t-L) - v_i, dp/dt = v + wind (random walk)
    properties
        B; tau; L; p; v = [0 0]; groundY; climb = 0; takeoffH = 80
        q = zeros(0, 3); u = [0 0]; t = 0; flying = false
        state = struct('bat', 87, 'h', 0); lastRc = [0 0 0 0]
        wind = [0 0]; windSigma = 3; rngS
        pending = ''; pendingT = 0; pendingReply = ''
    end
    methods
        function o = SimTello(model, startXY, windSigma)
            o.B = model.B; o.tau = [model.axes.x.tau, model.axes.y.tau]; o.L = model.axes.x.L;
            o.p = startXY; o.groundY = startXY(2);
            if nargin > 2, o.windSigma = windSigma; end
            o.rngS = RandStream('mt19937ar', 'Seed', 0);
        end
        function ok = connect(~), ok = true; end
        function b = battery(o), b = o.state.bat; end
        function startCmd(o, cmd)
            o.pending = ''; o.pendingReply = 'ok';
            if strcmp(cmd, 'takeoff'), o.flying = true; o.climb = o.takeoffH; o.pending = 'takeoff'; o.pendingT = o.t; end
            if strcmp(cmd, 'land'), o.flying = false; o.q = zeros(0, 3); o.u = [0 0]; end
        end
        function done = pollReply(o, ~)
            done = isempty(o.pending) || o.climb <= 0;
            if done, o.pending = ''; end
        end
        function rc(o, a, b, c, d)
            v = round(min(max([a b c d], -100), 100)); o.lastRc = v;
            o.q(end + 1, :) = [o.t + o.L, v(1), v(3)];
        end
        function land(o), o.startCmd('land'); end
        function emergency(o), o.startCmd('land'); end
        function pollState(~), end
        function close(~), end
        function step(o, dt)
            o.t = o.t + dt;
            while ~isempty(o.q) && o.q(1, 1) <= o.t, o.u = o.q(1, 2:3); o.q(1, :) = []; end
            if ~o.flying
                o.p(2) = max(o.groundY, o.p(2) - 50 * dt); o.v = [0 0]; return
            end
            if o.climb > 0
                dz = min(o.climb, 60 * dt); o.p(2) = o.p(2) + dz; o.climb = o.climb - dz; return
            end
            o.wind = o.wind + o.windSigma * sqrt(dt) * randn(o.rngS, 1, 2) - 0.3 * o.wind * dt;
            vss = (o.B * o.u(:))';
            o.v = o.v + (vss - o.v) ./ o.tau * dt;
            o.p = o.p + (o.v + o.wind) * dt;
            o.state.bat = max(0, o.state.bat - 0.02 * dt); o.state.h = o.p(2) - o.groundY;
        end
    end
end
