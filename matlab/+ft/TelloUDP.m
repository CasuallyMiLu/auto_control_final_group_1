classdef TelloUDP < handle
    %FT.TELLOUDP  Tello SDK over UDP with Java sockets (no toolbox needed; same text commands as ryzeRC.m).
    %   commands -> 192.168.10.1:8889 (replies to our port 8889); state <- port 8890 (~10 Hz).
    %   takeoff/land are sent asynchronously; pollReply() is called every loop so vision keeps running.
    properties
        sock; stateSock; addr; port = 8889
        state = struct(); stateT = 0
        flying = false; lastRc = [0 0 0 0]
        pending = ''; pendingT = 0; pendingReply = ''
    end
    methods
        function o = TelloUDP(ip)
            if nargin < 1, ip = '192.168.10.1'; end
            o.addr = java.net.InetAddress.getByName(ip);
            o.sock = java.net.DatagramSocket(8889); o.sock.setSoTimeout(1);
            o.stateSock = java.net.DatagramSocket(8890); o.stateSock.setSoTimeout(1);
        end
        function sendRaw(o, cmd)
            b = int8(uint8(cmd));
            o.sock.send(java.net.DatagramPacket(b, numel(b), o.addr, o.port));
        end
        function r = recvReply(o, timeout)
            r = ''; t0 = tic;
            while toc(t0) < timeout
                r = o.tryRecv(o.sock);
                if ~isempty(r), return; end
            end
        end
        function r = send(o, cmd, timeout)
            if nargin < 3, timeout = 7; end
            while ~isempty(o.tryRecv(o.sock)), end  % flush
            o.sendRaw(cmd); r = o.recvReply(timeout);
        end
        function ok = connect(o)
            ok = false;
            for k = 1:5
                r = o.send('command', 2);
                if startsWith(lower(r), 'ok'), ok = true; return; end
            end
        end
        function b = battery(o)
            o.pollState();
            if isfield(o.state, 'bat') && toc(uint64(o.stateT)) < 2 %#ok<*TOCMIX>
                b = o.state.bat; return
            end
            b = str2double(o.send('battery?', 2));
        end
        function startCmd(o, cmd)   % asynchronous takeoff / land
            o.pending = cmd; o.pendingT = tic; o.pendingReply = '';
            o.sendRaw(cmd);
            if strcmp(cmd, 'takeoff'), o.flying = true; end
            if strcmp(cmd, 'land'), o.flying = false; end
        end
        function done = pollReply(o, timeout)
            % true when the pending command was answered or timed out
            done = true;
            if isempty(o.pending), return; end
            r = o.tryRecv(o.sock);
            if ~isempty(r), o.pendingReply = r; o.pending = ''; return; end
            if toc(o.pendingT) > timeout, o.pending = ''; return; end
            done = false;
        end
        function rc(o, a, b, c, d)
            v = round(min(max([a b c d], -100), 100));
            o.lastRc = v;
            o.sendRaw(sprintf('rc %d %d %d %d', v));
        end
        function land(o)
            o.rc(0, 0, 0, 0); o.send('land', 8); o.flying = false;
        end
        function emergency(o), o.sendRaw('emergency'); o.flying = false; end
        function pollState(o)
            while true
                r = o.tryRecv(o.stateSock);
                if isempty(r), break; end
                kv = split(string(strtrim(r)), ';');
                for k = 1:numel(kv)
                    p = split(kv(k), ':');
                    if numel(p) == 2
                        val = str2double(p(2));
                        if ~isnan(val), o.state.(matlab.lang.makeValidName(char(p(1)))) = val; end
                    end
                end
                o.stateT = tic;
            end
        end
        function close(o)
            try, o.sock.close(); catch, end
            try, o.stateSock.close(); catch, end
        end
    end
    methods (Static, Access = private)
        function r = tryRecv(s)
            r = '';
            pkt = java.net.DatagramPacket(zeros(1, 1024, 'int8'), 1024);
            try
                s.receive(pkt);
                d = typecast(pkt.getData(), 'uint8');
                r = char(d(1:pkt.getLength())');
            catch
            end
        end
    end
end
