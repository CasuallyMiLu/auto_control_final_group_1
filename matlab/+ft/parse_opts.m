function opt = parse_opts(varargin)
%FT.PARSE_OPTS  name-value options shared by theme1 / theme2 / sysid_flight
ip = inputParser;
ip.addParameter('sim', false);            % simulated drone + rendered camera, no hardware
ip.addParameter('yes', false);            % skip the preview (G to go)
ip.addParameter('show', true);            % dashboard window
ip.addParameter('record', false);         % save dashboard.mp4 (slower; OBS screen recording is an alternative)
ip.addParameter('vision', 'auto');        % auto | bsm (MATLAB background subtraction) | yolo (Python YOLO26)
ip.addParameter('camera', 'auto');        % auto | webcam (MATLAB support package) | python (OpenCV)
ip.addParameter('set', {});               % gain overrides, e.g. {'path.v_max=30','cross.x.Kp=2.2'}
ip.addParameter('tag', '');
ip.addParameter('upHint', '');            % '' = rig.json (auto) | up | right | down | left
ip.addParameter('probe', 'auto');         % auto | off
ip.addParameter('stage_cm', 25);          % theme 1: hold this far outside the start before tracking (step)
ip.addParameter('model', '');             % other model.json
ip.addParameter('gains', '');             % other gains.json
ip.addParameter('simModel', '');          % sim: plant of the simulated drone (default config/model.json)
ip.addParameter('simWind', 3);            % sim: random-walk disturbance (cm/s/sqrt(s))
ip.addParameter('simFlipA', false);       % sim: drone faces the camera (rc a reversed)
ip.addParameter('axes', {'x', 'y'});      % sysid
ip.addParameter('amps', [25 40]);         % sysid doublet amplitudes (rc)
ip.addParameter('durs', [0.8 1.3]);       % sysid doublet durations (s)
ip.addParameter('box', 65);               % sysid: abort a block this far from the centre (cm)
ip.addParameter('seed', 0);
ip.addParameter('holdOffset', [0 30]);    % sysid hold point relative to the track centre (off the black cross)
ip.parse(varargin{:});
opt = ip.Results;
end
