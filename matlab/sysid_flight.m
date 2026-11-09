function sysid_flight(scene, varargin)
%SYSID_FLIGHT  系統辨識飛行: hold at the board centre, open-loop doublets + PRBS on rc a (X) and rc c (Y), land.
%   sysid_flight('circle')      then:  sysid_tello('../runs/<run>', true); design_pid([], 60, true)
if nargin < 1, scene = 'circle'; end
addpath(fileparts(mfilename('fullpath')));
ft.run_mission('sysid', scene, ft.parse_opts(varargin{:}));
end
