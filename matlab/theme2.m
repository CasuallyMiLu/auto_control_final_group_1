function theme2(varargin)
%THEME2  主題 2 循跡導引計時賽: take-off -> red path A -> B -> follow the green laser dot -> land as soon as the
%   drone is over the lower blue square.        theme2()      theme2('sim', true)
addpath(fileparts(mfilename('fullpath')));
ft.run_mission('theme2', 'path', ft.parse_opts(varargin{:}));
end
