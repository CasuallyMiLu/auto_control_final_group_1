function theme1(scene, varargin)
%THEME1  主題 1 循跡飛行: start -> take-off -> hold 25 cm outside the start -> one lap on the white centre line -> land
%   theme1('circle')            theme1('square')
%   theme1('circle', 'sim', true)                    % dry run, no drone / camera
%   theme1('circle', 'set', {'path.v_max=30'})       % override a gain
%   options: see +ft/parse_opts.m
addpath(fileparts(mfilename('fullpath')));
ft.run_mission('theme1', scene, ft.parse_opts(varargin{:}));
end
