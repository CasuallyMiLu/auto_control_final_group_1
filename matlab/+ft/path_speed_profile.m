function p = path_speed_profile(p, vmax, aLat, aLong, vmin)
%FT.PATH_SPEED_PROFILE  v(s) <= sqrt(aLat/kappa), then forward/backward passes limit acceleration (v dv/ds = a)
if nargin < 5, vmin = 6; end
v = min(vmax, sqrt(aLat ./ max(p.kappa, 1e-6)));
v = max(v, vmin);
ds = [p.seglen; p.seglen(end)]; ds = ds(1:numel(v));
for i = 2:numel(v), v(i) = min(v(i), sqrt(v(i - 1) ^ 2 + 2 * aLong * ds(i - 1))); end
for i = numel(v) - 1:-1:1, v(i) = min(v(i), sqrt(v(i + 1) ^ 2 + 2 * aLong * ds(i))); end
if ~p.closed, v(end) = vmin; end
p.vmax = v;
end
