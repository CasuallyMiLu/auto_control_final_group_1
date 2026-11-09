function p = path_make(pts, closed, ds)
%FT.PATH_MAKE  reference path in wall cm, resampled every ds cm (default 1). Fields: pts, closed, seg, seglen,
%   s (arclength of each vertex), L (length), tan (unit tangent per segment), kappa (curvature 1/cm), vmax.
if nargin < 3, ds = 1.0; end
P = pts; if closed, P = [pts; pts(1, :)]; end
d = vecnorm(diff(P), 2, 2); P = P([true; d > 1e-9], :);
s = [0; cumsum(vecnorm(diff(P), 2, 2))];
n = max(ceil(s(end) / ds), 4);
si = linspace(0, s(end), n + 1)';
Q = [interp1(s, P(:, 1), si), interp1(s, P(:, 2), si)];
if closed, Q = Q(1:end - 1, :); end
p.pts = Q; p.closed = closed;
if closed, p.seg = diff([Q; Q(1, :)]); else, p.seg = diff(Q); end
p.seglen = vecnorm(p.seg, 2, 2);
p.s = [0; cumsum(p.seglen)];
p.L = p.s(end);
p.tan = p.seg ./ max(p.seglen, 1e-9);
% curvature from the heading change over +-6 cm
k = max(1, round(6 / median(p.seglen)));
raw = atan2(p.tan(:, 2), p.tan(:, 1));
if closed
    hp = unwrap([raw(end - k + 1:end); raw; raw(1:k)]);
else
    h = unwrap(raw); hp = [repmat(h(1), k, 1); h; repmat(h(end), k, 1)];
end
dh = hp(2 * k + 1:end) - hp(1:end - 2 * k);
kap = abs(dh) / (2 * k * median(p.seglen));
n = size(Q, 1);
if numel(kap) < n, kap = [kap; repmat(kap(end), n - numel(kap), 1)]; end
p.kappa = kap(1:n);
p.vmax = [];
end
