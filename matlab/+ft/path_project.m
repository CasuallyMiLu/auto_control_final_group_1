function pr = path_project(p, x, sHint, back, fwd)
%FT.PATH_PROJECT  closest point of the path to x. With sHint only [sHint-back, sHint+fwd] is searched so the
%   progress never jumps to another part of the path. pr.s, pr.q, pr.tan, pr.nrm (left normal), pr.e_n (signed
%   cross-track error, + = left of the travel direction), pr.dist, pr.kappa
if nargin < 3, sHint = []; end
if nargin < 4, back = 25; end
if nargin < 5, fwd = 60; end
a = p.pts(1:numel(p.seglen), :);
t = sum((x - a) .* p.seg, 2) ./ max(p.seglen .^ 2, 1e-12);
t = min(max(t, 0), 1);
q = a + p.seg .* t;
d2 = sum((x - q) .^ 2, 2);
if ~isempty(sHint)
    sv = p.s(1:end - 1) + t .* p.seglen;
    ds = sv - sHint;
    if p.closed, ds = mod(ds + p.L / 2, p.L) - p.L / 2; end
    d2(ds < -back | ds > fwd) = inf;
    if all(isinf(d2)), pr = ft.path_project(p, x); return; end
end
[dmin, i] = min(d2);
pr.s = p.s(i) + t(i) * p.seglen(i);
pr.q = q(i, :);
pr.tan = p.tan(i, :);
pr.nrm = [-pr.tan(2), pr.tan(1)];
pr.e_n = dot(x - pr.q, pr.nrm);
pr.dist = sqrt(dmin);
pr.kappa = p.kappa(i);
end
