function sc = calibrate(rgb, kind, cfg, upHint)
%FT.CALIBRATE  Automatic scene calibration from one full-resolution RGB camera frame (MATLAB twin of ftrack/scene.py).
%   sc = ft.calibrate(rgb, 'circle'|'square'|'path', cfg, upHint)
%   Finds the red track, its white centre line (circle/square) or medial axis (path), blue squares and the black
%   obstacle, and the map image px <-> wall cm (X right, Y up):   xy = (uv - sc.origin) * sc.A'
%   Orientation: fine angle from the track/cross edges (mod 90 deg); coarse: path -> B end (next to a blue square)
%   is the top; circle/square -> side of the image with the wood floor is down (confirmed by the climb probe).
%   Scale: white centre line 150 cm (circle/square) or red line width 20 cm (path) -> zoom needs no setting.
if nargin < 4 || isempty(upHint), upHint = 'auto'; end
g = cfg.rig.geometry;
cp = cfg.rig.closed_path;
switch kind
    case 'circle', sc = calibCircle(rgb, g, upHint);
    case 'square', sc = calibSquare(rgb, g, upHint);
    case 'path',   sc = calibPath(rgb, g, upHint);
    otherwise, error('unknown scene %s', kind);
end
sc.kind = kind;
sc.shape = size(rgb);
sc.A = double(sc.A); sc.origin = double(sc.origin); sc.up = double(sc.up); sc.s_px = double(sc.s_px);
sc.blue_px = double(sc.blue_px); sc.obstacle_px = double(sc.obstacle_px);
if ~strcmp(kind, 'path')
    sa = cp.start_angle_deg;
    if isstruct(sa), sa = sa.(kind); end
    sc = ft.orient_closed(sc, cp.direction, sa);
end
end

% =========================================================================================== masks / helpers
function m = redMask(rgb)
hsv = rgb2hsv(rgb);
h = hsv(:, :, 1); s = hsv(:, :, 2); v = hsv(:, :, 3);
m = ((h < 10/180) | (h > 165/180)) & (s > 90/255) & (v > 60/255);
m = imopen(m, ones(3));
end

function blobs = blueBlobs(rgb)
hsv = rgb2hsv(rgb);
m = hsv(:, :, 1) > 92/180 & hsv(:, :, 1) < 128/180 & hsv(:, :, 2) > 80/255 & hsv(:, :, 3) > 60/255;
m = imopen(m, ones(5));
st = regionprops(m, 'Area', 'BoundingBox', 'Centroid');
st = st([st.Area] >= 0.0004 * numel(m));
[~, o] = sort([st.Area], 'descend'); st = st(o(1:min(4, end)));
blobs = struct('c', {}, 'box', {});
for k = 1:numel(st)
    b = st(k).BoundingBox; w = b(3); h = b(4);
    if w / h > 0.5 && w / h < 2 && st(k).Area > 0.6 * w * h
        blobs(end + 1) = struct('c', st(k).Centroid, 'box', b); %#ok<AGROW>
    end
end
end

function ang = dominantAngleMod90(mask)
m = imgaussfilt(double(mask), 1.2);
[gx, gy] = imgradientxy(m, 'sobel');
mag = hypot(gx, gy);
sel = mag > 0.2 * max(mag(:));
th = atan2(gy(sel), gx(sel));
z = sum(mag(sel) .* exp(4i * th));
ang = rad2deg(angle(z) / 4);
end

function v = rotv(v, deg)
c = cosd(deg); s = sind(deg);
v = [c * v(1) - s * v(2), s * v(1) + c * v(2)];
end

function up = upFromHint(hint, fine)
D = struct('up', [0 -1], 'right', [1 0], 'down', [0 1], 'left', [-1 0]);
up = rotv(D.(hint), fine);
end

function [side, conf] = floorSide(rgb)
small = imresize(rgb, [NaN 480]);
hsv = rgb2hsv(small);
g = double(rgb2gray(small));
tex = abs(imfilter(g, fspecial('laplacian', 0), 'replicate'));
[h, w] = size(g); b = round(0.07 * min(h, w));
S = struct('down', {{h - b + 1:h, 1:w}}, 'up', {{1:b, 1:w}}, 'left', {{1:h, 1:b}}, 'right', {{1:h, w - b + 1:w}});
names = fieldnames(S); score = zeros(1, 4);
for k = 1:4
    r = S.(names{k});
    H = hsv(r{1}, r{2}, 1); Sa = hsv(r{1}, r{2}, 2); V = hsv(r{1}, r{2}, 3);
    woody = mean(H(:) > 5/180 & H(:) < 28/180 & Sa(:) > 45/255 & V(:) > 60/255 & V(:) < 235/255);
    T = tex(r{1}, r{2});
    score(k) = woody + 0.01 * mean(T(:));
end
[ss, o] = sort(score, 'descend');
side = names{o(1)};
conf = max(0, min(1, (ss(1) - ss(2)) / (ss(1) + 1e-6)));
end

function [up, info] = resolveUp(rgb, upHint, fine)
OPP = struct('down', 'up', 'up', 'down', 'left', 'right', 'right', 'left');
if ~strcmp(upHint, 'auto')
    up = upFromHint(upHint, fine); info = struct('up_source', 'config');
else
    [side, conf] = floorSide(rgb);
    up = upFromHint(OPP.(side), fine);
    info = struct('up_source', 'floor', 'floor_side', side, 'floor_conf', conf);
end
end

function A = affineFrom(up, s)
right = [-up(2), up(1)];
A = [right * s; up * s];
end

function [B, big] = trackBoundaries(m)
cc = bwconncomp(m, 8);
n = cellfun(@numel, cc.PixelIdxList);
keep = n >= 0.002 * numel(m);
if ~any(keep), error('no red track found: check camera / lighting'); end
[~, o] = sort(n, 'descend'); o = o(1:min(2, nnz(keep)));
big = false(size(m)); big(vertcat(cc.PixelIdxList{o})) = true;
Bs = bwboundaries(big, 8);  % outer boundaries + holes
if numel(Bs) < 4, error('red track: expected 4 edges (2 stripes), found %d', numel(Bs)); end
ar = cellfun(@(b) polyarea(b(:, 2), b(:, 1)), Bs);
[~, o] = sort(ar, 'descend');
B = cellfun(@(b) fliplr(b), Bs(o(1:4)), 'UniformOutput', false);  % [x y] = [col row]
end

% =========================================================================================== circle
function sc = calibCircle(rgb, g, upHint)
m = redMask(rgb);
[B, ~] = trackBoundaries(m);
E0 = ft.fit_ellipse(B{1});
% circle test: normalised radius of the outer edge w.r.t. its fitted ellipse is ~constant for a circle
U0 = [cos(E0.theta) sin(E0.theta); -sin(E0.theta) cos(E0.theta)];
Z = (B{1} - E0.c) * U0';
rn = hypot(Z(:, 1) / E0.a, Z(:, 2) / E0.b);
if std(rn) > 0.02
    error('the red track is not a circle (radius spread %.3f); is this the square board?', std(rn));
end
E = ft.fit_ellipse([B{2}; B{3}]);  % white centre line
E3 = ft.fit_ellipse(B{4});
centre = E.c;
r = round(0.5 * min(E.a, E.b));
x0 = max(1, round(centre(1) - r)); y0 = max(1, round(centre(2) - r));
roi = rgb(y0:min(end, y0 + 2 * r), x0:min(end, x0 + 2 * r), :);
dark = rgb2gray(roi) < 90;
fine = 0; if nnz(dark) > 50, fine = dominantAngleMod90(dark); end
[up, info] = resolveUp(rgb, upHint, fine);
th = E.theta;
U = [cos(th) sin(th); -sin(th) cos(th)];
S = diag([g.centre_line_cm / 2 / E.a, g.centre_line_cm / 2 / E.b]);
Mpx = U' * S * U;
A = affineFrom(up, 1) * Mpx;
R = g.centre_line_cm / 2;
t = linspace(0, 2 * pi, 601)'; t(end) = [];
sc = struct();
sc.origin = centre; sc.up = up; sc.A = A; sc.s_px = sqrt(abs(det(A)));
sc.path = ft.path_make([R * cos(t), R * sin(t)], true);
sc.edges_px = {B{1}, B{4}};
sc.blue_px = zeros(0, 2); sc.obstacle_px = [];
info.ellipse_axes_px = 2 * [E.a E.b]; info.fine_deg = fine;
info.band_cm_measured = (E0.a - E3.a) * g.centre_line_cm / 2 / E.a;
sc.info = info;
sc.blue_px = cell2mat(arrayfun(@(b) b.c, blueBlobs(rgb), 'UniformOutput', false)');
if isempty(sc.blue_px), sc.blue_px = zeros(0, 2); end
end

% =========================================================================================== square
function sc = calibSquare(rgb, g, upHint)
m = redMask(rgb);
[B, ~] = trackBoundaries(m);
[~, ~, rw, rh] = minRect(B{1});
fill = polyarea(B{1}(:, 1), B{1}(:, 2)) / (rw * rh);
if fill < 0.93
    error('the red track is not a square (area / rectangle = %.2f); is this the circle board?', fill);
end
fine = dominantAngleMod90(m);
centre = mean([B{2}; B{3}], 1);
dirs = [1 1; -1 1; -1 -1; 1 -1] / sqrt(2);
Q = zeros(4, 2);
for k = 1:4
    d = rotv(dirs(k, :), fine);
    [~, i2] = max((B{2} - centre) * d'); [~, i3] = max((B{3} - centre) * d');
    Q(k, :) = (B{2}(i2, :) + B{3}(i3, :)) / 2;  % white line corner = between its two edges
end
[up, info] = resolveUp(rgb, upHint, fine);
right = [-up(2), up(1)];
h = g.centre_line_cm / 2;
W = [sign((Q - centre) * right') * h, sign((Q - centre) * up') * h];
X = [Q - centre, ones(4, 1)];
sol = X \ W;            % affine least squares on the 4 corners
A = sol(1:2, :)';
origin = centre - (A \ sol(3, :)')';
sc = struct();
sc.origin = origin; sc.up = up; sc.A = A; sc.s_px = sqrt(abs(det(A)));
sc.path = ft.path_make([h -h; h h; -h h; -h -h], true);
sc.edges_px = {B{1}, B{4}};
res = (Q - origin) * A' - W;
info.fine_deg = fine; info.corner_fit_rms_cm = sqrt(mean(sum(res .^ 2, 2)));
sc.info = info;
bb = blueBlobs(rgb);
sc.blue_px = zeros(0, 2); for k = 1:numel(bb), sc.blue_px(end + 1, :) = bb(k).c; end
sc.obstacle_px = [];
end

function [c, ang, w, h] = minRect(P)
% minimum-area rectangle of a point set (rotating calipers over the convex hull)
k = convhull(P(:, 1), P(:, 2)); H = P(k, :);
best = inf;
for i = 1:size(H, 1) - 1
    e = H(i + 1, :) - H(i, :); a = atan2(e(2), e(1));
    R = [cos(a) sin(a); -sin(a) cos(a)];
    Z = H * R';
    ww = max(Z(:, 1)) - min(Z(:, 1)); hh = max(Z(:, 2)) - min(Z(:, 2));
    if ww * hh < best, best = ww * hh; w = ww; h = hh; ang = a; c = mean(P); end
end
end

% =========================================================================================== theme-2 path
function sc = calibPath(rgb, g, upHint)
m = redMask(rgb);
cc = bwconncomp(m, 8);
n = cellfun(@numel, cc.PixelIdxList);
if isempty(n) || max(n) < 0.002 * numel(m), error('no red path found'); end
[~, i] = max(n);
track = false(size(m)); track(cc.PixelIdxList{i}) = true;
track = imclose(track, ones(7));
f = 0.5;
small = imresize(track, f, 'nearest');
dist = bwdist(~small);
sk = bwskel(small);
dv = sort(dist(sk));
dv = dv(floor(0.25 * numel(dv)) + 1:floor(0.75 * numel(dv)));
widthPx = double(2 * mean(dv) / f + 1.0);
s_px = g.path_width_cm / widthPx;
line = longestSkeletonPath(sk) / f;
line = resamplePx(line, 2.0);
nCut = max(2, round(widthPx / 2));
line = line(nCut + 1:end - nCut, :);
line = smoothPolyline(line, max(3, round(0.15 * widthPx)));
mDir = max(2, round(0.5 * widthPx / 2));
[H, W] = size(track);
for e = [1, 2]
    if e == 1, d = line(1, :) - line(1 + mDir, :); p0 = line(1, :); else, d = line(end, :) - line(end - mDir, :); p0 = line(end, :); end
    d = d / (norm(d) + 1e-9);
    % march straight on while still on the painted line (up to 2 line widths)
    steps = (2:2:2 * widthPx)';
    pts = p0 + steps * d;
    inside = false(size(steps));
    for j = 1:numel(steps)
        u = round(pts(j, 1)); v = round(pts(j, 2));
        inside(j) = u >= 1 && v >= 1 && u <= W && v <= H && track(v, u);
        if ~inside(j), break; end
    end
    ext = pts(inside, :);
    if e == 1, line = [flipud(ext); line]; else, line = [line; ext]; end
end
fine = dominantAngleMod90(track);
bb = blueBlobs(rgb);
if numel(bb) < 2
    error('theme-2 board needs 2 blue squares, found %d; is this the circle/square board?', numel(bb));
end
C = reshape([bb.c], 2, [])';
dB = [min(vecnorm(C - line(1, :), 2, 2)), min(vecnorm(C - line(end, :), 2, 2))];
if dB(1) < dB(2), line = flipud(line); end
info = struct('B_blue_dist_px', min(dB));
if ~strcmp(upHint, 'auto')
    up = upFromHint(upHint, fine); info.up_source = 'config';
else
    ab = line(end, :) - line(1, :);
    names = {'up', 'right', 'down', 'left'}; best = -inf;
    for k = 1:4
        c = upFromHint(names{k}, fine);
        if dot(c, ab) > best, best = dot(c, ab); up = c; end
    end
    info.up_source = 'path A->B';
end
sc = struct();
sc.A = affineFrom(up, s_px); sc.up = up; sc.s_px = s_px; sc.origin = line(1, :);
sc.path = ft.path_make(ft.to_cm(sc, line), false);
sc.edges_px = {};
Bb = bwboundaries(track, 8, 'noholes'); if ~isempty(Bb), sc.edges_px = {fliplr(Bb{1})}; end
sc.blue_px = C;
% black obstacle: dark, large, inside the box of track + blue squares
hsv = rgb2hsv(rgb);
dark = hsv(:, :, 3) < 50/255 & ~imdilate(track, ones(9));
[ys, xs] = find(track);
P = [xs ys; C];
lo = min(P); hi = max(P); pad = 0.08 * (hi - lo);
bm = false(size(dark));
bm(max(1, round(lo(2) - pad(2))):min(end, round(hi(2) + pad(2))), max(1, round(lo(1) - pad(1))):min(end, round(hi(1) + pad(1)))) = true;
dark = dark & bm;
st = regionprops(dark, 'Area', 'BoundingBox');
st = st([st.Area] >= (10 / s_px) ^ 2);
sc.obstacle_px = [];
if ~isempty(st)
    [~, k] = max([st.Area]); b = st(k).BoundingBox;
    sc.obstacle_px = [b(1) b(2); b(1) + b(3) b(2); b(1) + b(3) b(2) + b(4); b(1) b(2) + b(4)];
end
info.width_px = widthPx; info.fine_deg = fine; info.length_cm = sc.path.L;
sc.info = info;
end

function P = longestSkeletonPath(sk)
ep = find(bwmorph(sk, 'endpoints'));
if isempty(ep), [r, c] = find(sk, 1); ep = sub2ind(size(sk), r, c); end
D0 = bwdistgeodesic(sk, ep(1), 'quasi-euclidean');
D0(isnan(D0)) = -1; [~, a] = max(D0(:));
Da = bwdistgeodesic(sk, a, 'quasi-euclidean');
Dt = Da; Dt(isnan(Dt)) = -1; [L, b] = max(Dt(:));
Db = bwdistgeodesic(sk, b, 'quasi-euclidean');
on = abs(Da + Db - L) < 1.5 & sk;
idx = find(on);
[~, o] = sort(Da(idx));
[r, c] = ind2sub(size(sk), idx(o));
P = [c r];
end

function out = resamplePx(P, step)
d = [0; cumsum(vecnorm(diff(P), 2, 2))];
[d, iu] = unique(d); P = P(iu, :);
si = (0:step:d(end))';
out = [interp1(d, P(:, 1), si), interp1(d, P(:, 2), si)];
end

function out = smoothPolyline(P, win)
k = ones(win, 1) / win;
Pp = [repmat(P(1, :), win, 1); P; repmat(P(end, :), win, 1)];
out = [conv(Pp(:, 1), k, 'same'), conv(Pp(:, 2), k, 'same')];
out = out(win + 1:end - win, :);
out(1, :) = P(1, :); out(end, :) = P(end, :);
end
