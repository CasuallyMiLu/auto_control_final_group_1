function [sc, changedDeg] = set_up_vector(sc, upImg)
%FT.SET_UP_VECTOR  re-orient the world frame so +Y points along the measured image direction upImg (snapped to
%   the nearest of the 4 multiples of 90 deg of the calibrated up vector).
upImg = upImg / norm(upImg);
best = -inf;
for k = 0:3
    c = cosd(90 * k); s = sind(90 * k);
    cand = [c * sc.up(1) - s * sc.up(2), s * sc.up(1) + c * sc.up(2)];
    if dot(cand, upImg) > best, best = dot(cand, upImg); kk = k; cbest = cand; end
end
changedDeg = 90 * kk;
if kk == 0, return; end
R = [cosd(-90 * kk) -sind(-90 * kk); sind(-90 * kk) cosd(-90 * kk)];
sc.up = cbest;
sc.A = R * sc.A;
sc.path = ft.path_make(sc.path.pts * R', sc.path.closed);
end
