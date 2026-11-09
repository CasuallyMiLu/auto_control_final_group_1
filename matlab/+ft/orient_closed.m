function sc = orient_closed(sc, direction, startDeg)
%FT.ORIENT_CLOSED  travel direction (ccw/cw as seen from the camera) and start vertex (direction from the centre, deg)
P = sc.path.pts;
area = 0.5 * sum(P(:, 1) .* circshift(P(:, 2), -1) - circshift(P(:, 1), -1) .* P(:, 2));
if (area > 0) ~= strcmp(direction, 'ccw'), P = flipud(P); end
c = 0.5 * (min(P) + max(P));
ang = atan2(P(:, 2) - c(2), P(:, 1) - c(1));
[~, i0] = min(abs(mod(ang - deg2rad(startDeg) + pi, 2 * pi) - pi));
sc.path = ft.path_make(circshift(P, -(i0 - 1), 1), true);
sc.info.direction = direction; sc.info.start_angle_deg = startDeg;
end
