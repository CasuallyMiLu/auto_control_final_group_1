function [q, t] = path_point(p, s)
%FT.PATH_POINT  point and unit tangent at arclength s
s = ft.path_wrap(p, s);
i = find(p.s <= s, 1, 'last'); i = min(max(i, 1), numel(p.seglen));
q = p.pts(i, :) + p.tan(i, :) * (s - p.s(i));
t = p.tan(i, :);
end
