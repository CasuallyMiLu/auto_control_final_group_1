function v = path_speed_at(p, s)
s = ft.path_wrap(p, s);
i = find(p.s <= s, 1, 'last'); i = min(max(i, 1), numel(p.vmax));
v = p.vmax(i);
end
