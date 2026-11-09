function s = path_wrap(p, s)
if p.closed, s = mod(s, p.L); else, s = min(max(s, 0), p.L); end
end
