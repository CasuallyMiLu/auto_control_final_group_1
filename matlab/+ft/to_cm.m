function xy = to_cm(sc, uv)
%FT.TO_CM  image px (full resolution, [u v]) -> wall cm [X Y]
xy = (uv - sc.origin) * sc.A';
end
