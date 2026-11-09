function uv = to_px(sc, xy)
%FT.TO_PX  wall cm [X Y] -> image px (full resolution)
uv = xy / sc.A' + sc.origin;
end
