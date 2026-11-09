function [lo, hi] = bounds_cm(sc, margin)
%FT.BOUNDS_CM  working area in wall cm: track + blue squares, grown by margin. The black obstacle is NOT used for
%   anything in flight (theme 2: the drone only follows the laser; avoiding the obstacle is the operator's job).
P = sc.path.pts;
if ~isempty(sc.blue_px), P = [P; ft.to_cm(sc, sc.blue_px)]; end
lo = min(P, [], 1) - margin; hi = max(P, [], 1) + margin;
end
