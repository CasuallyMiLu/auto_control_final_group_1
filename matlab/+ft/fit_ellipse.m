function E = fit_ellipse(P)
%FT.FIT_ELLIPSE  Direct least-squares ellipse fit (Fitzgibbon 1999) to points P = [x y].
%   E.c centre, E.a / E.b semi-axes, E.theta angle (rad) of the a-axis from +x.
x = P(:, 1); y = P(:, 2);
mx = mean(x); my = mean(y); sx = std(x); sy = std(y); s = max(sx, sy);
x = (x - mx) / s; y = (y - my) / s;
D = [x .^ 2, x .* y, y .^ 2, x, y, ones(size(x))];
S = D' * D;
C = zeros(6); C(1, 3) = 2; C(3, 1) = 2; C(2, 2) = -1;
S1 = S(1:3, 1:3); S2 = S(1:3, 4:6); S3 = S(4:6, 4:6);
T = -S3 \ S2';
M = S1 + S2 * T;
M = [M(3, :) / 2; -M(2, :); M(1, :) / 2];
[V, ~] = eig(M);
cond = 4 * V(1, :) .* V(3, :) - V(2, :) .^ 2;
a1 = V(:, cond > 0); a1 = a1(:, 1);
p = [a1; T * a1];
A = p(1); B = p(2); Cc = p(3); Dd = p(4); Ee = p(5); F = p(6);
% centre
M2 = [2 * A, B; B, 2 * Cc];
c = -M2 \ [Dd; Ee];
% axes from the quadratic form at the centre
F0 = F + (A * c(1) ^ 2 + B * c(1) * c(2) + Cc * c(2) ^ 2 + Dd * c(1) + Ee * c(2));
Q = [A, B / 2; B / 2, Cc];
[R, L] = eig(Q);
ax = sqrt(-F0 ./ diag(L));
E.c = [c(1) * s + mx, c(2) * s + my];
[ax, o] = sort(ax * s, 'descend');
R = R(:, o);
E.a = ax(1); E.b = ax(2);
E.theta = atan2(R(2, 1), R(1, 1));
end
