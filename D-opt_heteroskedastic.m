% === 設定 ===
g = @(x) [1; x; x^2];             
lambda = @(x) 2 * x + 5;
% lambda = @(x) 1
x_vals = linspace(-1, 1, 101);    
n = length(x_vals);
q = 3;

g_list = cell(n, 1);
lambda_vals = zeros(n, 1);
for i = 1:n
    g_list{i} = g(x_vals(i));
    lambda_vals(i) = lambda(x_vals(i));
end

% === 使用 CVX 求解 D-optimal design ===
cvx_solver sdpt3
cvx_begin quiet
    cvx_precision best
    variable w(n)
    M = zeros(q, q);
    for i = 1:n
        gx = g_list{i};
        M = M + w(i) * lambda_vals(i) * (gx * gx');
    end
    minimize -log_det(M)
    subject to
        w >= 0
        sum(w) == 1
cvx_end

% === 計算 d_D(x) ===
M_val = zeros(q, q);
for i = 1:n
    M_val = M_val + w(i) * lambda_vals(i) * (g_list{i} * g_list{i}');
end
Minv = inv(M_val);

dD = zeros(n, 1);
for i = 1:n
    gx = g_list{i};
    dD(i) = gx' * Minv * gx * lambda_vals(i) - q;
end

% === 畫圖 ===
support_idx = find(w > 1e-4);
support_x = x_vals(support_idx);
support_w = w(support_idx);
design = [support_x; support_w']

figure;
plot(x_vals, dD, 'b-', 'LineWidth', 2); hold on;
yline(0, 'k--', 'LineWidth', 1.5);
for j = 1:length(support_idx)
    xj = x_vals(support_idx(j));
    line([xj xj], ylim, 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.2);
end
xlabel('x'); ylabel('d_D(x)');
title('D-optimality Equivalence Function with Heteroscedastic Errors');
legend('d_D(x)', 'Zero line', 'Support points', 'Location', 'Best');
grid on;