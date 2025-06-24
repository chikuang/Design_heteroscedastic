% clc; clear;
% === Settings ===
g = @(x) [1; x; x.^2];
% lambda = @(x) 1;  % Homoscedastic
lambda = @(x) 2*x + 5;  % Heteroscedastic
N = 501;
x_vals = linspace(-1, 1, N)';
q = 3;
alpha = 0.5;

% === Step 1: Precompute g(x) and lambda(x) ===
g_list = cell(N, 1);
lambda_vals = zeros(N, 1);
for i = 1:N
    g_list{i} = g(x_vals(i));
    lambda_vals(i) = lambda(x_vals(i));
end

% === Step 2: Solve G-optimal design via CVX ===
cvx_begin quiet
    cvx_precision best
    variable w(N, 1)
    expression M(q, q)
    M = zeros(q);
    for i = 1:N
        M = M + w(i) * lambda_vals(i) * (g_list{i} * g_list{i}');
    end
    minimize alpha * trace_inv(M) - (1-alpha) * log_det(M);
    subject to
        sum(w) == 1;
        w >= 0;
cvx_end

support_idx = find(w > 1e-4);
x_out = round(x_vals(support_idx), 3);
w_out = round(w(support_idx), 3);
disp('Support points and weights:');
disp(table(x_out, w_out, 'VariableNames', {'x', 'weight'}));


%% Equivalence theorem
% === Compute Fisher information and its inverse ===
M_val = zeros(q);
for i = 1:N
    M_val = M_val + w(i) * lambda_vals(i) * (g_list{i} * g_list{i}');
end
M_inv = inv(M_val);
M_inv2 = M_inv * M_inv;

% === Directional Derivatives ===
phiD = zeros(N, 1);     % D-opt: -g^T M^{-1} g
phiA = zeros(N, 1);     % A-opt: g^T M^{-2} g
phiMix = zeros(N, 1);   % Weighted combination

for i = 1:N
    gx = g_list{i};
    phiD(i) = lambda_vals(i) * gx' * M_inv * gx - q;
    phiA(i) = lambda_vals(i)  * gx' * M_inv2 * gx - trace(M_inv);
    phiMix(i) = alpha * phiA(i) + (1 - alpha) * phiD(i);
end

% === Create 1x3 subplot for phiD, phiA, phiMix + Δ ===
figure;

% --- Subplot 1: D-optimality ---
fontsize = 14;

% --- Subplot 1: D-optimality ---
subplot(1, 3, 1);
plot(x_vals, phiD, 'b-', 'LineWidth', 1.5); hold on;
plot(x_out, phiD(support_idx), 'bo', 'MarkerSize', 6);
yline(0, 'g-', 'LineWidth', 1.2);  % solid green
xlabel('x', 'FontSize', fontsize);
ylabel('\phi_D(x)', 'FontSize', fontsize);
title('D-opt Directional Derivative', 'FontSize', fontsize, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fontsize);

% --- Subplot 2: A-optimality ---
subplot(1, 3, 2);
plot(x_vals, phiA, 'r--', 'LineWidth', 1.5); hold on;
plot(x_out, phiA(support_idx), 'ro', 'MarkerSize', 6);
yline(0, 'g-', 'LineWidth', 1.2);  % solid green
xlabel('x', 'FontSize', fontsize);
ylabel('\phi_A(x)', 'FontSize', fontsize);
title('A-opt Directional Derivative', 'FontSize', fontsize, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fontsize);

% --- Subplot 3: Weighted mix and Δ(x) ---
subplot(1, 3, 3);
plot(x_vals, phiMix, 'k-.', 'LineWidth', 2); hold on;
plot(x_out, phiMix(support_idx), 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'k');
yline(0, 'g-', 'LineWidth', 1.2);  % solid green
xlabel('x', 'FontSize', fontsize);
ylabel('\phi_{mix}(x), \Delta(x)', 'FontSize', fontsize);
title(sprintf('Weighted A+D (\\alpha = %.2f)', alpha), ...
    'FontSize', fontsize, 'FontWeight', 'bold');
legend('\phi_{mix}(x)', '\Delta(x)', 'Location', 'best');
grid on;
set(gca, 'FontSize', fontsize);