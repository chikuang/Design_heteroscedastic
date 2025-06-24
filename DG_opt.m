clc; clear;

% === Settings ===
g = @(x) [1; x; x.^2];
lambda = @(x) 2*x + 5;   % Heteroscedastic
N = 101;
Nz = 101;
x_vals = linspace(-1, 1, N)';
z_vals = linspace(-1, 1, Nz)';
q = 3;
alpha = 0.4;

% === Step 1: Precompute g(x), g(z), and lambda(x) ===
g_list = cell(N, 1);
lambda_vals = zeros(N, 1);
for i = 1:N
    g_list{i} = g(x_vals(i));
    lambda_vals(i) = lambda(x_vals(i));
end

gz_list = cell(Nz, 1);
for j = 1:Nz
    gz_list{j} = g(z_vals(j));
end

% === Step 2: Solve compound D + G optimal design via CVX ===
cvx_begin quiet
    cvx_precision best
    variable w(N, 1)
    variable t
    expression M(q, q)
    M = zeros(q);
    for i = 1:N
        M = M + w(i) * lambda_vals(i) * (g_list{i} * g_list{i}');
    end
    minimize alpha * t - (1 - alpha) * log_det(M)
    subject to
        sum(w) == 1
        w >= 0
        for j = 1:Nz
            matrix_frac(gz_list{j}, M) <= t
        end
cvx_end

% === Step 3: Extract support points ===
support_idx = find(w > 1e-4);
x_out = round(x_vals(support_idx), 3);
w_out = round(w(support_idx), 3);
disp('Support points and weights:');
disp(table(x_out, w_out, 'VariableNames', {'x', 'weight'}));

% === Step 4: Compute Fisher Info matrix and inverse ===
M_val = zeros(q);
for i = 1:N
    M_val = M_val + w(i) * lambda_vals(i) * (g_list{i} * g_list{i}');
end
M_inv = inv(M_val);

% === Step 5: Compute phi_D and phi_G ===
phiD = zeros(N, 1);
phiG = zeros(N, 1);
for i = 1:N
    gx = g_list{i};
    phiD(i) = lambda_vals(i) * (gx' * M_inv * gx) - q;
end

v_vals = zeros(Nz, 1);
for j = 1:Nz
    gz = gz_list{j};
    v_vals(j) = gz' * M_inv * gz;
end
v_max = max(v_vals);
tol = 1e-8;
max_indices = find(abs(v_vals - v_max) < tol);
dim_A = length(max_indices);

gz_mat = zeros(q, dim_A);
for j = 1:dim_A
    gz_mat(:, j) = gz_list{max_indices(j)};
end

% === Step 6: Solve for mu_star ===
objective = @(mu) max(compute_c_given_mu(mu, g_list, gz_mat, lambda_vals, M_inv, v_max));
Aeq = ones(1, dim_A); beq = 1;
lb = zeros(dim_A,1); ub = ones(dim_A,1);
mu0 = ones(dim_A,1) / dim_A;
options = optimoptions('fmincon','Display','off','Algorithm','sqp');
mu_star = fmincon(objective, mu0, [], [], Aeq, beq, lb, ub, [], options);

% === Step 7: Compute phi_G (directional derivative form) ===
dphiG = zeros(N, 1);
for i = 1:N
    gx = g_list{i};
    r = zeros(dim_A, 1);
    for j = 1:dim_A
        gzz = gz_mat(:, j);
        r(j) = (gx' * M_inv * gzz)^2;
    end
    dphiG(i) = lambda_vals(i) * sum(mu_star .* r) - v_max;
end

% === Step 8: Combine into phi_mix ===
phiMix = alpha * dphiG + (1 - alpha) * phiD;

% === Step 9: Plot directional derivatives ===
fontsize = 18;
figure;
subplot(1, 3, 1);
plot(x_vals, phiD, 'k-', 'LineWidth', 1.5); hold on;
plot(x_vals(support_idx), phiD(support_idx), 'ro', 'MarkerSize', 6);
yline(0, 'g-', 'LineWidth', 2);
xlabel('x', 'FontSize', fontsize);
ylabel('\phi_D(x)', 'FontSize', fontsize);
title('D-opt Directional Derivative', 'FontSize', fontsize, 'FontWeight', 'bold');
set(gca, 'FontSize', fontsize); grid on;

subplot(1, 3, 2);
plot(x_vals, dphiG, 'k-', 'LineWidth', 1.5); hold on;
plot(x_vals(support_idx), dphiG(support_idx), 'ro', 'MarkerSize', 6);
yline(0, 'g-', 'LineWidth', 2);
xlabel('x', 'FontSize', fontsize);
ylabel('\phi_G(x)', 'FontSize', fontsize);
title('G-opt Directional Derivative', 'FontSize', fontsize, 'FontWeight', 'bold');
set(gca, 'FontSize', fontsize); grid on;

subplot(1, 3, 3);
plot(x_vals, phiMix, 'k-', 'LineWidth', 1.5); hold on;
plot(x_vals(support_idx), phiMix(support_idx), 'ro', 'MarkerSize', 6);
yline(0, 'g-', 'LineWidth', 2);
xlabel('x', 'FontSize', fontsize);
ylabel('\phi_{\mathrm{mix}}(x)', 'FontSize', fontsize);
title(sprintf('Compound Directional Derivative (\\alpha = %.2f)', alpha), ...
    'FontSize', fontsize, 'FontWeight', 'bold');
set(gca, 'FontSize', fontsize); grid on;

% === Local function ===
function c = compute_c_given_mu(mu, g_list, gz_mat, lambda_vals, M_inv, v_max)
    N = length(g_list);
    dim_A = size(gz_mat, 2);
    c = zeros(N, 1);
    for i = 1:N
        gx = g_list{i};
        r = zeros(dim_A, 1);
        for j = 1:dim_A
            gzz = gz_mat(:, j);
            r(j) = (gx' * M_inv * gzz)^2;
        end
        c(i) = lambda_vals(i) * sum(mu .* r) - v_max;
    end
end