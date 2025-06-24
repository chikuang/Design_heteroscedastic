% clc; clear;
% === Settings ===
g = @(x) [1; x; x.^2];
lambda = @(x) 2*x + 5;  % Heteroscedastic
N = 501;
Nz = 201;
z_vals = linspace(1,1.2 , Nz);
x_vals = linspace(-1, 1, N)';
p = 3;

% === Step 1: Precompute g(x) and lambda(x) ===
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

% === Step 2: Solve G-optimal design via CVX ===
cvx_begin quiet
    cvx_precision best
    variable w(N, 1)
    variable t
    expression M(p, p)
    M = zeros(p);
    for i = 1:N
        M = M + w(i) * lambda_vals(i) * (g_list{i} * g_list{i}');
    end
    minimize t
    subject to
        sum(w) == 1
        w >= 0
        for i = 1:Nz
            matrix_frac(gz_list{i}, M) <= t
        end
cvx_end

support_idx = find(w > 1e-4);
x_out = round(x_vals(support_idx), 3);
w_out = round(w(support_idx), 3);
disp('Support points and weights:');
disp(table(x_out, w_out, 'VariableNames', {'x', 'weight'}));

% === Step 3: Inverse of M and v(z, ξ) ===
M_inv = inv(M);
v_vals = zeros(Nz, 1);
for i = 1:Nz
    gz = g(z_vals(i));
    v_vals(i) = gz' * M_inv * gz;
end

% === Step 4: A(ξ*) = max_z v(z) ===
v_max = max(v_vals);
tol = 1e-8;
max_indices = find(abs(v_vals - v_max) < tol);
dim_A = length(max_indices);
gz_mat = zeros(p, dim_A);
for j = 1:dim_A
    gz_mat(:, j) = g(z_vals(max_indices(j)));
end

% === Step 3: Identify A(ξ*) = {z ∈ Z | v(z) ≈ max v(z)} ===
tol = 1e-8;
max_val = max(v_vals);
max_indices = find(abs(v_vals - max_val) < tol);
dim_A = length(max_indices);

gz_vec_A = zeros(p, dim_A);
for j = 1:dim_A
    gz_vec_A(:, j) = g(z_vals(max_indices(j)));
end
% % === Step 5: Optimize μ* using fmincon ===
% === Define optimization objective ===
objective = @(mu) max(compute_c_given_mu(mu, g_list, gz_vec_A, lambda_vals, M_inv, max_val));

% === Constraints ===
Aeq = ones(1, dim_A);
beq = 1;
lb = zeros(dim_A,1);
ub = ones(dim_A,1);
mu0 = ones(dim_A,1)/dim_A;

% === Run fmincon ===
options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
mu_star = fmincon(objective, mu0, [], [], Aeq, beq, lb, ub, [], options);

% === Step 6: Compute φ_G(x) and directional derivative ===
phiG = zeros(N,1);
dphi = zeros(N,1);
for i = 1:N
    gx = g_list{i};
    phiG(i) = gx' * M_inv * gx;

    r = zeros(dim_A,1);
    for j = 1:dim_A
        gzz = gz_mat(:, j);
        r(j) = (gx' * M_inv * gzz)^2;
    end
    dphi(i) = lambda_vals(i) * sum(mu_star .* r) - v_max;
end
PhiG = max(phiG);

% === Step 7: Plot ===
figure;
plot(x_vals, dphi, 'r-', 'LineWidth', 2); hold on;

% Add support points
plot(x_vals(support_idx), dphi(support_idx), 'ko', ...
     'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'Support Points');

% Add y=0 line
yline(0, 'k--');

% Axis labels and title
ylabel('Directional Derivative', 'Color', 'r', 'FontSize', 14)
xlabel('x', 'FontSize', 14);
title(sprintf(['Equivalence Theorem for G-optimal Design (Heteroscedastic)\n' ...
               'Extrapolation using A(\\xi^*),  |A| = %d'], dim_A), ...
               'FontSize', 14, 'FontWeight', 'bold');

% Set axis tick label font size
set(gca, 'FontSize', 14);
grid on;

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