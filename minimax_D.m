clc; clear;

%=== Parameter settings ===
p = 2;
N = 501;
x_vals = linspace(-1, 4, N)';
Theta0 = linspace(0, 2.5, 31);   % a values
Theta1 = linspace(1, 3, 31);     % b values
theta_grid = combvec(Theta0, Theta1)';  % All (a,b) combinations

num_theta = size(theta_grid, 1);

%=== Precompute Fisher information matrices for each θ and x ===
I_list = cell(num_theta, 1);
for k = 1:num_theta
    a = theta_grid(k, 1);
    b = theta_grid(k, 2);
    I_k = zeros(p, p, N);

    for i = 1:N
        x = x_vals(i);
        eta = b * (x - a);
        pval = 1 / (1 + exp(-eta));
        d = pval * (1 - pval);  % derivative

        % Fisher information contribution at x
        I_k(:, :, i) = d * [b^2, -b * (x - a); -b * (x - a), (x - a)^2];
    end
    I_list{k} = I_k;
end

%=== CVX Optimization: minimize the worst-case -logdet ===
cvx_begin quiet
    cvx_precision best
    variable w(N)
    variable t

    minimize t
    subject to
        w >= 0
        sum(w) == 1

        for k = 1:num_theta
            M = zeros(p);
            for i = 1:N
                M = M + w(i) * I_list{k}(:, :, i);
            end
            t >= -log_det(M);  % valid DCP constraint: affine >= concave
        end
cvx_end


%=== Output support points and weights ===
support_idx = find(w > 1e-4);
x_out = round(x_vals(support_idx), 4);
w_out = round(w(support_idx), 4);

disp('Minimax D-optimal design:');
disp(table(x_out, w_out, 'VariableNames', {'x', 'weight'}));