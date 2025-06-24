% === Settings ===
g = @(x) [1; x; x.^2];
lambda = @(x) 2*x + 5;
% lambda = @(x) 1;
N = 201;
x_vals = linspace(-1, 1, N)';
p = 3;
n = length(x_vals);

% Precompute g(x) and lambda(x)
g_list = cell(n,1);
lambda_vals = zeros(n,1);
for i = 1:n
    g_list{i} = g(x_vals(i));
    lambda_vals(i) = lambda(x_vals(i));
end

% === Solve G-optimal design via CVX ===
cvx_begin quiet
    cvx_precision best
    % cvx_solver mosek
    variable w(n)
    variable t
    expression M(p,p)
    M = zeros(p);
    for i = 1:n
        M = M + w(i) * lambda_vals(i) * (g_list{i} * g_list{i}');
    end
    minimize t
    subject to
        sum(w) == 1
        w >= 0
        M >= 1e-6 * eye(p)
        for i = 1:n
            matrix_frac(g_list{i}, M) <= t
        end
cvx_end

% % === Output support points ===
support_idx = find(w > 1e-4);
disp('Support points and weights:');
x_out = round(x_vals(support_idx), 3);
w_out = round(w(support_idx), 3);
disp(table(x_out, w_out, 'VariableNames', {'x', 'weight'}));

% % === Compute phi_G and directional derivative ===
% M_star = zeros(p);
% for i = 1:n
%     M_star = M_star + w(i) * lambda_vals(i) * (g_list{i} * g_list{i}');
% end
% M_inv = inv(M_star);
% 
% phiG = zeros(n,1);
% dphi = zeros(n,1);
% for i = 1:n
%     gx = g_list{i};
%     phiG(i) = gx' * M_inv * gx;
%     dphi(i) = -lambda_vals(i) * phiG(i)^2 + phiG(i);
% end
% PhiG = max(phiG);
% 

% 
% % === Plot ===
% figure;
% yyaxis left
% plot(x_vals, phiG, 'b-', 'LineWidth', 2); hold on;
% yline(PhiG, 'b--', 'LineWidth', 1.2);
% % Support points overlayed as black circles
% plot(x_vals(support_idx), phiG(support_idx), 'ko', 'MarkerSize', 7, 'LineWidth', 1.5, 'DisplayName', 'Support Points');
% ylabel('\phi_G(x)')
% 
% yyaxis right
% plot(x_vals, dphi, 'r-', 'LineWidth', 2);
% yline(0, 'k--');
% ylabel('Directional Derivative')
% 
% xlabel('x')
% legend({'\phi_G(x)', '\Phi_G', 'Support Points', 'd\phi_G/d\alpha'}, 'Location', 'Best')
% title('Equivalence Theorem for G-optimal Design (Heteroscedastic)')
% grid on