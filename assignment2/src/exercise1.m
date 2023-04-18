clear all;
close all;
clc;


%% Prepare problem parameters

H = [
    2 1 2;
    1 2 1;
    2 1 2;
];
c = [0; -2; 2];
n = length(c);
A = [eye(n); -1 * eye(n)];
b = -1 * ones(2*n, 1);


%% Solve linearly constrained convex problem

% The default solver struggles and stops with status Solved
% The MOSEK solver correctly detects the infeasibility
% See https://ask.cvxr.com/t/constraint-violated-in-linear-programming/11298/4
cvx_solver mosek

cvx_begin
    variable x(n);

    dual variable lambda;
    dual variable mu;

    minimize(1/2*x'*H*x + c'*x);
    subject to
        A*x - b >= 0;
        lambda : x + 1 >= 0;
        mu : -x + 1 >= 0;
cvx_end

disp('Optimal point:');
disp(x);


%% Verify if a point is optimal by using KKT conditions
% x* is optimal <=> there exists lambda satisfying the KKT conditions
% this is implemented as a feasibility problem

x_star = [0; 1; -1];

cvx_begin
    variable lambda(2*n);
    minimize(0);
    subject to
        A * x_star - b >= 0;                % (i)
        H * x_star + c - A'*lambda == 0;    % (ii)
        lambda >= 0;                        % (iii)
        lambda .* (A*x_star - b) == 0;      % (iv)
cvx_end

disp('x_star =');
disp(x_star);
fprintf('x_star is ');
if strcmp(cvx_status,'Infeasible')
    fprintf('NOT ');
end
fprintf('optimal\n');