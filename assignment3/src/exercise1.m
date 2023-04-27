clear all;
close all;
clc;

rng("default");


%% Solve problem

cvx_begin
    variable x;
    dual variable lambda;
    minimize(objective_function(x));
    subject to
        lambda : (x-2) .* (x-4) <= 0;
cvx_end
x_optimal = x;
y_optimal = cvx_optval;
lambda_optimal = lambda;


%% Plot objective function and Lagrangian

figure;
hold on;

x_min = 1;
x_max = 5;
x = sort([linspace(x_min, x_max) 2 4]);
x_feasible = x((x-2) .* (x-4) <= 0);

lambda = [0, 2, 3, 4, 5];
n = length(lambda);
colors = hot(20);
colors = colors(end/2:1:end,:);     % go from red (middle) to white (right)

for idx = 1:n
    plot( ...
        x, lagrangian(x, lambda(idx)), ...
        'Color', colors(idx,:), ...
        'DisplayName', sprintf('\\lambda=%d',lambda(idx)) ...
    );
end

area( ...
    [x_feasible(1), x_feasible(end)], [100, 100], ...
    'FaceColor', [.7, .7, .7], ...
    'FaceAlpha', .2, ...
    'DisplayName', 'feasible region' ...
);
plot( ...
    x_feasible, objective_function(x_feasible), ...
    'Color', [0, 0.5, 0], ...
    'LineWidth', 2, ...
    'DisplayName', 'feasible values' ...
    );
yline(y_optimal, 'b--', 'DisplayName', 'optimal value');

legend;
xticks(x_min:x_max);
xlabel('x');
ylabel('L(x,\lambda)');
ylim([0, 40]);


%% Plot Lagrangian dual function

figure;

lambda_min = 0;
lambda_max = 5;
lambda = linspace(lambda_min, lambda_max);

plot(lambda, lagrangian_dual_function(lambda));
xlabel('\lambda');
ylabel('g(\lambda)');


%% Functions

function y = objective_function(x)
    y = x.^2 + 1;
end

function y = lagrangian(x,lambda)
    y = objective_function(x) + lambda .* (x-2) .* (x-4);
end

function y = lagrangian_dual_function(lambda)
    y = -9*lambda.^2 ./ (1+lambda) + 8*lambda + 1;
end