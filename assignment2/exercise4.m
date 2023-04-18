clear all
close all
clc

rng("default");


%% Prepare problem parameters

A = [
    1 2 0 1;
    0 0 3 1;
    0 3 1 1;
    2 1 2 5;
    1 0 3 2;
];
[m,n] = size(A);

c_max = 100*ones(m,1);
p = [3 2 7 6]';
p_disc = [2 1 4 2]';
q = [4 10 5 10]';


%% Solve the linear program

cvx_begin
    variable t(n);
    variable x(n);

    minimize(sum(t));
    subject to
        x >= 0;
        A*x - c_max <= 0;
        t + p .* x >= 0;
        t + p .* q + p_disc .* (x - q) >= 0;
cvx_end

revenue = -cvx_optval;
revenue_activities = min(p .* x, p .* q + p_disc .* (x-q));
avg_price_per_unit = revenue_activities ./ x;

fprintf('Total revenue: %f\n\n', revenue);
disp('Optimal activity levels');
disp(x);
disp('Revenue of each activity');
disp(revenue_activities);
disp('Average price per unit');
disp(avg_price_per_unit);