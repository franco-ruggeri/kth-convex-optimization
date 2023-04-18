clear all
close all
clc

rng("default");


%% Generate data
m = 200;
k0 = 4;
l0 = 1;
x = randn(m,1);
y = k0*x + l0 + randn(m,1);
y(end)=3*y(end);    % outlier

fig = figure;
hold on;
legend();
scatter(x,y,'DisplayName','data');


%% Solve the inf-norm linear program

cvx_begin
    variable t;
    variable k;
    variable l;

    dual variable lambda;
    dual variable mu;

    minimize(t);
    subject to
        lambda : k*x + l - y + t >= 0;
        mu : -k*x - l + y + t >= 0;
cvx_end

disp('Inf-norm approximation');
fprintf('|k - k0| = %f\n', abs(k - k0));
fprintf('|l - l0| = %f\n', abs(l - l0));

y_pred = k*x + l;
plot(x,y_pred,'DisplayName','inf-norm');


%% Solve the 1-norm linear program

cvx_begin
    variable t(m);
    variable k;
    variable l;

    dual variable lambda;
    dual variable mu;

    minimize(sum(t));
    subject to
        lambda : k*x + l - y + t >= 0;
        mu : -k*x - l + y + t >= 0;
cvx_end

disp('L1-norm approximation');
fprintf('|k - k0| = %f\n', abs(k - k0));
fprintf('|l - l0| = %f\n', abs(l - l0));

y_pred = k*x + l;
plot(x,y_pred,'DisplayName','1-norm');