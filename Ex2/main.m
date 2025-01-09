clc; clear all; close all;

We = 1361; % W/m^2
c = 3e8; % m/s
sigma = 20; % g/m^2

mu_s = 1.327e11; % km^3/s^2



% Adimensional parameters D, T. D^3/T^2 = mu_s
D = 1.496e8; % km 1 
T = sqrt(D^3/mu_s); % s

beta = (2*We/(sigma*c))*D^2;
beta_adim = beta*T^2/D^3;

disp(beta_adim)

% Earth and Mercury orbits for plotting
theta_span = linspace(0, 2*pi, 1000);
rE = ones(1, 1000);
rM = 0.3871*ones(1, 1000);

xE = rE.*cos(theta_span);
yE = rE.*sin(theta_span);

xM = rM.*cos(theta_span);
yM = rM.*sin(theta_span);


% Planetary velocities
vE = sqrt(mu_s/D); % km/s
vM = sqrt(mu_s/(0.3871*D)); % km/s


% Initial conditions (adimensional)
r0_adim = 1; % AU
theta0_adim = 0; % rad 
vr0_adim = 0; % AU/s
vt0_adim = 1; % AU/s

% Imposed final conditions
rf_adim = 0.3871; % AU (Mercury orbit)
% thetaf free
vrf_adim = 0; % AU/s
vtf_adim= vM*T/D; % 

month_s = 24*60*60*30; % s  1 month in seconds
% Guess transfer time
tf_lb = month_s*18; % lower bound
tf_ub = month_s*20; % upper bound

tf_ig = 16*month_s; % initial guess

% Adimensionalize bounds and initial guess
tf_lb_adim = tf_lb/T;
tf_ub_adim = tf_ub/T;
tf_ig_adim = tf_ig/T;


% ode options
odeOpts = odeset('RelTol', 3e-14, 'AbsTol', 1e-14);


% Uknwown quantities are
% xx = [l_r0, l_vr0, l_vt0, tf]

% We can restrict the search interval for the unknown costates to be
% between -1 and 1 due to adjoint scalability

% Initial guess 
lb = [-1, -1, -1, tf_lb_adim];
ub = [1, 1, 1, tf_ub_adim];



% PSO options
pso_opts = optimoptions('particleswarm', 'Display', 'iter', 'SwarmSize', 50, 'MaxIterations', 100, 'FunctionTolerance', 1e-12, 'MaxStallIterations', 1e+3);

% Run PSO
[xxOpt, fval] = particleswarm(@cost, 4, lb, ub, pso_opts);
fprintf("--------------------\n")
fprintf("PSO results\n")

fprintf("Optimal costates: ig = [ %f, %f, %f, %f ]\n", xxOpt(1), xxOpt(2), xxOpt(3), xxOpt(4))

fprintf("In days: %f\n", xxOpt(4)*T/(24*60*60))


fprintf("Optimal cost: %f\n", fval)

% Propagate orbit and plot
% Extract optimal costates
l_r0 = xxOpt(1);
l_vr0 = xxOpt(2);
l_vt0 = xxOpt(3);
tf = xxOpt(4);

% Propagate dynamics following costate initial conditions
S0 = [r0_adim, theta0_adim, vr0_adim, vt0_adim];
L0 = [l_r0, 0, l_vr0, l_vt0];
X0 = [S0, L0];

% Propagate dynamics

[t, X] = ode113(@(t, X) fdyn(t, X, beta_adim), [0, tf], X0,odeOpts);
S = X(:, 1:4);
L = X(:, 5:8);

figure
hold on
title("PSO optimal trajectory")
plot(xE, yE, 'b')
plot(xM, yM, 'r')
plot(0, 0, 'k.')
axis("equal")


% Plot trajectory
plot(X(:, 1).*cos(X(:, 2)), X(:, 1).*sin(X(:, 2)), 'k', 'LineWidth', 2)


% Plot optimal control angle history
alpha = zeros(length(t), 1);
for i = 1:length(t)
    S = X(i, 1:4);
    L = X(i, 5:8);
    alpha(i) = atan((-3*L(3) + sqrt(9*L(3)^2 +  8*L(4)^2))/(4*L(4)));

end
figure

plot(t, alpha*180/pi)
xlabel("Time [s]")
ylabel("Optimal control angle [deg]")
title("Optimal control angle history - PSO")





fprintf("--------------------\n")
options_fmincon = optimoptions('fmincon', "Algorithm", "sqp", 'Display', 'iter', ...
            'OptimalityTolerance',1e-16,'MaxIterations',1e+4,'StepTolerance',1e-18, ...
            'MaxFunctionEvaluations',1e+4, 'TolCon', 1e-6);

[xOpt] = fmincon(@J, xxOpt, [], [], [], [], lb, ub, @constr, options_fmincon);

% options_fminsearch = optimset('Display', 'iter', 'TolFun', 1e-12, ...
%                                 'TolX', 1e-12, 'MaxFunEvals', 1e+4, 'MaxIter', 1e+4);
% [xOpt, Jopt] = fminsearch(@cost, xxOpt, options_fminsearch);

fprintf("--------------------\n")
fprintf("fmincon results\n")
fprintf("Initial guess: ig = [ %f, %f, %f, %f ]\n", xxOpt(1), xxOpt(2), xxOpt(3), xxOpt(4))
fprintf("Optimal costates: ig = [ %f, %f, %f, %f ]\n", xOpt(1), xOpt(2), xOpt(3), xOpt(4))

fprintf("Optimal cost: %f\n", J(xOpt))
fprintf("In days: %f\n", xOpt(4)*T/(24*60*60))



% Extract optimal costates
l_r0 = xOpt(1);
l_vr0 = xOpt(2);
l_vt0 = xOpt(3);
tf = xOpt(4);

% Propagate dynamics following costate initial conditions
S0 = [r0_adim, theta0_adim, vr0_adim, vt0_adim];
L0 = [l_r0, 0, l_vr0, l_vt0];
X0 = [S0, L0];

% Propagate dynamics

[t, X] = ode113(@(t, X) fdyn(t, X, beta_adim), [0, tf], X0,odeOpts);
S = X(:, 1:4);
L = X(:, 5:8);




figure
hold on
title("fmincon optimal trajectory")
plot(xE, yE, 'b')
plot(xM, yM, 'r')
plot(0, 0, 'k.')
axis("equal")


% Plot trajectory
plot(X(:, 1).*cos(X(:, 2)), X(:, 1).*sin(X(:, 2)), 'k', 'LineWidth', 2)


% Plot optimal control angle history
alpha = zeros(length(t), 1);
for i = 1:length(t)
    S = X(i, 1:4);
    L = X(i, 5:8);
    alpha(i) = atan((-3*L(3) + sqrt(9*L(3)^2 +  8*L(4)^2))/(4*L(4)));
    

end
figure
plot(t, alpha*180/pi)
xlabel("Time [s]")
ylabel("Optimal control angle [deg]")
title("Optimal control angle history - fmincon")




% Constraint function on final time and hamiltonian
function [ineq, eq] = constr(xx)
    % xx = [l_r0, l_vr0, l_vt0, tf]
    % State = [r, theta, vr, vt]
    % Costate = [lambda_r, lambda_t, lambda_vr, lambda_vt]
    We = 1361; % W/m^2
    c = 3e8; % m/s
    sigma = 20; % kg/m^2

    mu_s = 1.327e11; % km^3/s^2



    % Adimensional parameters D, T. D^3/T^2 = mu_s
    D = 1.496e8; % km 1 AU
    T = sqrt(D^3/mu_s); % s

    beta = (2*We/(sigma*c))*D^2;
    beta_adim = beta*T^2/D^3;


    % Planetary velocities
    vE = sqrt(mu_s/D); % km/s
    vM = sqrt(mu_s/(0.3871*D)); % km/s


    % Propagate dynamics following costate initial conditions
    
    % Initial conditions (adimensional)
    r0_adim = 1; % AU
    theta0_adim = 0; % rad 
    vr0_adim = 0; % AU/s
    vt0_adim = 1; % AU/s

    
    S0 = [r0_adim, theta0_adim, vr0_adim, vt0_adim];
    L0 = [xx(1), 0, xx(2), xx(3)];
    X0 = [S0, L0];
    % Imposed final conditions
    rf_adim = 0.3871; % AU (Mercury orbit)
    % thetaf free
    vrf_adim = 0; % AU/s
    vtf_adim = vM*T/D; %

    Yf = [rf_adim, vrf_adim, vtf_adim];

    tf = xx(4);

    % Propagate dynamics
    odeOpts = odeset('RelTol', 3e-14, 'AbsTol', 1e-14);

    [t, X] = ode113(@(t, X) fdyn(t, X, beta_adim), [0, tf], X0, odeOpts);

    % Extract final state
    Sf = X(end, 1:4);
    Lf = X(end, 5:8);
    alpha = atan((-3*Lf(3) + sqrt(9*Lf(3)^2 +  8*Lf(4)^2))/(4*Lf(4)));
    % Compute Hamiltonian
    Hf = Lf(1)*Sf(3) + Lf(2)*Sf(4)/Sf(1) + Lf(3)*(Sf(4)^2/Sf(1) - 1/Sf(1)^2 + beta_adim/Sf(1)^2*cos(alpha)^3) + ...
        Lf(4)*(-Sf(3)*Sf(4)/Sf(1) + beta_adim/Sf(1)^2*cos(alpha)^2*sin(alpha)) - 1;
    
    % Compute constraints
    ineq = -Hf;
    eq = [abs(Sf(1) - Yf(1)), abs(Sf(3) - Yf(2)), abs(Sf(4) - Yf(3))];



end

function [tf] = J(xx)
    tf = xx(4);
end

function [C] = cost(xx)
    % Cost function is the final time
    % xx = [l_r0, l_vr0, l_vt0, tf]
    % State = [r, theta, vr, vt]
    % Costate = [lambda_r, lambda_t, lambda_vr, lambda_vt]
    We = 1361; % W/m^2
    c = 3e8; % m/s
    sigma = 20; % kg/m^2

    mu_s = 1.327e11; % km^3/s^2



    % Adimensional parameters D, T. D^3/T^2 = mu_s
    D = 1.496e8; % km 1 AU
    T = sqrt(D^3/mu_s); % s

    beta = (2*We/(sigma*c))*D^2;
    beta_adim = beta*T^2/D^3;


    % Planetary velocities
    vE = sqrt(mu_s/D); % km/s
    vM = sqrt(mu_s/(0.3871*D)); % km/s


    % Propagate dynamics following costate initial conditions
    
    % Initial conditions (adimensional)
    r0_adim = 1; % AU
    theta0_adim = 0; % rad 
    vr0_adim = 0; % AU/s
    vt0_adim = 1; % AU/s

    
    S0 = [r0_adim, theta0_adim, vr0_adim, vt0_adim];
    L0 = [xx(1), 0, xx(2), xx(3)];
    X0 = [S0, L0];
    % Imposed final conditions
    rf_adim = 0.3871; % AU (Mercury orbit)
    % thetaf free
    vrf_adim = 0; % AU/s
    vtf_adim = vM*T/D; %

    Yf = [rf_adim, vrf_adim, vtf_adim];

    tf = xx(4);

    % Propagate dynamics
    odeOpts = odeset('RelTol', 3e-14, 'AbsTol', 1e-14);

    [t, X] = ode113(@(t, X) fdyn(t, X, beta_adim), [0, tf], X0, odeOpts);

    % Extract final state
    Sf = X(end, 1:4);

   
    
    % Compute cost
    C = tf/(16*1e3) + (abs(Sf(1) - Yf(1)) + abs(Sf(3) - Yf(2)) + abs(Sf(4) - Yf(3)));	
end



 