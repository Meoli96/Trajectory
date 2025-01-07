We = 1361; % W/m^2
c = 3e8; % m/s
sigma = 20; % kg/m^2

mu_s = 1.327e11; % km^3/s^2



% Adimensional parameters D, T. D^3/T^2 = mu_s
D = 1.496e8; % km 1 AU
T = sqrt(D^3/mu_s); % s

beta = 1e-3*(2*We/(sigma*c))*D^2;
beta_adim = beta*T^2/D^3;

disp(beta_adim)


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

% Guess transfer time
tf_lb = 86400*30*2; % 2 months in seconds
tf_ub = 86400*30*6; % 6 months in seconds

tf_ig = 86400*30*4; % 4 months in seconds
% Adimensionalize bounds
tf_lb_adim = tf_lb/T;
tf_ub_adim = tf_ub/T;
tf_ig_adim = tf_ig/T;



% Uknwown quantities are
% xx = [l_r0, l_vr0, l_vt0, tf]

% We can restrict the search interval for the unknown costates to be
% between -1 and 1 due to adjoint scalability

% Initial guess 
lb = [-1, -1, -1, tf_lb_adim];
ub = [1, 1, 1, tf_ub_adim];
ig = [0.1, 0.1, 0.1, tf_ig_adim];
%% Use fmincon to solve for minimum time

% Initial guess
% bounds for alpha

% constraint_eq(l_r0, l_vr0, l_vt0, tf)

    % r(f) = rf_mer
    % vr(f) = 0
    % vt(f) = vtf_mer



% constraint_ineq(l_r0, l_vr0, l_vt0, tf)
    % H(f) > 0



options_fmincon = optimoptions('fmincon', 'Display', 'iter','OptimalityTolerance',1e-16,'MaxIterations',1e+4,'StepTolerance',1e-12,'MaxFunctionEvaluations',1e+4);

[xOpt] = fmincon(@J, ig, [], [], [], [], lb, ub, @constr, options_fmincon);

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
options = odeset('RelTol', 1e-14, 'AbsTol', 1e-14);
[t, X] = ode113(@(t, X) fdyn(t, X, beta_adim), [0, tf], X0, options);

% Plot results
figure
% Plot Earth as a point in y = 0
plot(1, 0, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b')
hold on
% Plot Mercury as a point in y = 0
plot(0.3871, 0, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r')
hold on

% Plot trajectory
plot(X(:, 1).*cos(X(:, 2)), X(:, 1).*sin(X(:, 2)), 'k', 'LineWidth', 2)




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

    beta = 1e-3*(2*We/(sigma*c))*D^2;
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
    options = odeset('RelTol', 3e-14, 'AbsTol', 1e-14);
    [t, X] = ode113(@(t, X) fdyn(t, X, beta_adim), [0, tf], X0, options);

    % Extract final state
    Sf = X(end, 1:4);
    Lf = X(end, 5:8);
    alpha = atan((-3*Lf(3) + sqrt(9*Lf(3)^2 +  8*Lf(4)^2))/(4*Lf(4)));
    % Compute Hamiltonian
    Hf = Lf(1)*Sf(3) + Lf(2)*Sf(4)/Sf(1) + Lf(3)*(Sf(4)^2/Sf(1) - 1/Sf(1)^2 + beta_adim/Sf(1)^2*cos(alpha)^3) + ...
        Lf(4)*(-Sf(3)*Sf(4)/Sf(1) + beta_adim/Sf(1)^2*cos(alpha)^2*sin(alpha)) - 1;
    
    % Compute constraints
    ineq = Hf;
    eq = [abs(Sf(1) - Yf(1)), abs(Sf(3) - Yf(2)), abs(Sf(4) - Yf(3))];



end

function [tf] = J(xx)
    % Cost function is the final time
    tf = xx(4);
end