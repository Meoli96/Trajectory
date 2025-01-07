function [dX] = fdyn(t, X, beta)
    
    mu_s = 1.327e11; % km^3/s^2
    
    
    
    % Adimensional parameters D, T. D^3/T^2 = mu_s
    D = 1.496e8; % km 1 AU
    T = sqrt(D^3/mu_s); % s
    % State = [r, theta, vr, vt]
    % Costate = [lambda_r, lambda_t, lambda_vr, lambda_vt]
    % Unpack state and costate
    S = X(1:4);
    L = X(5:8);
    
    % Finer unpacking
    r = S(1);
    theta = S(2);
    vr = S(3);
    vt = S(4);

    l_r = L(1);
    l_t = L(2);
    l_vr = L(3);
    l_vt = L(4);

    % Optimal control angle
    alpha = atan((-3*l_vr + sqrt(9*l_vr^2 +  8*l_vt^2))/(4*l_vt));

    % Dynamics
    dS = zeros(4, 1);
    dL = zeros(4, 1);

    dS(1) = vr;
    dS(2) = vt/r;
    dS(3) = vt^2/r - 1/r^2 + beta/r^2*cos(alpha)^3;
    dS(4) = -vr*vt/r + beta/r^2*cos(alpha)^2*sin(alpha);

    dL(1) = l_t*vt/r^2 + l_vr*(vt^2/r^2 - 2/r^3 + 2*beta/r^3*cos(alpha)^3) - ...
            l_vt*(vr*vt/r^2 - 2*beta/r^3*cos(alpha)^2*sin(alpha));
    dL(2) = 0;
    dL(3) = -l_r - l_vt*vt/r;
    dL(4) = -l_t/r - 2*l_vr*vt/r + l_vt*vr/r;


    
    dX = [T*dS; dL];


end