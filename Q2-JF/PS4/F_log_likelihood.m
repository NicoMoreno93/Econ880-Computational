function LL = F_log_likelihood(lambda_temp, MODEL_, par,tol_FP,max_iter)
    % Extract needed data from structures
    simdata = MODEL_.simdata;
    P_0 = MODEL_.P_0;
    P_1 = MODEL_.P_1;
    SS_ = MODEL_.SS_;
    F_0 = MODEL_.F_0;
    F_1 = MODEL_.F_1;
    error_ = 100;
    % Initial CCP
    P = 1 ./ (1 + exp(-(P_1 - P_0)));
    % Compute utilities with temporary lambda
    U_1 = par.alpha * SS_.C - SS_.P;
    U_0 = (SS_.I > 0) .* par.alpha .* SS_.C + (SS_.I == 0) .* (SS_.C > 0) .* lambda_temp;
    % ITERATING
    kk = 0; % Start Counting
    while (error_ > tol_FP && kk < max_iter)
        kk = kk + 1;
        F = (1 - P) .* F_0 + P .* F_1;
        EV = (eye(size(F)) - par.beta * F) \ ...
             ((1 - P).*(U_0 + par.eps_0) + P .* (U_1 + par.eps_1));
        V_tilde = (U_1 + par.beta * F_1 * EV) - ...
                  (U_0 + par.beta * F_0 * EV);
        P_next = 1 ./ (1 + exp(-V_tilde));
        error_ = norm(P - P_next,inf);
        P = P_next;
    end
    % Calculate log-likelihood value after convergence:
    prob = P(simdata.state_id + 1);
    LL = sum(simdata.choice .* log(prob) + (1 - simdata.choice) .* log(1 - prob));
end