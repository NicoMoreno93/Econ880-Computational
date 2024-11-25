% % ------------------- % %
% COMPUTATIONAL
% Author: Nicolas Moreno
% Date: 11/24/2024
% Problem Set 4
% % ------------------- % %
% Preamble:
clear all; clc;
addpath("Data")
tic();
Plot_opt = 1; % Set it to 1 if you want to plot
if exist("Printed_Results.txt","file")
    diary off
    delete *.txt
end
diary("Printed_Results.txt")
%% SETTING UP THE ENVIRONMENT
% ---------------------------------%
% Numerical Parameters:
error_   = 1e4;
tol_FP   = 1e-14;
max_iter = 1e14;
% Setting Model Parameters:
N_      = 36; % Number of Discrete Points
par     = struct(); % Create the Parameters' structure
par.lambda = -4.0;                % Parameter for Stockout penalty
par.alpha  = 2.0;                 % Parameter for Consumption shock
par.beta   = 0.99;                % Discount Factor
par.gamma  = 0.577215664901532860606512090082; % Euler-Mascheroni Constant
% % IMPORT DATA % %
SS_   = readtable(['Data' filesep 'PS4_state_space.csv']);
FF_0 = readtable(['Data' filesep 'PS4_transition_a0.csv']);
FF_1 = readtable(['Data' filesep 'PS4_transition_a1.csv']);

% Convert tables to matrices:
F_0 = table2array(FF_0(:, 3:end));  % Exclude first two columns (ids)
F_1 = table2array(FF_1(:, 3:end));

%% PART 1
% ---------------------------------%
%      VALUE FUNCTION ITERATION    %
% ---------------------------------%
% % Per-period payoff functions:
U_1 = par.alpha * SS_.C - SS_.P;
U_0 = (SS_.I > 0) .* par.alpha .* SS_.C + (SS_.I == 0) .* (SS_.C > 0) .* par.lambda;
% % VFI on expected value:
EV_PFI = log(exp(U_0) + exp(U_1)) + par.gamma;
% % ITERATING OVER VF's:
jj = 0; % Start Counting:
while (error_ > tol_FP && jj < max_iter)
    jj = jj + 1;
    EV_next = log(exp(U_0 + par.beta * F_0 * EV_PFI) + exp(U_1 + par.beta * F_1 * EV_PFI)) + par.gamma;
    error_ = norm(EV_PFI - EV_next,inf);
    EV_PFI = EV_next;
end
disp('% % % Results for Question 1')
fprintf('% Number of iterations of VFI: %d\n', jj);
EV_VFI = EV_PFI;
% Organize Results for Table 1
OUT_table1 = SS_(:,2:end);
OUT_table1.U0 = round(U_0, 4);
OUT_table1.U1 = round(U_1, 4);
OUT_table1.EV = round(EV_PFI, 4);
% Report Results for VFI:
disp('% % TABLE 1')
disp(OUT_table1)
writetable(OUT_table1, 'OUT_Table1.csv');
%% PART 2
% ---------------------------------%
%     POLICY FUNCTION ITERATION    %
% ---------------------------------%
% Read simulation data
SIMUL_ = readtable('Data/PS4_simdata.csv');
% Calculate estimated frequencies
[unique_states, ~, idx] = unique(SIMUL_.state_id);
P_1 = accumarray(idx, SIMUL_.choice, [], @mean);
P_0 = 1 - P_1;
% Constrain the probabilities to be apporx. in range [0,1]:
P_0 = max(min(P_0, 0.999), 0.001);
P_1 = max(min(P_1, 0.999), 0.001);
% Expectation of T1EV shocks
par.eps_0  = par.gamma - log(P_0);
par.eps_1 = par.gamma - log(P_1);
% Initial value for CCP
PP = 1 ./ (1 + exp(-(P_1 - P_0)));
% % ITERATING:
jj = 0; % Start Counting
while (error_ > tol_FP && jj < max_iter)
    jj = jj + 1;
    F = (1 - PP).* F_0 + PP.* F_1;
    EV_PFI = (eye(size(F)) - par.beta * F) \ ...
         ((1 - PP).*(U_0 + par.eps_0) + PP .* (U_1 + par.eps_1));
    V_tilde = (U_1 + par.beta * F_1 * EV_PFI) - (U_0 + par.beta * F_0 * EV_PFI);
    P_next = 1 ./ (1 + exp(-V_tilde));
    error_ = max(abs(PP - P_next));
    PP = P_next;
end
disp('% % % Results for Question 2')
fprintf('% Number of iterations of PFI: %d\n', jj);
% Organize Results for Table 2:
OUT_table2 = SS_(:,2:end);
OUT_table2.P0hat   = round(P_0, 4);
OUT_table2.P1hat   = round(P_1, 4);
OUT_table2.EV      = OUT_table1.EV;
OUT_table2.EVhat   = round(EV_PFI, 4);
OUT_table2.Diff_EV = OUT_table2.EVhat-OUT_table2.EV;
writetable(OUT_table2, 'OUT_Table2.csv');
% Calculate maximum and mean relative differences
max_diff = max(abs((EV_VFI - EV_PFI)./EV_VFI));
mean_diff = mean(abs((EV_VFI - EV_PFI)./EV_VFI));
% Report Results for PFI:
disp('% % TABLE 2')
disp(OUT_table2)
disp('% % Differences Stats')
disp(['Max' 'Mean'])
disp([max_diff mean_diff])
%% PART 4
% Create model structure:
MODEL_ = struct();
MODEL_.simdata = SIMUL_;
MODEL_.P_0 = P_0;
MODEL_.P_1 = P_1;
MODEL_.SS_ = SS_;
MODEL_.F_0 = F_0;
MODEL_.F_1 = F_1;
% Iterate over lambdas:
Lambda_MAT = -15:0.25:0;
LL_MAT     = zeros(size(Lambda_MAT)); % Pre-allocation

% % Evaluating differente lambdas:
for jj = 1:length(Lambda_MAT)
    LL_MAT(jj) = F_log_likelihood(Lambda_MAT(jj), MODEL_, par,tol_FP,max_iter);
end

% Create optimization function handle with fixed parameters
obj_fun = @(x) -F_log_likelihood(x, MODEL_, par,tol_FP,max_iter);

% % Optimize using fminbnd:
% Uncomment next 2 lines if you want to see graphically how it converges
% options = optimset('PlotFcns',@optimplotfval); 
% [lambda_opt, fval] = fminbnd(obj_fun, -10, 0,options);
[lambda_opt, fval] = fminbnd(obj_fun, -10, 0);
fprintf('Optimal lambda: %.4f\n', lambda_opt);
fprintf('Maximum log-likelihood: %.4f\n', -fval);

% Closure:
disp('% To solve all questions...')
toc()
diary off;

% % Plot Log-likelihood function:
if Plot_opt == 1
    fig_1 = figure('Color','w','Position',[200 100 1200 800],'Visible','off');
    orient(fig_1,'landscape')
    ax = gca;
    plot(Lambda_MAT, LL_MAT,'Color',[0 0.6 0.45] , ...
                                   'LineStyle','-', 'Marker','hexagram','MarkerSize',3, ...
                                   'LineWidth',1.75)
    hold on
    xline(lambda_opt,'Color',[0 0.3 0.9],'LineStyle',':','LineWidth',1.5)
    yline(-fval,'Color',[0 0.2 0.7],'LineStyle','-.','LineWidth',1.5)
    grid on
    ylabel('$$Log-likelihood$$','Interpreter','latex');
    xlabel('$$\lambda$$','Interpreter','latex');
    legend({'$$L(X;\lambda)$$', ['$$\hat{\lambda} = ',num2str(lambda_opt,3) ,' $$'], ['$$L(X;\hat{\lambda}) = ',num2str(-fval) ,' $$']},'Location','southwest','FontSize',14,'Interpreter','latex')
    title('Log-likelihood Function over different $$\lambda ''s$$','FontSize',14,'Interpreter','latex')
    ax.FontSize = 14;
    print(['Figures' filesep 'Log-likelihood Function'],'-dpdf','-r0','-bestfit')
end


