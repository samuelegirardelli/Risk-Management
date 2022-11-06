%% Calculating the returns of the financial series about Oil company, in particularly CVX COP TOT ENI,from 1-Jan-2018 to 31-Dec-2022

clear all; close all; clc;

% Loading the database from Excel. The closing price has been downloaded from
% Datasource.
cvx_price = table2timetable(readtable('Database.xlsx','Sheet','CVX'));
cvx_price = renamevars(cvx_price,'Close','CVX');
cop_price = table2timetable(readtable('Database.xlsx','Sheet','COP'));
cop_price = renamevars(cop_price,'Close','COP');
tot_price = table2timetable(readtable('Database.xlsx','Sheet','TOT'));
tot_price = renamevars(tot_price,'Close','TOT');
eni_price = table2timetable(readtable('Database.xlsx','Sheet','ENI'));
eni_price = renamevars(eni_price,'Close','ENI');

oil_prices = synchronize(cvx_price,cop_price,tot_price,eni_price,'first');
oil_prices = fillmissing(oil_prices,'nearest');

% Calculating the log returns
cvx_return = timetable(cvx_price.Date(2:end),tick2ret(cvx_price.CVX,'Method','continuous'),'VariableNames',{'CVX'});
cop_return = timetable(cop_price.Date(2:end),tick2ret(cop_price.COP,'method','continuous'),'VariableNames',{'COP'});
tot_return = timetable(tot_price.Date(2:end),tick2ret(tot_price.TOT,'method','continuous'),'VariableNames',{'TOT'});
eni_return = timetable(eni_price.Date(2:end),tick2ret(eni_price.ENI,'method','continuous'),'VariableNames',{'ENI'});

oil_returns = synchronize(cvx_return,cop_return,tot_return,eni_return,'first');
oil_returns = fillmissing(oil_returns,'nearest');

% Clearing the workspace from useful data
clear('cop_price','cop_return','cvx_price','cvx_return','eni_price','eni_return','tot_price','tot_return');

% Saving the file
save OilReturns

%% Visualizing the data of the Oil Company

% Through the plot function visualizing the prices
figure('Name','Closing prices of oil companies')
hold on
grid on
plot(oil_prices.Date,ret2price(oil_returns.CVX)*100,'Color','red');
plot(oil_prices.Date,ret2price(oil_returns.COP)*100,'Color','green');
plot(oil_prices.Date,ret2price(oil_returns.TOT)*100,'Color','cyan');
plot(oil_prices.Date,ret2price(oil_returns.ENI)*100,'Color','yellow');

% Font
datetick('x')
xlabel('Date','FontSize',12,'FontWeight','bold');
ylabel('Companies Price Values','FontSize',12,'FontWeight','bold');
title('Relative Daily Closing Prices')
legend('CVX','COP','TOT','ENI','location','best');

hold off

%% Plotting the returns
figure('Name','Returns of oil companies')
%CVX Return
subplot(2,2,1);
plot(oil_returns.Time,oil_returns.CVX)
xlabel('Date','FontSize',8,'FontWeight','bold');
ylabel('CVX Return','FontSize',8,'FontWeight','bold');
%COP Return
subplot(2,2,2);
plot(oil_returns.Time,oil_returns.COP)
xlabel('Date','FontSize',8,'FontWeight','bold');
ylabel('COP Return','FontSize',8,'FontWeight','bold');
%TOT Return
subplot(2,2,3);
plot(oil_returns.Time,oil_returns.TOT)
xlabel('Date','FontSize',8,'FontWeight','bold');
ylabel('TOT Return','FontSize',8,'FontWeight','bold');
%ENI Return
subplot(2,2,4);
plot(oil_returns.Time,oil_returns.ENI)
xlabel('Date','FontSize',8,'FontWeight','bold');
ylabel('ENI Return','FontSize',8,'FontWeight','bold');

%% Looking if some empirical distributions fit our data - tStudent
figure('Name','t-Student distribution fitting')
subplot(2,2,1)
histfit(oil_returns.CVX,50,'tlocationscale');
subplot(2,2,2)
histfit(oil_returns.COP,50,'tlocationscale');
subplot(2,2,3)
histfit(oil_returns.TOT,50,'tlocationscale');
subplot(2,2,4)
histfit(oil_returns.ENI,50,'tlocationscale');

%% Validanting the assumption of independence and identically distributed

%% Verifing Autocorrelation function
figure('Name','ACF graphs of oil companies');
% CVX ACF
subplot(2,2,1)
autocorr(oil_returns.CVX);
xlabel('Lag','FontSize',8,'FontWeight','bold')
ylabel('CVX ACF','FontSize',8,'FontWeight','bold')
ylim([-0.2 1])
% COP ACF
subplot(2,2,2)
autocorr(oil_returns.COP);
xlabel('Lag','FontSize',8,'FontWeight','bold')
ylabel('COP ACF','FontSize',8,'FontWeight','bold')
ylim([-0.2 1])
% TOT ACF
subplot(2,2,3)
autocorr(oil_returns.TOT);
xlabel('Lag','FontSize',8,'FontWeight','bold')
ylabel('TOT ACF','FontSize',8,'FontWeight','bold')
ylim([-0.2 1])
% ENI ACF
subplot(2,2,4)
autocorr(oil_returns.ENI);
xlabel('Lag','FontSize',8,'FontWeight','bold')
ylabel('ENI ACF','FontSize',8,'FontWeight','bold')
ylim([-0.2 1])

%% Control Partial Autocorrelation Function

figure('Name','PACF for oil companies')
% CVX PACF
subplot(2,2,1)
parcorr(oil_returns.CVX)
xlabel('Lag','FontSize',8,'FontWeight','bold')
ylabel('CVX PACF','FontSize',8,'FontWeight','bold')
ylim([-0.2 1])
% COP PACF
subplot(2,2,2)
parcorr(oil_returns.COP)
xlabel('Lag','FontSize',8,'FontWeight','bold')
ylabel('COP PACF','FontSize',8,'FontWeight','bold')
ylim([-0.2 1])
% TOT PACF
subplot(2,2,3)
parcorr(oil_returns.TOT)
xlabel('Lag','FontSize',8,'FontWeight','bold')
ylabel('TOT PACF','FontSize',8,'FontWeight','bold')
ylim([-0.2 1])
% ENI PACF
subplot(2,2,4)
parcorr(oil_returns.ENI)
xlabel('Lag','FontSize',8,'FontWeight','bold')
ylabel('ENI PACF','FontSize',8,'FontWeight','bold')
ylim([-0.2 1])

%% Checking the square residuals dependence
figure('Name','Squared Residuals ACF')
% CVX ACF Squared Residuals
subplot(2,2,1)
autocorr(oil_returns.CVX.^2)
xlabel('Lag','FontSize',8,'FontWeight','bold')
ylabel('CVX ACF Squared Residuals','FontSize',8,'FontWeight','bold')
ylim([0 1])
% COP ACF Squared Residuals
subplot(2,2,2)
autocorr(oil_returns.COP.^2)
xlabel('Lag','FontSize',8,'FontWeight','bold')
ylabel('COP ACF Squared Residuals','FontSize',8,'FontWeight','bold')
ylim([0 1])
% TOT ACF Squared Residuals
subplot(2,2,3)
autocorr(oil_returns.TOT.^2)
xlabel('Lag','FontSize',8,'FontWeight','bold')
ylabel('TOT ACF Squared Residuals','FontSize',8,'FontWeight','bold')
ylim([0 1])
% ENI ACF Squared Residuals
subplot(2,2,4)
autocorr(oil_returns.ENI.^2)
xlabel('Lag','FontSize',8,'FontWeight','bold')
ylabel('ENI ACF Squared Residuals','FontSize',8,'FontWeight','bold')
ylim([0 1])

%% Working on IID data and heteroschedasticity: in order to do correct analysis we have to standardize our returns.
% Summary of ours data
% CVX: 1217×1 double     COP: 1217×1 double     TOT: 1217×1 double     ENI: 1217×1 double
% Values:              % Values:              % Values:              % Values:
% Min         -0.25006 % Min       -0.28555   % Min       -0.24116   % Min         -0.23385
% Median    0.00025825 % Median           0   % Median           0   % Median    0.00029176
% Max           0.2049 % Max        0.22485   % Max         0.1308   % Max          0.13916

% Modelling the first moment of the distribution MEAN and the
% second moment VARIANCE for the whole dataset.

% Fitting a model for the mean but from a preliminary analysiss of the
% summary we note that our data seems already with no mean.

% ARIMA Model and GARCH Model togheter 
model = arima('AR',NaN,'Distribution','Gaussian','Variance',gjr(1,1));
option = optimoptions(@fmincon,'Display','off','Diagnostic','off','Algorithm','sqp','TolCon',1e-7);

% CVX Estimate and inference
cvx_fit = estimate(model,oil_returns.CVX,'Option',option);
[~,CVX_variances] = infer(cvx_fit,oil_returns.CVX);

% COP Estimate and inference
cop_fit = estimate(model,oil_returns.COP,'Option',option);
[~,COP_variances] = infer(cop_fit,oil_returns.COP);

% TOT Estimate and inference
tot_fit = estimate(model,oil_returns.TOT,'Option',option);
[~,TOT_variances] = infer(tot_fit,oil_returns.TOT);

% ENI Estimate and inference
eni_fit = estimate(model,oil_returns.ENI,'Option',option);
[~,ENI_variances] = infer(eni_fit,oil_returns.ENI);
%% Remarks
% From the model that we generate we don't extrapolate any significative
% conclusion about AR process for the first moment.
% Conversly the whole sample seems to have significative variance
% dependence among time and the parameter of the GARCH model result
% significant for at least 5%.
% Some significance has been found for the 'leverage' GARCH parameter.
%% Creating dataframe with standardized returns in order to apply analysis.

% CVX Standardized Residuals and look for confirmation
cvx_std = oil_returns.CVX./sqrt(CVX_variances);

figure('Name','ACF of Squared STD Residuals')

subplot(2,2,1)
autocorr(cvx_std.^2)
xlabel('Lag','FontSize',8,'FontWeight','bold')
ylabel('CVX ACF STD Squared Residuals','FontSize',8,'FontWeight','bold')
ylim([0 1])

% COP Standardized Residuals and look for confirmation
cop_std = oil_returns.COP./sqrt(COP_variances);

subplot(2,2,2)
autocorr(cop_std.^2)
xlabel('Lag','FontSize',8,'FontWeight','bold')
ylabel('COP ACF STD Squared Residuals','FontSize',8,'FontWeight','bold')
ylim([0 1])

% TOT Standardized Residuals and look for confirmation
tot_std = oil_returns.TOT./sqrt(TOT_variances);

subplot(2,2,3)
autocorr(tot_std.^2)
xlabel('Lag','FontSize',8,'FontWeight','bold')
ylabel('TOT ACF STD Squared Residuals','FontSize',8,'FontWeight','bold')
ylim([0 1])

% ENI Standardized Residuals and look for confirmation
eni_std = oil_returns.ENI./sqrt(ENI_variances);

subplot(2,2,4)
autocorr(eni_std.^2)
xlabel('Lag','FontSize',8,'FontWeight','bold')
ylabel('ENI ACF STD Squared Residuals','FontSize',8,'FontWeight','bold')
ylim([0 1])

%% Results: now we have the confirmation that we are working on I.I.D. Data

% Rearrange the data for the next one analysis
oil_std_returns = table(cvx_std,cop_std,tot_std,eni_std,'VariableNames',{'CVX_res','COP_res','TOT_res','ENI_res'});
oil_variances = table(CVX_variances,COP_variances,TOT_variances,ENI_variances,'VariableNames',{'CVX_var','COP_var','TOT_var','ENI_var'});
oil_ret = table(oil_returns.CVX,oil_returns.COP,oil_returns.TOT,oil_returns.ENI,'VariableNames',{'CVX_ret','COP_ret','TOT_ret','ENI_ret'});
oil_date = table(oil_returns.Time,'VariableNames',{'Date'});
clear('COP_residuals','cop_std','COP_variances','CVX_residuals','cvx_std','CVX_variances','ENI_residuals','eni_std','ENI_variances','model','option','TOT_residuals','tot_std','TOT_variances','oil_prices','oil_returns');
save DataSTDoil.mat

%% In this part we want to explain differents method in order to calculate VAR and the relative backtesing applying the following procedures:
%  1. Normal Distribution approach (Parametric)
%  2. Historical Simulation approach (Non-Parametric)
%  3. Extreme Value Theory approach (Semi-Parametric)

%% Arrange data in order to make the subsequent analysis more fluid
% Define the window
oil_date = oil_date.Date;
TestWindowStart = find(year(oil_date)==2019,1);
TestWindowEnd = find(year(oil_date)==2022,1,'last');
TestWindow = TestWindowStart:TestWindowEnd;
WindowSize = 250; % Window Size for HS,ND,WHS
index = oil_std_returns.Properties.VariableNames;
% Define the VaR confindance level
pVaR = [0.05,0.01];

%% 1. Normal distribution approach using movable windows (Paramatric)
Zscore = norminv(pVaR);


% Calculating the VaR in a for loop for every point of test window and for
% every companies
% NB: the var is calculated for the negative side of the distribution of
% residuals
Normal95 = zeros(length(TestWindow),width(index));
Normal99 = zeros(length(TestWindow),width(index));

for j = 1:width(index)
    for t=TestWindow
        i=t-TestWindowStart+1;
        EstimationWindow = t-WindowSize:t-1;
        Sigma_window = std(oil_std_returns{EstimationWindow,index(j)});
        Normal95(i,j) = -Zscore(1)*Sigma_window;
        Normal99(i,j) = -Zscore(2)*Sigma_window;
    end
% Rappresent the VaR for each confidance on a figure
    figure('Name','VaR Normal Distribution Method')
    plot(oil_date(TestWindow),[Normal95(:,j) Normal99(:,j)])
    xlabel('Date');
    ylabel('VaR')
    legend({'95% Confidence level','99% Confidence level'},'location','Best');
    title(strcat(['VaR Normal Distribution method '],index(j)));

% VaR Backtesting of Normal distribution approach

    Returns_test = oil_std_returns{TestWindow,index(j)};
    Date_test = oil_date(TestWindow);
    figure('Name','Return Test and VaR Estimation')
    plot(Date_test,[Returns_test -Normal95(:,j) -Normal99(:,j)]);
    xlabel('VaR Estimated');
    ylabel('Date')
    title(strcat(['Returns test vs VaR '],index(j)));
    legend({'Return Test','Normal VaR 95','Normal VaR 99'});
    vbt = varbacktest(Returns_test,[Normal95(:,j) Normal99(:,j)],'PortfolioID',index(j),'VaRID',{'Normal95','Normal99'},'VaRLevel',0.95);
    backtest = runtests(vbt);
    disp(backtest)
end
%% 2. Historical Simulation Approach using movable windows
    Historical95 = zeros(length(TestWindow),width(index));
    Historical99 = zeros(length(TestWindow),width(index));

for j = 1:width(oil_std_returns)
    for t = TestWindow
        i = t-TestWindowStart+1;
        EstimationWindow = t-WindowSize:t-1;
        Historical95(i,j) = -quantile(oil_std_returns{EstimationWindow,index(j)},pVaR(1));
        Historical99(i,j) = -quantile(oil_std_returns{EstimationWindow,index(j)},pVaR(2));
    end
    figure('Name','VaR Historical Simulation Method')
    plot(oil_date(TestWindow),[Historical95(:,j),Historical99(:,j)]);
    xlabel('VaR Historical')
    ylabel('Date')
    title(strcat(['VaR Historical Method'],index(j)));
    legend({'Historical 95','Historical 99'});

    % VaR backtesting for Historical simulation approach
    Returns_test = oil_std_returns{TestWindow,index(j)};
    Date_test = oil_date(TestWindow);
    figure('Name','Return Test and Historical VaR')
    plot(Date_test,[Returns_test,-Historical95(:,j),-Historical99(:,j)]);
    xlabel('Date')
    ylabel('Returns Test and Historical VaR');
    title(strcat(['VaR and Returns Test for'],index(j)));
    legend({'Returns','Historical 95','Historical 99'},'location','best');
    vbt =varbacktest(Returns_test,[Historical95(:,j),Historical99(:,j)],'PortfolioID',index(j),'VaRID',{'Historical 95','Historical 99'},'VaRLevel',1-pVaR(1));
    backtest = runtests(vbt);
    disp(backtest)
end

%% Conclusion for Normal Distribution
   % PortfolioID      VaRID       VaRLevel     TL       Bin       POF       TUFF       CC       CCI       TBF       TBFI 
   % ___________    __________    ________    _____    ______    ______    ______    ______    ______    ______    ______

     %"CVX_res"     "Normal95"      0.95      green    accept    accept    accept    accept    accept    accept    accept
     %"CVX_res"     "Normal99"      0.95      green    reject    reject    accept    reject    accept    reject    reject

   % PortfolioID      VaRID       VaRLevel     TL       Bin       POF       TUFF       CC       CCI       TBF       TBFI 
   % ___________    __________    ________    _____    ______    ______    ______    ______    ______    ______    ______

    % "COP_res"     "Normal95"      0.95      green    accept    accept    accept    accept    accept    accept    accept
    % "COP_res"     "Normal99"      0.95      green    reject    reject    accept    reject    accept    reject    reject

   % PortfolioID      VaRID       VaRLevel     TL       Bin       POF       TUFF       CC       CCI       TBF       TBFI 
   % ___________    __________    ________    _____    ______    ______    ______    ______    ______    ______    ______

    % "TOT_res"     "Normal95"      0.95      green    accept    accept    accept    accept    accept    accept    accept
    % "TOT_res"     "Normal99"      0.95      green    reject    reject    accept    reject    accept    reject    reject

   % PortfolioID      VaRID       VaRLevel     TL       Bin       POF       TUFF       CC       CCI       TBF       TBFI 
   % ___________    __________    ________    _____    ______    ______    ______    ______    ______    ______    ______

    % "ENI_res"     "Normal95"      0.95      green    accept    accept    accept    accept    accept    accept    accept
    % "ENI_res"     "Normal99"      0.95      green    reject    reject    accept    reject    accept    reject    reject
%% Conclusion for Historical Simulation
    %PortfolioID         VaRID         VaRLevel     TL       Bin       POF       TUFF       CC       CCI       TBF       TBFI 
    %___________    _______________    ________    _____    ______    ______    ______    ______    ______    ______    ______

     %"CVX_res"     "Historical 95"      0.95      green    accept    accept    accept    accept    accept    accept    accept
     %"CVX_res"     "Historical 99"      0.95      green    reject    reject    accept    reject    accept    reject    reject

    %PortfolioID         VaRID         VaRLevel     TL       Bin       POF       TUFF       CC       CCI       TBF       TBFI 
    %___________    _______________    ________    _____    ______    ______    ______    ______    ______    ______    ______

     %"COP_res"     "Historical 95"      0.95      green    accept    accept    accept    accept    accept    accept    accept
     %"COP_res"     "Historical 99"      0.95      green    reject    reject    reject    reject    accept    reject    reject

   % PortfolioID         VaRID         VaRLevel     TL       Bin       POF       TUFF       CC       CCI       TBF       TBFI 
   % ___________    _______________    ________    _____    ______    ______    ______    ______    ______    ______    ______

    % "TOT_res"     "Historical 95"      0.95      green    accept    accept    accept    accept    accept    accept    accept
     %"TOT_res"     "Historical 99"      0.95      green    reject    reject    accept    reject    accept    reject    reject

   % PortfolioID         VaRID         VaRLevel     TL       Bin       POF       TUFF       CC       CCI       TBF       TBFI 
    %___________    _______________    ________    _____    ______    ______    ______    ______    ______    ______    ______

     %"ENI_res"     "Historical 95"      0.95      green    accept    accept    accept    accept    accept    accept    accept
     %"ENI_res"     "Historical 99"      0.95      green    reject    reject    reject    reject    accept    reject    reject

%% Returns expected given the best fitted model

%% Normal 95 
% Calcolo VAR in t+1

% CVX
    Un_var_CVX = cvx_fit.Variance.UnconditionalVariance;
    ARCH_par_CVX = cell2mat(cvx_fit.Variance.ARCH);
    GARCH_par_CVX =cell2mat(cvx_fit.Variance.GARCH);
    last_res_CVX = table2array(oil_std_returns(end,1));
    last_var_CVX = table2array(oil_variances(end,1));

    Sigma_t1 = std(oil_std_returns{t-WindowSize+1:t,index(1)});
    Normal95_t1 = -Zscore(1)*Sigma_t1;
    VAR_t1_CVX = Normal95_t1*(Un_var_CVX+ARCH_par_CVX*(last_res_CVX-Un_var_CVX)+ ...
    GARCH_par_CVX*(last_var_CVX-Un_var_CVX));

% COP 
    Un_var_COP = cop_fit.Variance.UnconditionalVariance;
    ARCH_par_COP = cell2mat(cop_fit.Variance.ARCH);
    GARCH_par_COP =cell2mat(cop_fit.Variance.GARCH);
    last_res_COP = table2array(oil_std_returns(end,2));
    last_var_COP = table2array(oil_variances(end,2));

    Sigma_t1 = std(oil_std_returns{t-WindowSize+1:t,index(2)});
    Normal95_t1 = -Zscore(1)*Sigma_t1;
    VAR_t1_COP = Normal95_t1*(Un_var_COP+ARCH_par_COP*(last_res_COP-Un_var_COP)+ ...
    GARCH_par_COP*(last_var_COP-Un_var_COP));

% TOT 
    Un_var_TOT = tot_fit.Variance.UnconditionalVariance;
    ARCH_par_TOT = cell2mat(tot_fit.Variance.ARCH);
    GARCH_par_TOT =cell2mat(tot_fit.Variance.GARCH);
    last_res_TOT = table2array(oil_std_returns(end,3));
    last_var_TOT = table2array(oil_variances(end,3));

    Sigma_t1 = std(oil_std_returns{t-WindowSize+1:t,index(3)});
    Normal95_t1 = -Zscore(1)*Sigma_t1;
    VAR_t1_TOT = Normal95_t1*(Un_var_TOT+ARCH_par_TOT*(last_res_TOT-Un_var_TOT)+ ...
    GARCH_par_TOT*(last_var_TOT-Un_var_TOT));
% ENI 
    Un_var_ENI = tot_fit.Variance.UnconditionalVariance;
    ARCH_par_ENI = cell2mat(eni_fit.Variance.ARCH);
    GARCH_par_ENI =cell2mat(eni_fit.Variance.GARCH);
    last_res_ENI = table2array(oil_std_returns(end,4));
    last_var_ENI = table2array(oil_variances(end,4));

    Sigma_t1 = std(oil_std_returns{t-WindowSize+1:t,index(4)});
    Normal95_t1 = -Zscore(1)*Sigma_t1;
    VAR_t1_ENI = Normal95_t1*(Un_var_ENI+ARCH_par_ENI*(last_res_ENI-Un_var_ENI)+ ...
    GARCH_par_ENI*(last_var_ENI-Un_var_ENI));

 ResultsN = [VAR_t1_CVX VAR_t1_COP VAR_t1_TOT VAR_t1_ENI]

%% Historical 95 
% Calcolo VAR in t+1

% CVX
    Un_var_CVX = cvx_fit.Variance.UnconditionalVariance;
    ARCH_par_CVX = cell2mat(cvx_fit.Variance.ARCH);
    GARCH_par_CVX =cell2mat(cvx_fit.Variance.GARCH);
    last_res_CVX = table2array(oil_std_returns(end,1));
    last_var_CVX = table2array(oil_variances(end,1));

   
    Historical95_t1 = -quantile(oil_std_returns{t-WindowSize+1:t,index(1)},pVaR(1));
    VAR_t1_CVX_H = Historical95_t1*(Un_var_CVX+ARCH_par_CVX*(last_res_CVX-Un_var_CVX)+ ...
    GARCH_par_CVX*(last_var_CVX-Un_var_CVX));

% COP 
    Un_var_COP = cop_fit.Variance.UnconditionalVariance;
    ARCH_par_COP = cell2mat(cop_fit.Variance.ARCH);
    GARCH_par_COP =cell2mat(cop_fit.Variance.GARCH);
    last_res_COP = table2array(oil_std_returns(end,2));
    last_var_COP = table2array(oil_variances(end,2));

    Historical95_t1 = -quantile(oil_std_returns{t-WindowSize+1:t,index(2)},pVaR(1));
    VAR_t1_COP_H = Historical95_t1*(Un_var_COP+ARCH_par_COP*(last_res_COP-Un_var_COP)+ ...
    GARCH_par_COP*(last_var_COP-Un_var_COP));

% TOT 
    Un_var_TOT = tot_fit.Variance.UnconditionalVariance;
    ARCH_par_TOT = cell2mat(tot_fit.Variance.ARCH);
    GARCH_par_TOT =cell2mat(tot_fit.Variance.GARCH);
    last_res_TOT = table2array(oil_std_returns(end,3));
    last_var_TOT = table2array(oil_variances(end,3));

    Historical95_t1 = -quantile(oil_std_returns{t-WindowSize+1:t,index(3)},pVaR(1));
    VAR_t1_TOT_H = Historical95_t1*(Un_var_TOT+ARCH_par_TOT*(last_res_TOT-Un_var_TOT)+ ...
    GARCH_par_TOT*(last_var_TOT-Un_var_TOT));
% ENI 
    Un_var_ENI = tot_fit.Variance.UnconditionalVariance;
    ARCH_par_ENI = cell2mat(eni_fit.Variance.ARCH);
    GARCH_par_ENI =cell2mat(eni_fit.Variance.GARCH);
    last_res_ENI = table2array(oil_std_returns(end,4));
    last_var_ENI = table2array(oil_variances(end,4));

    Historical95_t1 = -quantile(oil_std_returns{t-WindowSize+1:t,index(4)},pVaR(1));
    VAR_t1_ENI_H = Historical95_t1*(Un_var_ENI+ARCH_par_ENI*(last_res_ENI-Un_var_ENI)+ ...
    GARCH_par_ENI*(last_var_ENI-Un_var_ENI));

 ResultsHS = [VAR_t1_CVX_H VAR_t1_COP_H VAR_t1_TOT_H VAR_t1_ENI_H]
 
%% We want to analyze our companies with a semi-parametric method: Extreme Value Theory
% Does the EVT perform better than the previous method used to calculate
% the VaR?
% Our assumption is that we use a tail cut off of 10%

%% First of all we want to compute an EVT distribution to approximate
n_point = 200;
tail_fr = 0.10;


%% Working on the whole dataset
tails_cvx = paretotails(oil_std_returns.CVX_res,tail_fr,1-tail_fr,'kernel');
tails_cop = paretotails(oil_std_returns.COP_res,tail_fr,1-tail_fr,'kernel');
tails_tot = paretotails(oil_std_returns.TOT_res,tail_fr,1-tail_fr,'kernel');
tails_eni = paretotails(oil_std_returns.ENI_res,tail_fr,1-tail_fr,'kernel');

%% Plotting the distribution applying power law decay on tails and kernel distribution in the middle of our distributions
figure('Name','GPD on Tails Distributions')
% CVX
subplot(2,2,1)
hold on
grid on
min_prob = cdf(tails_cvx,min(oil_std_returns.CVX_res));
max_prob = cdf(tails_cvx,max(oil_std_returns.CVX_res));

plow_tail_cvx = linspace(min_prob,tail_fr,n_point);
pup_tail_cvx = linspace(max_prob,1-tail_fr,n_point);
pint_cvx = linspace(tail_fr,1-tail_fr,n_point);

plot(icdf(tails_cvx,plow_tail_cvx),plow_tail_cvx,'red','LineWidth',4);
plot(icdf(tails_cvx,pint_cvx),pint_cvx,'black','LineWidth',2);
plot(icdf(tails_cvx,pup_tail_cvx),pup_tail_cvx,'blue','LineWidth',2);

xlabel('Standardized Residuals')
ylabel('Probability')
title('Empirical CDF CVX')
legend({'Pareto Lower Tail','Kernel Smoothed Interior','Pareto Upper Tail'},'location','northwest')
hold off

% COP 
subplot(2,2,2)
hold on 
grid on

min_prob = cdf(tails_cop,min(oil_std_returns.COP_res));
max_prob = cdf(tails_cop,max(oil_std_returns.COP_res));

plow_tail_cop = linspace(min_prob,tail_fr,n_point);
pint_cop = linspace(tail_fr,1-tail_fr,n_point);
pup_tail_cop = linspace(1-tail_fr,max_prob,n_point);

plot(icdf(tails_cop,plow_tail_cop),plow_tail_cop,'red','LineWidth',4);
plot(icdf(tails_cop,pint_cop),pint_cop,'black','LineWidth',2);
plot(icdf(tails_cop,pup_tail_cop),pup_tail_cop,'blue','LineWidth',2);

xlabel('Standardized Residuals')
ylabel('Probability')
title('Empirical Distribution Function COP')
legend({'Pareto Lower Tail','Kernel Smoothed Interior','Pareto Upper Tail'},'location','northwest')
hold off

% TOT
subplot(2,2,3)
hold on 
grid on

min_prob = cdf(tails_tot,min(oil_std_returns.TOT_res));
max_prob = cdf(tails_tot,max(oil_std_returns.TOT_res));

plow_tail_tot = linspace(min_prob,tail_fr,n_point);
pint_tot = linspace(tail_fr,1-tail_fr,n_point);
pup_tail_tot = linspace(1-tail_fr,max_prob,n_point);

plot(icdf(tails_tot,plow_tail_tot),plow_tail_tot,'red','LineWidth',4);
plot(icdf(tails_tot,pint_tot),pint_tot,'black','LineWidth',2);
plot(icdf(tails_tot,pup_tail_tot),pup_tail_tot,'blue','LineWidth',2);

xlabel('Standardized Residuals')
ylabel('Probability')
title('Empirical Distribution Function TOT')
legend({'Pareto Lower Tail','Kernel Smoothed Interior','Pareto Upper Tail'},'location','northwest')
hold off

% ENI
subplot(2,2,4)
hold on 
grid on

min_prob = cdf(tails_eni,min(oil_std_returns.ENI_res));
max_prob = cdf(tails_eni,max(oil_std_returns.ENI_res));

plow_tail_eni = linspace(min_prob,tail_fr,n_point);
pint_eni = linspace(tail_fr,1-tail_fr,n_point);
pup_tail_eni = linspace(1-tail_fr,max_prob,n_point);

plot(icdf(tails_eni,plow_tail_eni),plow_tail_eni,'red','LineWidth',4);
plot(icdf(tails_eni,pint_eni),pint_eni,'black','LineWidth',2);
plot(icdf(tails_eni,pup_tail_eni),pup_tail_eni,'blue','LineWidth',2);

xlabel('Standardized Residuals')
ylabel('Probability')
title('Empirical Distribution Function ENI')
legend({'Pareto Lower Tail','Kernel Smoothed Interior','Pareto Upper Tail'},'location','northwest')
hold off

%% Asses the GPD Fit in the Lower Tails
figure('Name','GPD fit of the oil companies')
% CVX 
subplot(2,2,1)
[P,Q] = boundary(tails_cvx);
y = sortrows(oil_std_returns.CVX_res(oil_std_returns.CVX_res<Q(1),1))-Q(1);
plot(y,(cdf(tails_cvx,y+Q(1)))/P(1));
[F,x] = ecdf(y,'Bounds','on');
hold on
stairs(x,F,'r')
grid on

legend('Fitted GPD','Empirical CDF','Location','best')
xlabel('Exceedance')
ylabel('Probability')
title('Lower Tail of Standardized residuals of CVX')
ylim([0 1])
% COP 
subplot(2,2,2)
[P,Q] = boundary(tails_cop);
y = sortrows(oil_std_returns.COP_res(oil_std_returns.COP_res<Q(1),1))-Q(1);
plot(y,(cdf(tails_cop,y+Q(1)))/P(1));
[F,x] = ecdf(y);
hold on
stairs(x,F,'r')
grid on

legend('Fitted GPD','Empirical CDF','Location','best')
xlabel('Exceedance')
ylabel('Probability')
title('Lower Tail of Standardized residuals of COP')
ylim([0 1])

% TOT 
subplot(2,2,3)
[P,Q] = boundary(tails_tot);
y = sortrows(oil_std_returns.TOT_res(oil_std_returns.TOT_res<Q(1),1))-Q(1);
plot(y,(cdf(tails_tot,y+Q(1)))/P(1));
[F,x] = ecdf(y);
hold on
stairs(x,F,'r')
grid on

legend('Fitted GPD','Empirical CDF','Location','best')
xlabel('Exceedance')
ylabel('Probability')
title('Lower Tail of Standardized residuals of TOT')
ylim([0 1])

% ENI
subplot(2,2,4)
[P,Q] = boundary(tails_eni);
y = sortrows(oil_std_returns.ENI_res(oil_std_returns.ENI_res<Q(1),1))-Q(1);
plot(y,(cdf(tails_eni,y+Q(1)))/P(1));
[F,x] = ecdf(y);
hold on
stairs(x,F,'r')
grid on

legend('Fitted GPD','Empirical CDF','Location','best')
xlabel('Exceedance')
ylabel('Probability')
title('Lower Tail of Standardized residuals of ENI')
ylim([0 1])

%% Calculating the VAR with movable windows in order to compare and evaluate the method against the others.

TestWindowStart = find(year(oil_date)==2019,1);
TestWindowEnd = find(year(oil_date)==2022,1,'last');
TestWindow = TestWindowStart:TestWindowEnd;
WindowSize = 250;
index = oil_std_returns.Properties.VariableNames;
pVaR = [0.05,0.01];
VaREVT95 = zeros(length(TestWindow),width(index));
VaREVT99 = zeros(length(TestWindow),width(index));


%% Building the ciclo for 
for j = 1 : width(index)
    for t = TestWindow
        i = t-TestWindowStart+1;
        EstimationWindow = t-WindowSize:t-1;
        tails = paretotails(oil_std_returns{EstimationWindow,index(j)},tail_fr,1-tail_fr,'kernel');
        Q = boundary(tails);
        u = Q(1);
        beta = tails.LowerParameters(2);
        shape = tails.LowerParameters(1);
        N = WindowSize;
        Nu = sum(oil_std_returns{EstimationWindow,index(j)}<Q(1));
        VaREVT95(i,j) = EVT(u,beta,shape,N,Nu,pVaR(1));
        VaREVT99(i,j) = EVT(u,beta,shape,N,Nu,pVaR(2));
    end
    figure('Name','VaR EVT method');
    plot(oil_date(TestWindow,1),[oil_std_returns{TestWindow,index(j)}]);
    hold on
    grid on
    plot(oil_date(TestWindow,1),[-VaREVT95(:,j),-VaREVT99(:,j)]);
    xlabel('Date');
    ylabel('VaR EVT ');
    legend({'Returns','EVT 95','EVT 99'},'location','best');
    title(strcat(['VaR calculated with EVT for '],index(j)));
    % backtesting
    Returns_test = oil_std_returns{TestWindow,index(j)};
    Date_test = oil_date(TestWindow,1);  
    vbt =varbacktest(Returns_test,[VaREVT95(:,j),VaREVT99(:,j)],'PortfolioID',index(j),'VaRID',{'EVT 95','EVT 99'},'VaRLevel',1-pVaR(1));
    backtest = runtests(vbt);
    disp(backtest)
end

%% Building graphs in order to compare the two method at 95 % of confidance on the whole period
for j = 1:width(index)
    VaRData = [-Normal95(:,j),-Historical95(:,j),-VaREVT95(:,j)];
    VaRFormat = {'-','--','.'};
    IndexNormal95 = oil_std_returns{TestWindow,index(j)}<VaRData(1); %% Da vedere
    IndexHistorical95 = oil_std_returns{TestWindow,index(j)}<VaRData(2);
    IndexEVT95 = oil_std_returns{TestWindow,index(j)}<VaRData(3);
    figure('Name','VaR Excedance')
    bar(oil_date(TestWindow),oil_std_returns{TestWindow,index(j)},'FaceColor',['k',0.7,0.7]);
    hold on
    for i=1:width(VaRData)
        stairs(oil_date(TestWindow),VaRData(:,i),VaRFormat{i});
    end
    xlabel('Date')
    ylabel('VaR Normal EVT Historical with Exceedances')
    title(strcat(['VaR 95% violations for Normal and Historical EVT of'],index(j)));
    ax = gca;
    ax.ColorOrderIndex = 2;
    plot(oil_date(TestWindow(IndexNormal95)),-Normal95(IndexNormal95,j),'o',oil_date(TestWindow(IndexHistorical95)),-Historical95(IndexHistorical95,j),'x',oil_date(TestWindow(IndexEVT95)),-VaREVT95(IndexEVT95,j),'p','MarkerSize',8,'LineWidth',1);
    hold off
    legend({'Returns','Normal','Historical','EVT','Normal Violations','Historical Violations','EVT Violations'},'Location','best')
end

%% Conclusion EVT
% PortfolioID     VaRID      VaRLevel      TL       Bin       POF       TUFF       CC       CCI       TBF       TBFI 
   % ___________    ________    ________    ______    ______    ______    ______    ______    ______    ______    ______

    % "CVX_res"     "EVT 95"      0.95      yellow    reject    reject    accept    reject    accept    reject    reject
    % "CVX_res"     "EVT 99"      0.95      green     reject    reject    accept    reject    accept    reject    reject

% PortfolioID     VaRID      VaRLevel     TL       Bin       POF       TUFF       CC       CCI       TBF       TBFI 
 %   ___________    ________    ________    _____    ______    ______    ______    ______    ______    ______    ______

   %  "COP_res"     "EVT 95"      0.95      red      reject    reject    accept    reject    reject    reject    reject
    % "COP_res"     "EVT 99"      0.95      green    reject    reject    reject    reject    reject    reject    reject

% PortfolioID     VaRID      VaRLevel     TL       Bin       POF       TUFF       CC       CCI       TBF       TBFI 
   % ___________    ________    ________    _____    ______    ______    ______    ______    ______    ______    ______

    % "TOT_res"     "EVT 95"      0.95      red      reject    reject    accept    reject    reject    reject    reject
    % "TOT_res"     "EVT 99"      0.95      green    accept    accept    accept    accept    reject    reject    reject
        
% PortfolioID     VaRID      VaRLevel     TL       Bin       POF       TUFF       CC       CCI       TBF       TBFI 
   % ___________    ________    ________    _____    ______    ______    ______    ______    ______    ______    ______

     %"ENI_res"     "EVT 95"      0.95      green    accept    accept    accept    accept    accept    accept    accept
     %"ENI_res"     "EVT 99"      0.95      green    reject    reject    accept    reject    accept    reject    reject

%% Our solution WS = 500
%PortfolioID     VaRID      VaRLevel     TL       Bin       POF       TUFF       CC       CCI       TBF       TBFI 
    %___________    ________    ________    _____    ______    ______    ______    ______    ______    ______    ______

    % "CVX_res"     "EVT 95"      0.95      green    accept    accept    accept    accept    accept    accept    accept
    % "CVX_res"     "EVT 99"      0.95      green    reject    reject    accept    reject    accept    reject    reject

    % PortfolioID     VaRID      VaRLevel      TL       Bin       POF       TUFF       CC       CCI       TBF       TBFI 
   % ___________    ________    ________    ______    ______    ______    ______    ______    ______    ______    ______

     %"COP_res"     "EVT 95"      0.95      yellow    reject    reject    accept    reject    accept    reject    reject
     %"COP_res"     "EVT 99"      0.95      green     reject    reject    accept    reject    accept    reject    reject
    %  PortfolioID     VaRID      VaRLevel      TL       Bin       POF       TUFF       CC       CCI       TBF       TBFI 
  %  ___________    ________    ________    ______    ______    ______    ______    ______    ______    ______    ______

   %  "TOT_res"     "EVT 95"      0.95      yellow    reject    reject    accept    accept    accept    reject    reject
    % "TOT_res"     "EVT 99"      0.95      green     reject    reject    accept    reject    accept    reject    reject
%  PortfolioID     VaRID      VaRLevel     TL       Bin       POF       TUFF       CC       CCI       TBF       TBFI 
   % ___________    ________    ________    _____    ______    ______    ______    ______    ______    ______    ______

    % "ENI_res"     "EVT 95"      0.95      green    accept    accept    accept    accept    reject    reject    reject
    % "ENI_res"     "EVT 99"      0.95      green    reject    reject    accept    reject    accept    reject    reject


    