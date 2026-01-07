clc; clear all; close all

%total number of qubits
% N        = (10^3)*[1 2 3 4 6 8 10 20 30 40 50 60 80 ...
%            100 120 140 160 180 200 300 600 800 1000 ...
%            3000 6000 10^4 3*10^4 6*10^4 10^5 3*10^5 6*10^5 10^6];

N        = [logspace(3,5,50), logspace(5,6,50), logspace(6,9,10)];

%qber     = [0.0001 0.01:0.01:0.11]; %exact value of the qber
qber = linspace(0.0001, 0.11, 100);
l        = [0 10];         %array of distances in km to sample the key rate
cher_bnd = 10^(-5);        %epsilon: value of the Chernoff bound
hash     = 10^(-5);        %hashing failure
gamma    = 1.02;           %error correction inefficiency

%initialise tensors
R         = zeros(length(qber), length(N), length(l));  %N intependent rate
Rn        = zeros(length(qber), length(N), length(l));  %N dependent rate
Runc      = zeros(length(qber), length(N), length(l));  %Unc rel rate
qber_est1 = zeros(length(qber), length(N), length(l));  %Estimated qber 
qber_est2 = zeros(length(qber), length(N), length(l));  %Estimated qber
H         = zeros(length(qber), length(N), length(l));  %Von neumann relative entropy
H2        = zeros(length(qber), length(N), length(l));  %Shannon binary entropy  
pguess    = zeros(length(qber), length(N), length(l));  %guessing probability
max1      = zeros(length(qber), length(N), length(l));  %maxima
max2      = zeros(length(qber), length(N), length(l));
max3      = zeros(length(qber), length(N), length(l));

%initial cond
%cond=[0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9];
cond    = 0.05:0.05:0.9;
maxvec1 = zeros(length(cond),1);
maxvec2 = zeros(length(cond),1);
maxvec3 = zeros(length(cond),1);
rateval1= zeros(length(cond),1);
rateval2= zeros(length(cond),1);
rateval3= zeros(length(cond),1);
%%
tic;
for i=1:length(qber)
    for j=1:length(N)
        for k=1:length(l) 
            for m=1:length(cond)
            %maximization of the rate on the fraction of qubit qubits for error
            %estimation for both N-dip and N-indip key rate

            %iteration on different initial conditions
            %maxvec saves the optimal est_fraction wrt the initial condition
            [maxvec1(m), rateval1(m)] = fminsearch(@(f) -Rate_indip(qber(i), N(j), ...
                f, cher_bnd, hash, gamma, l(k)),cond(m));
            [maxvec2(m), rateval2(m)] = fminsearch(@(f) -Rate_dip_approx(qber(i), N(j), ...
                f, cher_bnd, hash, gamma,l(k)),cond(m));
            [maxvec3(m), rateval3(m)] = fminsearch(@(f) -Rate_unc_approx(qber(i), N(j), ...
                f, cher_bnd, hash, gamma, l(k)),cond(m));
            end
            
            %select the maximum wrt all possible initial condition
            %save the index of the initial condition I and thus the optimal
            %est_frac
            [max11, I1] = min(rateval1);
            [max22, I2] = min(rateval2);
            [max33, I3] = min(rateval3);
            
            %N-indip key rate
            R(i,j,k)    = -max11;
            %N-dip key rate
            Rn(i,j,k)   = -max22;
            %Unc rel key rate
            Runc(i,j,k) = -max33;

            %save the estimated qber, guessing prob, Relative and binary
            %entropy for fixed values of (exact) qber (i), number of
            %qubits (j), distance (k)

            %est quber for pguess
            qber_est1(i,j,k) = EstimQBER(qber(i), N(j), maxvec1(I1), cher_bnd, l(k));
            %est quber for rel entropy
            qber_est2(i,j,k) = EstimQBER(qber(i), N(j), maxvec2(I2), cher_bnd, l(k));
            %pguess(i,j,k) = real(pg);
            H2(i,j,k) = BinEntropy(qber_est2(i,j,k));
            H(i,j,k)  = 1-H2(i,j,k);
            %maxima (optimal estimation fraction)
            max1(i,j,k) = maxvec1(I1);
            max2(i,j,k) = maxvec2(I2);
            max3(i,j,k) = maxvec3(I3);
        end
    end
end

%Compute asymptotic key rates

R_asym  = zeros(length(qber),length(l));
Rn_asym = zeros(length(qber),length(l));

pguess_asym = zeros(length(qber),length(l));
H2_asym     = zeros(length(qber));
H_asym      = zeros(length(qber),length(l));

for i=1:length(qber)
        for k=1:length(l)

            [r, pg] = R_indip_asym(qber(i),l(k));

            pguess_asym(i,k) = pg;
            R_asym(i,k)      = r;

            H2_asym(i)   = BinEntropy(qber(i));
            H_asym(i,k)  = 1-H2_asym(i);

            Rn_asym(i,k) = R_dip_asym(qber(i),l(k));
        end
end
toc;

%%

%plot: rate vs block size for fixed distance d=l(2)=10km, 
% confront between: FME, AEP and UncRel key rates 
% Qber estimated with chernoff bound: Qber=0.03,0.06

% Define QBERs and distances to plot r(QBER, dist)
qber_choice = [0.03 0.06];
dist_choice = 2;

% Chiama la funzione driver
driver_rate_vs_blocksize(N, R, Rn, Runc, qber, qber_choice, dist_choice);

%%

%plot: rate vs QBER for fixed distance d=l(2)=10km, 
% confront between: FME, AEP and UncRel key rates 

% Define Block size and distance to plot r(N, dist)
target_N = 10^5;  % Il block size che cerchi
dist_idx = 2;     % 10 km

% Chiamata al driver
driver_rate_vs_qber(qber, R_asym, Rn_asym, R, Rn, Runc, N, target_N, dist_idx);

%%

% --- Inizializzazione Figura Asintotica ---
f_asym = figure('Color', 'w');
f_asym.Position = [100 100 900 650];

% --- Plotting ---
h1 = plot(qber, R_asym(:,2), 'Color', [0 0 1], 'LineWidth', 2.5); hold on;
h2 = plot(qber, Rn_asym(:,2), '--', 'Color', [1 0 0], 'LineWidth', 2.5);

% --- Configurazione Assi ---
ax = gca;
grid on; grid minor;
ax.FontSize = 20;
ax.TickLabelInterpreter = 'latex';

ax.XLim = [0 0.11];
ax.YLim = [0.0 0.7];
ax.YTick = 0:0.1:0.6;

% --- Etichette e Titoli ---
title('Asymptotic Key Rate vs QBER ($d=10\,$km)', ...
      'FontSize', 24, 'Interpreter', 'latex');

xlabel('$Q$', 'FontSize', 30, 'Interpreter', 'latex');
ylabel(''); % Rimosso label Y

% --- Legenda ---
legend([h1, h2], {'$r_\infty^{\textrm{FME}}$', '$r_\infty^{\textrm{AEP}}$'}, ...
    'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 22);

% --- Raffinamento ---
box on;
set(ax, 'LineWidth', 1.2);


%%

% --- Inizializzazione Figura Regime Finito ---
f_fin = figure('Color', 'w');
f_fin.Position = [100 100 900 650];

% --- Plotting ---
h1 = plot(qber, R(:,50,2), 'Color', [0 0 1], 'LineWidth', 2.5); hold on;
h2 = plot(qber, Rn(:,50,2), '--', 'Color', [1 0 0], 'LineWidth', 2.5);
h3 = plot(qber, Runc(:,50,2), ':', 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2.5);

% --- Configurazione Assi ---
ax = gca;
grid on; grid minor;
ax.FontSize = 20;
ax.TickLabelInterpreter = 'latex';

ax.XLim = [0.0 0.11];
ax.XTick = 0:0.02:0.1;
ax.YLim = [0.0 0.35]; % Leggermente aumentato per non tagliare il titolo
ax.YTick = 0:0.1:0.3;

% --- Etichette e Titoli ---
title('Finite Regime Key Rate ($N=10^5$, $d=10\,$km)', ...
      'FontSize', 24, 'Interpreter', 'latex');

xlabel('$Q$', 'FontSize', 30, 'Interpreter', 'latex');
ylabel(''); % Rimosso label Y

% --- Legenda ---
legend([h1, h2, h3], {'$r_N^{\textrm{FME}}$', '$r_N^{\textrm{AEP}}$', '$r_N^{\textrm{EUR}}$'}, ...
    'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 22);

% --- Raffinamento ---
box on;
set(ax, 'LineWidth', 1.2);

%%

%plot: rate vs #qubits for fixed distance d=l(2)=10km for different values of error probability: 
% confront between: N dependent/independent key rates 
% Qber estimated with chernoff bound 0<Qber<0.04

f1=figure;
f1.Position = [100 100 850 550];%dimensione grafico

semilogx(N,Rn(1,:,2),'-*','color',[1 0 0],'LineWidth',1.5);
hold on
semilogx(N,Rn(2,:,2),'-*','color',[0.4660 0.6740 0.1880],'LineWidth',1.5);
hold on
semilogx(N,Rn(3,:,2),'-*','color',[0.4940 0.1840 0.5560],'LineWidth',1.5);


semilogx(N,R(1,:,2),'-','color',[1 0 0],'LineWidth',1.5);
hold on
semilogx(N,R(2,:,2),'-','color',[0.4660 0.6740 0.1880],'LineWidth',1.5);
hold on
semilogx(N,R(3,:,2),'-','color',[0.4940 0.1840 0.5560],'LineWidth',1.5);




set(gca,'FontSize',18);%dimensione font dei valori sugli assi
set(gca,'XLim',[600 10^9]);
set(gca,'YLim',[0.0 0.6],'YTick',0:0.1:0.7);


   title('Key rate vs number of qubits for estimated $Qber<0.04$ (distance: 10km)', ...
     'FontSize',22,'FontName','Computer Modern','Interpreter','latex', ...
     'Color',[1 0 0],Position=[8*10^5 0.61 0]);
   xlabel('$N$','FontSize',24,'FontName','Computer Modern','Interpreter','latex', ...
       'Color',[0 0 1] ...
    ,Position=[3*10^9 -0.03 -1]);
   ylabel('$r_N$','FontSize',24,'FontName','Computer Modern','Interpreter','latex', ...
       'Color',[0 0 1],'Rotation',0 ...
    ,Position=[150 0.60 -1]);

   lgd2=legend('Qber=0.01','Qber=0.02', ...
    'Qber=0.03');
   fontsize(lgd2,20,'points');
   set(lgd2,'FontName','Computer Modern','Interpreter','latex');

%%

%plot: rate vs #qubits for fixed distance d=l(2)=10km for different values of error probability: 
% confront between: N dependent/independent key rates 
% Qber estimated with chernoff bound 0.04<Qber<0.06

f2=figure;
f2.Position = [100 100 850 550];%dimensione grafico



semilogx(N,Rn(4,:,2),'-*','color',[0 0 1],'LineWidth',1.5);
hold on
semilogx(N,Rn(5,:,2),'-*','color',[0.9290 0.6940 0.1250],'LineWidth',1.5);
hold on
semilogx(N,Rn(6,:,2),'-*','color',[0 0 0],'LineWidth',1.5);
hold on

semilogx(N,R(4,:,2),'-','color',[0 0 1],'LineWidth',1.5);
hold on
semilogx(N,R(5,:,2),'-','color',[0.9290 0.6940 0.1250],'LineWidth',1.5);
hold on
semilogx(N,R(6,:,2),'-','color',[0 0 0],'LineWidth',1.5);
hold on

set(gca,'FontSize',18);%dimensione font dei valori sugli assi
set(gca,'XLim',[600 10^9]);
set(gca,'YLim',[0.0 0.6],'YTick',0:0.1:0.7);


 title('Key rate vs number of qubits for estimated $Qber\ge0.04$ (distance: 10km)', ...
     'FontSize',22,'FontName','Computer Modern','Interpreter','latex', ...
     'Color',[1 0 0],Position=[8*10^5 0.61 0]);
   xlabel('$N$','FontSize',24,'FontName','Computer Modern','Interpreter','latex', ...
       'Color',[0 0 1] ...
    ,Position=[3*10^9 -0.03 -1]);
   ylabel('$r_N$','FontSize',24,'FontName','Computer Modern','Interpreter','latex', ...
       'Color',[0 0 1],'Rotation',0 ...
    ,Position=[150 0.6 -1]);


lgd3=legend('Qber=0.04','Qber=0.05','Qber=0.06');
   fontsize(lgd3,20,'points');
   set(lgd3,'FontName','Computer Modern','Interpreter','latex');

   %%

f3=figure;
f3.Position = [100 100 850 550];%dimensione grafico

semilogx(N,qber_est1(1,:,2),'-*','color',[1 0 0],'LineWidth',1.5);
hold on
semilogx(N,qber_est1(2,:,2),'-*','color',[0.4660 0.6740 0.1880],'LineWidth',1.5);
hold on
semilogx(N,qber_est1(3,:,2),'-*','color',[0.4940 0.1840 0.5560],'LineWidth',1.5);
hold on
semilogx(N,qber_est1(4,:,2),'-*','color',[0 0 1],'LineWidth',1.5);
hold on
semilogx(N,qber_est1(5,:,2),'-*','color',[0.9290 0.6940 0.1250],'LineWidth',1.5);
hold on
semilogx(N,qber_est1(6,:,2),'-*','color',[0 0 0],'LineWidth',1.5);

lgd4=legend('Qber=0.01','Qber=0.02', ...
    'Qber=0.03','Qber=0.04','Qber=0.05','Qber=0.06');
   fontsize(lgd4,20,'points');
   set(lgd4,'FontName','Computer Modern','Interpreter','latex');

%%

f4=figure;
f4.Position = [100 100 850 550];%dimensione grafico

semilogx(N,qber_est2(1,:,2),'-*','color',[1 0 0],'LineWidth',1.5);
hold on
semilogx(N,qber_est2(2,:,2),'-*','color',[0.4660 0.6740 0.1880],'LineWidth',1.5);
hold on
semilogx(N,qber_est2(3,:,2),'-*','color',[0.4940 0.1840 0.5560],'LineWidth',1.5);
hold on
semilogx(N,qber_est2(4,:,2),'-*','color',[0 0 1],'LineWidth',1.5);
hold on
semilogx(N,qber_est2(5,:,2),'-*','color',[0.9290 0.6940 0.1250],'LineWidth',1.5);
hold on
semilogx(N,qber_est2(6,:,2),'-*','color',[0 0 0],'LineWidth',1.5);

lgd5=legend('Qber=0.01','Qber=0.02', ...
    'Qber=0.03','Qber=0.04','Qber=0.05','Qber=0.06');
   fontsize(lgd5,20,'points');
   set(lgd5,'FontName','Computer Modern','Interpreter','latex');



%%
% 
% f2=figure;
% 
% plot(qber_est(:,21,2),H(:,21,2),'-*','color',[0 0 1],'LineWidth',1.5);
% hold on
% plot(qber_est(:,21,2),1-H2(:,21,2),'-o','color',[1 0 0],'LineWidth',1.5);
% 
% f3=figure;
% plot(qber_est(:,21,2),1-2*H2(:,21,2),'-o','color',[1 0 0],'LineWidth',1.5);
% hold on
% plot(qber_est(:,21,2),Rn(:,21,2),'-o','color',[1 0 0],'LineWidth',1.5);
% 
% 
% 
% %plot(qber_est(:,22,2),pguess(:,22,2),'-*','color',[0 0 1],'LineWidth',1.5);


%%

% ========================================================================
%  HELPER: FINITE SIZE RATE CALCULATION
% =========================================================================

function pguess = GuessingProb(qber, N, f, cher_bnd, dist)

%function for computing the guessing probability via SDP; returns also the
%optimal state ro

%estimated qber
qber_est = EstimQBER(qber, N, f, cher_bnd, dist);

%guessing probability
pguess   = 0.5+ sqrt(qber_est*(1-qber_est));

end

function h2 = BinEntropy(p)
%function for the computation of the binary entropy

    if p~=0
        h2= -p*log2(p)-(1-p)*log2(1-p);
    else
        h2=0;
    end

end

function H = RelEntropy (A,B)
%function for the computation the Von Neumann relative entropy

    H=real(trace(A * ( (logm(A)./log(2)) - (logm(B)./log(2)) )));

end

function Z_A = checkboard(A)
%function for the computation the checkboard matrix Z_A of a given 4x4
%matrix A

        A(1,2) = 0;
        A(2,1) = 0;
        A(1,4) = 0;
        A(4,1) = 0;
        A(2,3) = 0;
        A(3,2) = 0;
        A(3,4) = 0;
        A(4,3) = 0;

        Z_A = A;

end

function r_dip_ex = Rate_dip_ex(qber, N, f, cher_bnd, dist, RO)
%function for the computation the N dependent rate (exact)

a = 0.2;                %specific attenuation
eta = 10^(-0.1*a*dist); %trassmissivity
Nerr = N*eta*f;         %number of qubits for error estimation
Nkey = N*eta*(1-f);     %number of qubits for key construction

qber_est = EstimQBER(qber, N, f, cher_bnd, dist); %estimated qber


epsilon  = 10^(-10); %parameters for N-depending rate 
Delta    = 4*log2(2+sqrt(2))*sqrt(log2(2/epsilon^2));

r_dip_ex = max ( (Nkey/N)*(  RelEntropy(RO,checkboard(RO))...
    -BinEntropy(qber_est) - Delta*(Nkey)^(-0.5)  ) ,0 );
  
end

function r_ind = Rate_indip(qber, N, f, cher_bnd, hash, gamma, dist)
%function for the computation the the N independent rate; return also the
%optimal state RO and the guessing probability associated pg

a    = 0.2;              %specific attenuation
eta  = 10^(-0.1*a*dist); %trassmissivity
Nkey = N*eta*(1-f);      %number of qubits for key construction

qber_est = EstimQBER(qber, N, f, cher_bnd, dist);  %estimated qber

pg       = GuessingProb(qber, N, f, cher_bnd, dist); 
r_ind    = max((Nkey/N)*( -log2(pg) - gamma*BinEntropy(qber_est) )+ ...
    2*log2(sqrt(2)*hash)/(Nkey),0);

%r_ind = (Nkey/N)*( GuessingProb(qber, N, f, cher_bnd) - BinEntropy(qber_est) );
end

function r_dip_app = Rate_dip_approx(qber, N, f, cher_bnd,hash, gamma, dist)
%function for the computation the N dependent rate (approximated)

a    = 0.2;               %specific attenuation
eta  = 10^(-0.1*a*dist);  %trassmissivity
Nerr = N*eta*f;           %number of qubits for error estimation
Nkey = N*eta*(1-f);       %number of qubits for key construction

qber_est = EstimQBER(qber, N, f, cher_bnd, dist); %estimated qber

epsilon = 10^(-10);  %parameters for N-depending rate 
Delta   = 4*log2(2+sqrt(2))*sqrt(log2(2/epsilon^2));

r_dip_app = max ( (Nkey/N)*(  1 -(1+gamma)*BinEntropy(qber_est)...
    - Delta*(Nkey)^(-0.5)  ) +  2*log2(sqrt(2)*hash)/(Nkey),0 );

end

function r_unc_app = Rate_unc_approx(qber, N, f, cher_bnd,hash, gamma, dist)
%function for the computation the N dependent rate (approximated)

a    = 0.2;              %specific attenuation
eta  = 10^(-0.1*a*dist); %trassmissivity
Nerr = N*eta*f;          %number of qubits for error estimation
Nkey = N*eta*(1-f);      %number of qubits for key construction

qber_est = EstimQBER(qber, N, f, cher_bnd, dist); %estimated qber

r_unc_app = max ( (Nkey/N)*(  1 -(1+gamma)*BinEntropy(qber_est))+...
    2*log2(sqrt(2)*hash)/(Nkey),0);

end

function QBER = EstimQBER(qber, N, f, cher_bnd, dist)
%function for the computation the estimated qber
    a    = 0.2;
    eta  = 10^(-0.1*a*dist);
    QBER = qber + sqrt( 1/(eta*N*f) * (log(2/cher_bnd)) );
end

function pguess = GuessingProb_asym(qber)
%function for asymptotic pguess
pguess=0.5+sqrt(qber*(1-qber));
end

function H = RelEntropy_app(qber)
%function for Relative entropy approximated
H=1-BinEntropy(qber);
end

function [r_ind_asym, pg] = R_indip_asym(qber, dist)
%function for rate N indipendent asymptotic

a=0.2;
eta=10^(-0.1*a*dist);
pg=GuessingProb_asym(qber);
r_ind_asym=max(eta*(-log2(pg)-BinEntropy(qber)),0);

end

function r_dip_asym = R_dip_asym(qber, dist)
%function for rate N dipendent asymptotic
a=0.2;

eta=10^(-0.1*a*dist);

r_dip_asym=max(eta*(1-2*BinEntropy(qber)),0);


end


% ========================================================================
%  DRIVER FUNCTIONS (Plotting)
% =========================================================================
function driver_rate_vs_blocksize(N, R, Rn, Runc, qber, target_qbers, d_idx)
% driver_rate_vs_blocksize - Genera plot del key rate vs block size per QBER specifici
%
% Sintassi: 
%   driver_rate_vs_blocksize(N, R, Rn, Runc, qber, target_qbers, d_idx)
%
% Input:
%   N            - Vettore dei block size
%   R, Rn, Runc  - Tensori dei rate (QBER x N x Distanza)
%   qber         - Vettore di tutti i QBER campionati nel calcolo
%   target_qbers - Vettore dei QBER desiderati (es. [0.03, 0.06])
%   d_idx        - Indice della distanza fissa da plottare (es. 2 per 10km)

    for q_target = target_qbers
        % --- Ricerca del valore QBER più vicino nell'array ---
        [~, q_idx] = min(abs(qber - q_target));
        q_actual = qber(q_idx); % Il valore reale trovato nel vettore qber
        
        % --- Creazione Figura ---
        figure('Color', 'w', 'Position', [100 100 900 650]);
        
        % --- Tracciamento Dati ---
        h1 = loglog(N, R(q_idx, :, d_idx), 'Color', [0 0 1], 'LineWidth', 2.5); hold on;
        h2 = loglog(N, Rn(q_idx, :, d_idx), '--', 'Color', [1 0 0], 'LineWidth', 2.5);
        h3 = loglog(N, Runc(q_idx, :, d_idx), ':', 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2.5);

        % --- Configurazione Assi ---
        ax = gca;
        grid on; grid minor;
        ax.FontSize = 20;
        ax.TickLabelInterpreter = 'latex';
        
        % Limiti specifici basati sul valore di QBER effettivo
        if q_actual <= 0.04
            ax.XLim = [1000 10^9];
            ax.YLim = [10^-3 0.5];
        else
            ax.XLim = [10^4 10^9];
            ax.YLim = [10^-4 0.5];
        end

        % --- Etichette e Titoli ---
        % Mostra il valore effettivo q_actual con 2 decimali
        title_str = sprintf('Key rate vs Block size -- QBER $= %.2f$ ($d=10\\,$km)', q_actual);
        title(title_str, 'FontSize', 24, 'Interpreter', 'latex');
        
        xlabel('$N$', 'FontSize', 30, 'Interpreter', 'latex');
        ylabel(''); % Asse Y senza label come richiesto

        % --- Legenda ---
        legend([h1, h2, h3], {'$r_N^{\textrm{FME}}$', '$r_N^{\textrm{AEP}}$', '$r_N^{\textrm{EUR}}$'}, ...
            'Location', 'southeast', 'Interpreter', 'latex', 'FontSize', 22);

        % --- Ottimizzazione Layout ---
        box on;
        set(ax, 'LineWidth', 1.2);
    end
end

function driver_rate_vs_qber(qber, R_asym, Rn_asym, R, Rn, Runc, N, N_target, dist_idx)

% driver_rate_vs_qber - Genera i plot del rate vs QBER (Asintotico e Finite Size)
%
%
% Input:
%   qber      - Vettore dei valori di QBER
%   R_asym    - Rate asintotico FME
%   Rn_asym   - Rate asintotico AEP
%   R,Rn,Runc - Tensori dei rate (QBER x N x Distanza)
%   N         - Vettore dei block size calcolati
%   N_target  - Block size desiderato per il plot finito
%   dist_idx  - Indice della distanza 

    %% 1. Ricerca del Block Size più vicino
    [~, n_idx] = min(abs(N - N_target));
    N_actual = N(n_idx);
    
    % Formattazione scientifica per il titolo (es. 10^5)
    pow10 = log10(N_actual);
    if mod(pow10, 1) == 0
        n_str = sprintf('10^{%.0f}', pow10);
    else
        n_str = sprintf('%.2e', N_actual);
    end

    %% 2. Grafico Limite Asintotico
    f_asym = figure('Color', 'w', 'Position', [100 100 900 650]);
    h1 = plot(qber, R_asym(:, dist_idx), 'Color', [0 0 1], 'LineWidth', 2.5); hold on;
    h2 = plot(qber, Rn_asym(:, dist_idx), '--', 'Color', [1 0 0], 'LineWidth', 2.5);
    
    ax1 = gca;
    grid on; grid minor;
    ax1.FontSize = 20;
    ax1.TickLabelInterpreter = 'latex';
    ax1.XLim = [0 0.11];
    ax1.YLim = [0.0 0.7];
    ax1.YTick = 0:0.1:0.7;
    
    title('Key Rate vs QBER ($N=\infty$, $d=10\,$km)', 'FontSize', 24, 'Interpreter', 'latex');
    xlabel('$\textrm{QBER}$', 'FontSize', 30, 'Interpreter', 'latex');
    ylabel('');
    
    legend([h1, h2], {'$r_\infty^{\textrm{FME}}$', '$r_\infty^{\textrm{AEP}}$'}, ...
        'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 22);
    box on; set(ax1, 'LineWidth', 1.2);

    %% 3. Grafico Regime Finito
    f_fin = figure('Color', 'w', 'Position', [100 100 900 650]);
    hf1 = plot(qber, R(:, n_idx, dist_idx), 'Color', [0 0 1], 'LineWidth', 2.5); hold on;
    hf2 = plot(qber, Rn(:, n_idx, dist_idx), '--', 'Color', [1 0 0], 'LineWidth', 2.5);
    hf3 = plot(qber, Runc(:, n_idx, dist_idx), ':', 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2.5);
    
    ax2 = gca;
    grid on; grid minor;
    ax2.FontSize = 20;
    ax2.TickLabelInterpreter = 'latex';
    ax2.XLim = [0.0 0.11];
    ax2.XTick = 0:0.02:0.1;
    ax2.YLim = [0.0 0.35];
    ax2.YTick = 0:0.1:0.3;
    
    title_fin = sprintf('Key Rate vs QBER ($N=%s$, $d=10\\,$km)', n_str);
    title(title_fin, 'FontSize', 24, 'Interpreter', 'latex');
    xlabel('$\textrm{QBER}$', 'FontSize', 30, 'Interpreter', 'latex');
    ylabel('');
    
    legend([hf1, hf2, hf3], {'$r_N^{\textrm{FME}}$', '$r_N^{\textrm{AEP}}$', '$r_N^{\textrm{EUR}}$'}, ...
        'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 22);
    box on; set(ax2, 'LineWidth', 1.2);
end