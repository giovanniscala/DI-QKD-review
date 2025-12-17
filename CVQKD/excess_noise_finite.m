%% MAIN SCRIPT: Finite-Size Key Rate for BPSK and QPSK
clear; clc; %close all;

% --- Preset parameters ----------------
dimE      = 10;                         % Local dim of each Eve mode 
distances = [0.0001 5:5:250];         % distance in km
%etas      = [0.0001, 0.001, 0.01, 0.1:0.05:0.95, 0.999];
etas      = 10.^(-0.02 * distances);    % Transmittance 
alphasB   = [0.5];                      % BPSK amplitudes
alphasQ   = [0.85];                     % QPSK amplitudes
xis       = [0.000000001, 0.01];        % Excess-noise parameter
a_list    = [1.05];                      % Renyi parameter
r_test    = 0.;                        % Fraction of bits for parameter estimation
epsilon   = 1.0e-10;                    % Security parameter epsilon
epsilonp  = 1.0e-10;                    % Security parameter epsilon'

% Block sizes for finite-size analysis
block_size = logspace(3, 10, 20);        % From 10^3 to 10^10
%block_size = [10^5, 10^7];

% Dimensions
nD      = numel(distances);
%nD = numel(etas);
nAlphaB = numel(alphasB);
nAlphaQ = numel(alphasQ);
nXi     = numel(xis);
nA      = numel(a_list);
nN      = numel(block_size);

%% ========================================================================
%  1. BPSK COMPUTATION LOOP
%  ========================================================================
fprintf('--- Starting BPSK Calculations (Finite Size) ---\n');

% Preallocation BPSK Arrays
% Indices: (Distance, Alpha, Xi, BlockSize) for vN
% Indices: (Distance, Alpha, Xi, BlockSize, Renyi_Index) for Sand
B_Rate_vN     = zeros(nD, nAlphaB, nXi, nN);
B_Rate_Sand   = zeros(nD, nAlphaB, nXi, nN, nA);

% Intermediate storage for entropy and leakage (independent of n)
B_Ent_vN      = zeros(nD, nAlphaB, nXi);
B_Ent_Sand    = zeros(nD, nAlphaB, nXi, nA);
B_Leakage     = zeros(nD, nAlphaB, nXi);

% Analytic results (Pure Loss)
B_Rate_Ana_Asymp = zeros(nD, nAlphaB, nA); 

tic;
for ia = 1:nAlphaB
    alpha = alphasB(ia);
    
    % --- A) Calcolo ANALITICO (Pure Loss Reference - Asymptotic) ---
    % Utile per confronto
    for id = 1:nD
        eta = etas(id);
        [~, ~, ~, p_y_pure] = build_rho_YE_BPSK(dimE, eta, 0, alpha);
        leak_pure = leakage_BPSK(p_y_pure);
        for ir = 1:nA
            a_val = a_list(ir);
            h_ana_bits = Analytic_SandDown(eta, alpha, a_val);
            B_Rate_Ana_Asymp(id, ia, ir) = (1-r_test)*(h_ana_bits - leak_pure);
        end
    end
    
    % --- B) Calcolo NUMERICO (Loop su Xi) ---
    for im = 1:nXi
        xi = xis(im);
        fprintf('  BPSK Numeric: Alpha=%.2f, Xi=%.3f\n', alpha, xi);
        
        for id = 1:nD
            eta = etas(id);
            if eta >= 1, mu = 0; else, mu = eta*xi / (2 * (1 - eta)); end
            
            % 1. Generazione Stato e Quantità Base
            [rhoYE, ~, ~, p_y_given_x] = build_rho_YE_BPSK(dimE, eta, mu, alpha);
            rhoE = compute_rho_E_from_rhoYE(rhoYE, 2);
            
            leak = leakage_BPSK(p_y_given_x);
            B_Leakage(id, ia, im) = leak;
            
            % 2. Entropie
            h_vn = conditional_vN_entropy(rhoYE, rhoE);
            B_Ent_vN(id, ia, im) = h_vn;
            
            % 3. Rate Finite Size (von Neumann / AEP)
            for in = 1:nN
                n = block_size(in);
                B_Rate_vN(id, ia, im, in) = compute_rate_finite_size(...
                    h_vn, leak, dimE, eta, mu, 2, n, [], ...
                    epsilon, epsilonp, r_test, 'aep');
            end
            
            % 4. Rate Finite Size (Sandwiched Renyi)
            for ir = 1:nA
                a_val = a_list(ir);
                h_sand = conditional_SandDown_entropy(rhoYE, rhoE, a_val);
                B_Ent_Sand(id, ia, im, ir) = h_sand;
                
                for in = 1:nN
                    n = block_size(in);
                    B_Rate_Sand(id, ia, im, in, ir) = compute_rate_finite_size(...
                        h_sand, leak, dimE, eta, mu, 2, n, a_val, ...
                        epsilon, epsilonp, r_test, 'renyi');
                end
            end
        end
    end
end
toc;

%% ========================================================================
%  2. QPSK COMPUTATION LOOP
%  ========================================================================
fprintf('\n--- Starting QPSK Calculations (Finite Size) ---\n');

% Preallocation QPSK Arrays
Q_Rate_vN     = zeros(nD, nAlphaQ, nXi, nN);
Q_Rate_Sand   = zeros(nD, nAlphaQ, nXi, nN, nA);

Q_Ent_vN      = zeros(nD, nAlphaQ, nXi);
Q_Ent_Sand    = zeros(nD, nAlphaQ, nXi, nA);
Q_Leakage     = zeros(nD, nAlphaQ, nXi);

tic;
for ia = 1:nAlphaQ
    alpha = alphasQ(ia);
    
    for im = 1:nXi
        xi = xis(im);
        fprintf('  QPSK Numeric: Alpha=%.2f, Xi=%.3f\n', alpha, xi);
        
        for id = 1:nD
            eta = etas(id);
            if eta >= 1, mu = 0; else, mu = eta*xi / (2 * (1 - eta)); end
            
            % 1. Generazione Stato
            [rhoYE, ~, ~, p_y_given_k] = build_rho_YE_QPSK(dimE, eta, mu, alpha);
            rhoE = compute_rho_E_from_rhoYE(rhoYE, 4);
            
            leak = leakage_QPSK(p_y_given_k);
            Q_Leakage(id, ia, im) = leak;
            
            % 2. Entropie
            h_vn = conditional_vN_entropy(rhoYE, rhoE);
            Q_Ent_vN(id, ia, im) = h_vn;
            
            % 3. Rate Finite Size (von Neumann / AEP)
            for in = 1:nN
                n = block_size(in);
                Q_Rate_vN(id, ia, im, in) = compute_rate_finite_size(...
                    h_vn, leak, dimE, eta, mu, 4, n, [], ...
                    epsilon, epsilonp, r_test, 'aep');
            end
            
            % 4. Rate Finite Size (Sandwiched Renyi)
            for ir = 1:nA
                a_val = a_list(ir);
                h_sand = conditional_SandDown_entropy(rhoYE, rhoE, a_val);
                Q_Ent_Sand(id, ia, im, ir) = h_sand;
                
                for in = 1:nN
                    n = block_size(in);
                    Q_Rate_Sand(id, ia, im, in, ir) = compute_rate_finite_size(...
                        h_sand, leak, dimE, eta, mu, 4, n, a_val, ...
                        epsilon, epsilonp, r_test, 'renyi');
                end
            end
        end
    end
end
toc;

%% ========================================================================
%  3. PLOTTING DRIVERS CALL
%  ========================================================================
fprintf('\n--- Generating Plots ---\n');

% Selezioniamo un caso specifico da plottare (es. xi = 0.01, alpha fisso)
xi_idx = 2; % Indice per xi=0.01 (se presente nel vettore xis)
if xi_idx > nXi, xi_idx = 1; end

alphaB_idx = 1;
alphaQ_idx = 1;
renyi_idx  = 1; % Primo valore di 'a' in a_list

% --- Rate vs Distance ---
driver_finite_size_rate_BPSK(distances, block_size, ...
    B_Rate_vN(:, alphaB_idx, xi_idx, :), ...
    B_Rate_Sand(:, alphaB_idx, xi_idx, :, renyi_idx), ...
    alphasB(alphaB_idx), xis(xi_idx), a_list(renyi_idx));

driver_finite_size_rate_QPSK(distances, block_size, ...
    Q_Rate_vN(:, alphaQ_idx, xi_idx, :), ...
    Q_Rate_Sand(:, alphaQ_idx, xi_idx, :, renyi_idx), ...
    alphasQ(alphaQ_idx), xis(xi_idx), a_list(renyi_idx));

driver_finite_size_rate_QPSK_with_ref(distances, block_size, ...
    Q_Rate_vN(:, alphaQ_idx, xi_idx, :), ...
    Q_Rate_Sand(:, alphaQ_idx, xi_idx, :, renyi_idx), ...
    alphasQ(alphaQ_idx), xis(xi_idx), a_list(renyi_idx));

%%

fprintf('\n--- Generating Plots ---\n');

% Selezioniamo un caso specifico da plottare (es. xi = 0.01, alpha fisso)
xi_idx = 2; % Indice per xi=0.01 (se presente nel vettore xis)
if xi_idx > nXi, xi_idx = 1; end

alphaB_idx = 1;
alphaQ_idx = 1;
renyi_idx  = 1; % Primo valore di 'a' in a_list

% --- Rate vs Distance ---
driver_finite_size_rate_BPSK_eta(etas, block_size, ...
    B_Rate_vN(:, alphaB_idx, xi_idx, :), ...
    B_Rate_Sand(:, alphaB_idx, xi_idx, :, renyi_idx), ...
    alphasB(alphaB_idx), xis(xi_idx), a_list(renyi_idx));

driver_finite_size_rate_QPSK_eta(etas, block_size, ...
    Q_Rate_vN(:, alphaQ_idx, xi_idx, :), ...
    Q_Rate_Sand(:, alphaQ_idx, xi_idx, :, renyi_idx), ...
    alphasQ(alphaQ_idx), xis(xi_idx), a_list(renyi_idx));

%%

% --- Rate vs Block Size (fixed distance = 10km) ---
fixed_dist_km = 10;
% Find index of distance closest to 10 km
[~, id_dist] = min(abs(distances - fixed_dist_km));

driver_finite_size_vs_blocksize_BPSK(block_size, ...
    B_Rate_vN(id_dist, alphaB_idx, xi_idx, :), ...
    B_Rate_Sand(id_dist, alphaB_idx, xi_idx, :, renyi_idx), ...
    distances(id_dist), alphasB(alphaB_idx), ...
    xis(xi_idx), a_list(renyi_idx));

driver_finite_size_vs_blocksize_QPSK(block_size, ...
    Q_Rate_vN(id_dist, alphaQ_idx, xi_idx, :), ...
    Q_Rate_Sand(id_dist, alphaQ_idx, xi_idx, :, renyi_idx), ...
    distances(id_dist), alphasQ(alphaQ_idx), ...
    xis(xi_idx), a_list(renyi_idx));

%% ========================================================================
%  HELPER: FINITE SIZE RATE CALCULATION
% =========================================================================

function r = compute_rate_finite_size(H, leak, dimE, eta, mu, dimY, n, a, epsilon, epsilonp, r_test, type)
%COMPUTE_RATE_FINITE_SIZE Calcola il rate con correzioni finite-size.
%   type: 'aep' (von Neumann) o 'renyi' (Sandwiched)

    if isnan(H), r = NaN; return; end

    % --- Common Finite Size Corrections ---
    % 1. Parameter Estimation Cost (approximate/standard)
    fact = 1 + 2*log2(1/epsilonp); % Simplified term for parameter est failure prob
    fact_n = fact/n;
    
    % 2. Hilbert Space Truncation Correction (Delta_omega)
    % Probability of photon number > N_cutoff (dimE-1)
    
    % Eve's local mean photon number (approx for truncation bound)
    term_eve = ( (1-eta)*mu ) / ( 1+(1-eta)*mu );
    omega = term_eve^dimE; 
    
    % Correction term Delta(omega)
    omega_arg = sqrt(omega)/(1+sqrt(omega));
    h2_omega = BinEntropy(omega_arg);
    
    if dimY == 4 % QPSK
        Delta_omega = 2*sqrt(omega) + (1+sqrt(omega))*h2_omega;
    else % BPSK
        Delta_omega = sqrt(omega) + (1+sqrt(omega))*h2_omega;
    end
    %Delta_omega_n = Delta_omega/n;

    % --- Specific Corrections ---
    if strcmp(type, 'aep')
        % AEP Correction
        if dimY == 4
             delta_eps = 2 * log2(7) * sqrt( log2( 2/epsilon ) );
        else
             delta_eps = 2 * log2(5) * sqrt( log2( 2/epsilon ) );
        end
        delta_eps_n = delta_eps/sqrt(n);
        
        r = (1-r_test) * (H - leak - delta_eps_n - Delta_omega) - fact_n;

    elseif strcmp(type, 'renyi')
        % Renyi Correction (Generalized Entropy Accumulation / AEP)
        g_a_eps = log2(2/epsilon^2)/(a-1);
        g_eps_n = g_a_eps/n;
        
        r = (1-r_test) * (H - leak - g_eps_n - Delta_omega) - fact_n;
    else
        error('Unknown rate type');
    end
    
    % Rate cannot be negative (physically) -> 0
    %Typically clamp to NaN for log plots.
    if r < 0, r = NaN; end
end

%% ========================================================================
%  NEW DRIVER FUNCTIONS (Plotting)
% =========================================================================

function driver_finite_size_rate_BPSK(distances, block_size, Rate_vN, Rate_Sand, alpha, xi, a)
%DRIVER_FINITE_SIZE_RATE_BPSK Plot Key Rate vs Distance for different n.
%   Rate_vN: Matrix (nD x 1 x 1 x nN) -> squeeze to (nD x nN)
%   Rate_Sand: Matrix (nD x 1 x 1 x nN x 1) -> squeeze to (nD x nN)
    R_vN_sq   = squeeze(Rate_vN);
    R_Sand_sq = squeeze(Rate_Sand);
    etas      = 10.^(-0.02 * distances);
    
    nN = length(block_size);
    colors = lines(nN);
    
    figure('Name', sprintf('BPSK Finite Size (alpha=%.2f, xi=%.3f)', alpha, xi)); 
    hold on; box on; grid on;
    
    % IMPOSTAZIONE FONT DEI TICK (Times New Roman)
    set(gca, 'FontName', 'Times New Roman','FontSize', 18);
    
    xlabel('$d$ [km]', 'Interpreter', 'latex','FontSize', 18); 
    % Etichetta asse Y rimossa come da richiesta precedente
    
    % Plot von Neumann (AEP) curves
    for i = 1:nN
        plot(distances, R_vN_sq(:, i), '--*', 'Color', colors(i,:), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('$r_{\\epsilon'''',\\omega}^{AEP}$ ($n=10^{%.0f}$)', log10(block_size(i))));
    end
    
    % Plot Sandwiched Renyi curves
    for i = 1:nN
        plot(distances, R_Sand_sq(:, i), '-o', 'Color', colors(i,:), 'LineWidth', 1.2, ...
            'DisplayName', sprintf('$r_{\\epsilon'''', \\omega}^{SD}$ ($n=10^{%.0f}$)', log10(block_size(i))));
    end
    
    set(gca, 'YScale', 'log'); 
    %set(gca, 'XScale', 'log'); 
    ylim([1e-6, 1]);
    
    % IMPOSTAZIONE LEGENDA (Alto a Destra, Carattere Aumentato)
    legend('show', 'Location', 'NorthEast', 'Interpreter', 'latex', 'FontSize', 20); 
end

function driver_finite_size_rate_QPSK(distances, block_size, Rate_vN, Rate_Sand, alpha, xi, a)
%DRIVER_FINITE_SIZE_RATE_QPSK Plot Key Rate vs Distance for different n.
    R_vN_sq   = squeeze(Rate_vN);
    R_Sand_sq = squeeze(Rate_Sand);
    
    nN = length(block_size);
    colors = lines(nN);
    
    figure('Name', sprintf('QPSK Finite Size (alpha=%.2f, xi=%.3f)', alpha, xi)); 
    hold on; box on; grid on;
    
    % IMPOSTAZIONE FONT DEI TICK (Times New Roman)
    set(gca, 'FontName', 'Times New Roman','FontSize', 14);
    
    xlabel('$d$ [km]', 'Interpreter', 'latex','FontSize', 14);
    
    % Plot von Neumann (AEP) curves
    for i = 1:nN
        plot(distances, R_vN_sq(:, i), '--', 'Color', colors(i,:), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('$r_{\\epsilon''}^{AEP}$ ($n=10^{%.0f}$)', log10(block_size(i))));
    end
    
    % Plot Sandwiched Renyi curves
    for i = 1:nN
        plot(distances, R_Sand_sq(:, i), '-o', 'Color', colors(i,:), 'LineWidth', 1.2, ...
            'DisplayName', sprintf('$r_{\\epsilon''}^{SD}$ ($a=%.2f, n=10^{%.0f}$)', a, log10(block_size(i))));
    end
    
    set(gca, 'YScale', 'log'); 
    %set(gca, 'XScale', 'log'); 
    ylim([1e-10, 2]);
    
    % IMPOSTAZIONE LEGENDA 
    legend('show', 'Location', 'NorthEast', 'Interpreter', 'latex', 'FontSize', 14); 
end

function driver_finite_size_rate_BPSK_eta(etas, block_size, Rate_vN, Rate_Sand, alpha, xi, a)
%DRIVER_FINITE_SIZE_RATE_BPSK Plot Key Rate vs Distance for different n.
%   Rate_vN: Matrix (nD x 1 x 1 x nN) -> squeeze to (nD x nN)
%   Rate_Sand: Matrix (nD x 1 x 1 x nN x 1) -> squeeze to (nD x nN)
    R_vN_sq   = squeeze(Rate_vN);
    R_Sand_sq = squeeze(Rate_Sand);
    
    nN = length(block_size);
    colors = lines(nN);
    
    figure('Name', sprintf('BPSK Finite Size (alpha=%.2f, xi=%.3f)', alpha, xi)); 
    hold on; box on; grid on;
    
    % IMPOSTAZIONE FONT DEI TICK (Times New Roman)
    set(gca, 'FontName', 'Times New Roman','FontSize', 18);
    
    xlabel('$\eta$', 'Interpreter', 'latex','FontSize', 18); 
    
    % Plot von Neumann (AEP) curves
    for i = 1:nN
        plot(etas, R_vN_sq(:, i), '--*', 'Color', colors(i,:), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('$r_{\\epsilon'''',\\omega}^{AEP}$ ($n=10^{%.0f}$)', log10(block_size(i))));
    end
    
    % Plot Sandwiched Renyi curves
    for i = 1:nN
        plot(etas, R_Sand_sq(:, i), '-o', 'Color', colors(i,:), 'LineWidth', 1.2, ...
            'DisplayName', sprintf('$r_{\\epsilon'''',\\omega}^{SD}$ ($n=10^{%.0f}$)', log10(block_size(i))));
    end
    
    set(gca, 'YScale', 'log'); 
    %set(gca, 'XScale', 'log'); 
    ylim([5*1e-4, 1]);

    % --- 3. LEGENDA ESTERNA E SPAZIATA ---
    lgd = legend('show');
    set(lgd, 'Location', 'EastOutside', ...
             'Interpreter', 'latex', ...
             'FontSize', 20); 

end

function driver_finite_size_rate_QPSK_eta(etas, block_size, Rate_vN, Rate_Sand, alpha, xi, a)
%DRIVER_FINITE_SIZE_RATE_QPSK Plot Key Rate vs Distance for different n.
    R_vN_sq   = squeeze(Rate_vN);
    R_Sand_sq = squeeze(Rate_Sand);
    
    nN = length(block_size);
    colors = lines(nN);
    
    figure('Name', sprintf('QPSK Finite Size (alpha=%.2f, xi=%.3f)', alpha, xi)); 
    hold on; box on; grid on;
    
    % IMPOSTAZIONE FONT DEI TICK (Times New Roman)
    set(gca, 'FontName', 'Times New Roman','FontSize', 18);
    
    xlabel('$\eta$', 'Interpreter', 'latex','FontSize', 18);
    % Etichetta asse Y rimossa come da richiesta precedente
    
    % Plot von Neumann (AEP) curves
    for i = 1:nN
        plot(etas, R_vN_sq(:, i), '--*', 'Color', colors(i,:), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('$r_{\\epsilon'''',\\omega}^{AEP}$ ($n=10^{%.0f}$)', log10(block_size(i))));
    end
    
    % Plot Sandwiched Renyi curves
    for i = 1:nN
        plot(etas, R_Sand_sq(:, i), '-o', 'Color', colors(i,:), 'LineWidth', 1.2, ...
            'DisplayName', sprintf('$r_{\\epsilon'''',\\omega}^{SD}$ ($n=10^{%.0f}$)',  log10(block_size(i))));
    end
    
    set(gca, 'YScale', 'log'); 
    %set(gca, 'XScale', 'log'); 
    ylim([5*1e-4, 1]);
    % --- 3. LEGENDA ESTERNA E SPAZIATA ---
    lgd = legend('show');
    set(lgd, 'Location', 'EastOutside', ...
             'Interpreter', 'latex', ...
             'FontSize', 20); 

end

function driver_finite_size_rate_QPSK_with_ref(distances, block_size, Rate_vN, Rate_Sand, alpha, xi, a)
%DRIVER_FINITE_SIZE_RATE_QPSK_WITH_REF_STRICT 
% Plot Key Rate vs Distance for different n.

    R_vN_sq   = squeeze(Rate_vN);
    R_Sand_sq = squeeze(Rate_Sand);
    
    nN = length(block_size);
    colors = lines(nN); % Colori per le tue simulazioni
    
    figure('Name', sprintf('QPSK Finite Size w/ Ref Strict (alpha=%.2f, xi=%.3f)', alpha, xi)); 
    hold on; box on; grid on;
    
    % IMPOSTAZIONE FONT (Times New Roman)
    set(gca, 'FontName', 'Times New Roman','FontSize', 18);
    
    % Etichette Assi
    xlabel('$d$ [km]', 'Interpreter', 'latex','FontSize', 18);
    %ylabel('Secure Key Rate (Bits per Channel use)', 'Interpreter', 'latex', 'FontSize', 14);
    
    % --- 1. DATASET DI RIFERIMENTO  ---
    
    % Green Curve (N = 1E9) - 
    ref_x_1e9 = [1, 5, 10, 15, 17, 18, 19, 20, 21, 22];
    ref_y_1e9 = [0.25, 0.12, 0.063, 0.025, 0.015, 0.01, 0.006, 0.004, 0.002, 3e-4];
    
    % Purple/Magenta Curve (N = 1E10) - 
    ref_x_1e10 = [1, 10, 15, 20, 25, 30, 33, 35, 37, 38, 38.8];
    ref_y_1e10 = [0.32, 0.11, 0.06, 0.035, 0.02, 0.011, 0.006, 0.004, 0.002, 0.001, 5e-4];
    
    % Blue Curve (N = 1E11) - 
    ref_x_1e11 = [1, 10, 20, 30, 35, 40, 45, 50, 52, 54, 55, 56];
    ref_y_1e11 = [0.38, 0.15, 0.07, 0.025, 0.015, 0.009, 0.005, 0.0025, 0.0012, 0.0006, 0.0003, 2e-6];
    
    % Red Curve (N = 1E12) - 
    ref_x_1e12 = [1, 10, 20, 30, 40, 50, 60, 65, 68, 69, 70];
    ref_y_1e12 = [0.4, 0.19, 0.08, 0.038, 0.018, 0.009, 0.0035, 0.0015, 0.0006, 0.0002, 3e-5];
    
    % Black Curve - Asymptotic 
    ref_x_asy = [0, 80];
    ref_y_asy = [0.45, 0.0022]; % Approssimazione retta logaritmica

    % --- 2. PLOT REFERENCE (Stile Immagine) ---
    % Uso colori fissi (g, m, b, r) per matchare lo screenshot
    
    plot(ref_x_1e9, ref_y_1e9, ':v', 'Color', [0, 0.8, 0], 'LineWidth', 1.2, 'MarkerSize', 6, ...
        'DisplayName', 'Ref. [13] ($n=10^9$)');
    
    %plot(ref_x_1e10, ref_y_1e10, ':o', 'Color', 'm', 'LineWidth', 1.2, 'MarkerSize', 6, ...
     %   'DisplayName', 'Ref $N=10^{10}$');
    
    %plot(ref_x_1e11, ref_y_1e11, ':+', 'Color', 'b', 'LineWidth', 1.2, 'MarkerSize', 6, ...
       % 'DisplayName', 'Ref $N=10^{11}$');
    
    %plot(ref_x_1e12, ref_y_1e12, ':x', 'Color', 'r', 'LineWidth', 1.2, 'MarkerSize', 6, ...
     %   'DisplayName', 'Ref $N=10^{12}$');
    
    %plot(ref_x_asy, ref_y_asy, 'k-', 'LineWidth', 1.5, ...
     %   'DisplayName', 'Ref Asymptotic');

    % --- 3. PLOT SIMULAZIONE (Le tue curve calcolate) ---
    
    % Plot von Neumann (AEP) curves
    % for i = 1:1
    %     plot(distances, R_vN_sq(:, i), '--', 'Color', colors(i,:), 'LineWidth', 1.5, ...
    %         'DisplayName', sprintf('$r_{\\epsilon''}^{AEP}$ ($n=10^{%.0f}$)', log10(block_size(i))));
    % end
    
    % Plot Sandwiched Renyi curves
    for i = 1:nN
        plot(distances, R_Sand_sq(:, i), '-o', 'Color', colors(i,:), 'LineWidth', 1.2, ...
            'DisplayName', sprintf('$r_{\\epsilon'''',\\omega}^{SD}$ ($n=10^{%.0f}$)',  log10(block_size(i))));
    end
    
    % --- 4. FORMATTAZIONE ASSI ---
    set(gca, 'YScale', 'log'); 
    % set(gca, 'XScale', 'log'); 
    ylim([1e-6, 1]); 
    xlim([0, 250]);
    
    % IMPOSTAZIONE LEGENDA
    legend('show', 'Location', 'SouthWest', 'Interpreter', 'latex', 'FontSize', 20, 'NumColumns', 2); 
end

function driver_finite_size_vs_blocksize_BPSK(block_size, Rate_vN, Rate_Sand, distance, alpha, xi, a)
%DRIVER_FINITE_SIZE_VS_BLOCKSIZE_BPSK Plot BPSK Key Rate vs Block Size.

    R_vN_vec = squeeze(Rate_vN);
    R_Sand_vec = squeeze(Rate_Sand);
    
    % Definizione del verde scuro (RGB: R=0, G=0.6, B=0)
    dark_green = [0, 0.6, 0]; 
    
    figure('Name', sprintf('BPSK Key Rate vs Block Size (d=%.1f km)', distance));
    hold on; box on; grid on;
    
    % --- 1. PLOT CURVES ---
    loglog(block_size, R_vN_vec, ...
        'Color', dark_green, ...           % Linea verde scuro
        'LineWidth', 1.5, ...
        'Marker', 's', ...
        'MarkerSize', 8, ...
        'MarkerFaceColor', dark_green, ... % Faccia verde scuro
        'LineStyle', 'none', ...
        'DisplayName', '$r_{\epsilon'''',\omega}^{AEP}$'); 
        
    loglog(block_size, R_Sand_vec, 'r', 'LineWidth', 1.5, 'Marker', '^', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'LineStyle', 'none', ...
        'DisplayName', '$r_{\epsilon'''',\omega}^{SD}$'); 
    
    % --- 2. IMPOSTAZIONI ASSE ---
    set(gca, 'XScale', 'log');
    ylim([2*10^-2, 0.16]);
    xlim([10^4, 2*10^10]);
    
    set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 18); 
    set(gca, 'FontName', 'Times New Roman'); 
    
    xlabel('$n$', 'Interpreter', 'latex', 'FontSize', 18); 
    
    % --- 3. LEGENDA ESTERNA E SPAZIATA ---
    lgd = legend('show');
    set(lgd, 'Location', 'EastOutside', ...
             'Interpreter', 'latex', ...
             'FontSize', 20); 
end

function driver_finite_size_vs_blocksize_QPSK(block_size, Rate_vN, Rate_Sand, distance, alpha, xi, a)
%DRIVER_FINITE_SIZE_VS_BLOCKSIZE_QPSK Plot QPSK Key Rate vs Block Size.

    R_vN_vec = squeeze(Rate_vN);
    R_Sand_vec = squeeze(Rate_Sand);

    % Definizione del verde scuro
    dark_green = [0, 0.6, 0]; 
    
    figure('Name', sprintf('QPSK Key Rate vs Block Size (d=%.1f km)', distance));
    hold on; box on; grid on;
    
    % --- 1. PLOT CURVES ---
    loglog(block_size, R_vN_vec, ...
        'Color', dark_green, ...           % Linea verde scuro
        'LineWidth', 1.5, ...
        'Marker', 's', ...
        'MarkerSize', 8, ...
        'MarkerFaceColor', dark_green, ... % Faccia verde scuro
        'LineStyle', 'none', ...
        'DisplayName', '$r_{\epsilon'''',\omega}^{AEP}$'); 
        
    loglog(block_size, R_Sand_vec, 'r', 'LineWidth', 1.5, 'Marker', '^', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'LineStyle', 'none', ...
        'DisplayName', '$r_{\epsilon'''',\omega}^{SD}$'); 
    
    % --- 2. IMPOSTAZIONI ASSE ---
    set(gca, 'XScale', 'log'); 
    ylim([6*10^-2, 0.2]);
    xlim([10^4, 2*10^10]);
    
    set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 18); 
    set(gca, 'FontName', 'Times New Roman'); 
    
    xlabel('$n$', 'Interpreter', 'latex', 'FontSize', 18); 
    
    % --- 3. LEGENDA ESTERNA E SPAZIATA ---
    lgd = legend('show');
    set(lgd, 'Location', 'EastOutside', ...
             'Interpreter', 'latex', ...
             'FontSize', 20);
end

%% ========================================================================
%  HELPER FUNCTIONS (States, Math, Entropy)
% =========================================================================
% build_rho_YE_BPSK, build_rho_YE_QPSK, validate_state, build_rho_E_symbol, 
% bs_coeff, conditional_vN_entropy, conditional_SandDown_entropy, 
% Analytic_SandDown, compute_rho_E_from_rhoYE, vonNeumann_entropy, 
% leakage_BPSK, leakage_QPSK, BinEntropy) ...

function [rhoYE, rhoE_y, rhoE_x, p_y_given_x] = build_rho_YE_BPSK(dimE, eta, mu, alpha)
    if dimE < 1, error('dimE must be >= 1.'); end
    rhoE_x = cell(1,2);
    alpha0 = alpha; alpha1 = -alpha;
    rhoE_x{1} = build_rho_E_symbol(alpha0, dimE, eta, mu);
    rhoE_x{2} = build_rho_E_symbol(alpha1, dimE, eta, mu);
    a = abs(alpha);
    VB = 0.5 + (1-eta)*mu;
    arg = sqrt(2*eta)*a / sqrt(2*VB);
    p_correct = 0.5 * (1 + erf(arg));
    p_error   = 1 - p_correct;
    p_y_given_x = zeros(2,2);
    p_y_given_x(1,1) = p_correct; p_y_given_x(2,1) = p_error;
    p_y_given_x(1,2) = p_error;   p_y_given_x(2,2) = p_correct;
    dimE2 = dimE^2;
    rhoE_y = cell(1,2);
    for y = 0:1
        rho_tmp = zeros(dimE2, dimE2);
        for x = 0:1
            rho_tmp = rho_tmp + p_y_given_x(y+1,x+1) * rhoE_x{x+1};
        end
        rhoE_y{y+1} = rho_tmp;
    end
    p_y = 0.5 * ones(2,1);
    dimY = 2;
    rhoYE = zeros(dimY*dimE2, dimY*dimE2);
    for y = 0:1
        rows = (y*dimE2+1):((y+1)*dimE2);
        cols = rows;
        rhoYE(rows, cols) = p_y(y+1) * rhoE_y{y+1};
    end
    validate_state(rhoYE, sprintf('rhoYE_BPSK (eta=%.4f)', eta));
    rhoYE = (rhoYE + rhoYE')/2;
    rhoYE = rhoYE / trace(rhoYE);
end

function [rhoYE, rhoE_y, rhoE_k, p_y_given_k] = build_rho_YE_QPSK(dimE, eta, mu, alpha)
    if dimE < 1, error('dimE must be >= 1.'); end
    rhoE_k = cell(1,4);
    for k = 0:3
        alpha_k = (1i^k) * exp(1i*pi/4) * alpha;
        rhoE_k{k+1} = build_rho_E_symbol(alpha_k, dimE, eta, mu);
    end
    a = abs(alpha);
    Vhet = (1 + (1-eta)*mu)/2;
    m    = sqrt(eta) * a / sqrt(2);
    arg  = m / sqrt(2*Vhet);
    P_plus  = 0.5 * (1 + erf(arg));
    P_minus = 1 - P_plus;
    p_y_given_k = zeros(4,4);
    for k = 0:3
        for y = 0:3
            delta = mod(y - k, 4);
            if delta == 0
                p_y_given_k(y+1,k+1) = P_plus^2;
            elseif delta == 2
                p_y_given_k(y+1,k+1) = P_minus^2;
            else
                p_y_given_k(y+1,k+1) = P_plus * P_minus;
            end
        end
    end
    dimE2 = dimE^2;
    rhoE_y = cell(1,4);
    for y = 0:3
        rho_tmp = zeros(dimE2, dimE2);
        for k = 0:3
            rho_tmp = rho_tmp + p_y_given_k(y+1,k+1) * rhoE_k{k+1};
        end
        rhoE_y{y+1} = rho_tmp;
    end
    dimY = 4;
    rhoYE = zeros(dimY*dimE2, dimY*dimE2);
    p_y = 0.25 * ones(4,1);
    for y = 0:3
        rows = (y*dimE2+1):((y+1)*dimE2);
        cols = rows;
        rhoYE(rows, cols) = p_y(y+1) * rhoE_y{y+1};
    end
    validate_state(rhoYE, sprintf('rhoYE_QPSK (eta=%.4f)', eta));
    rhoYE = (rhoYE + rhoYE')/2;
    rhoYE = rhoYE / trace(rhoYE);
end

function validate_state(rho, tag)
    tr = trace(rho);
    if abs(tr - 1) > 1e-8
        warning('[%s] Normalization violation: Trace = %.6f (Diff: %.2e)', ...
            tag, tr, abs(tr - 1));
    end
    rho_sym = (rho + rho') / 2;
    ev = eig(rho_sym); 
    min_ev = min(ev);
    if min_ev < -1e-10
        warning('[%s] Positivity violation: Min Eigenvalue = %.6e', tag, min_ev);
    end
end

function rhoE = build_rho_E_symbol(alpha_x, dimE, eta, mu)
    Nmax  = dimE - 1; dimE2 = dimE^2;
    n = (0:Nmax).';
    c_n = sqrt( (mu.^n) ./ ((1+mu).^(n+1)) );
    coh_pref = exp(-0.5 * abs(alpha_x)^2);
    rhoE = zeros(dimE2, dimE2);
    for j = 0:Nmax
        psi_j = zeros(dimE2,1);
        for k = 0:Nmax
            for n0 = 0:Nmax
                amp_in = coh_pref * (alpha_x^k / sqrt(factorial(k))) * c_n(n0+1);
                r = k + n0 - j;
                if r < 0 || r > Nmax, continue; end
                U = bs_coeff(j,k,n0,eta);
                if U == 0, continue; end
                s = n0;
                idx_E = r*dimE + s + 1;
                psi_j(idx_E) = psi_j(idx_E) + amp_in * U;
            end
        end
        rhoE = rhoE + psi_j * psi_j';
    end
    rhoE = (rhoE + rhoE')/2;
    trE  = trace(rhoE);
    if trE > 0, rhoE = rhoE / trE; end
end

function U = bs_coeff(j,k,n,eta)
    if j < 0 || k < 0 || n < 0, U = 0; return; end
    if j > (k + n), U = 0; return; end
    normFactor = sqrt( factorial(k) * factorial(n) / ( factorial(j) * factorial(k+n-j) ) );
    sumTerm = 0;
    r_min = max(0, j - n);
    r_max = min(j, k);
    for r = r_min:r_max
        comb1 = nchoosek(j, r);
        comb2 = nchoosek(k + n - j, k - r);
        phase = (-1)^(n - j + r);
        pow_eta = (n - j + 2*r) / 2;
        pow_omp = (k + j - 2*r) / 2;
        term = phase * comb1 * comb2 * (eta^pow_eta) * ((1-eta)^pow_omp);
        sumTerm = sumTerm + term;
    end
    U = normFactor * sumTerm;
    if abs(U) < 1e-15, U = 0; end
end

function H = conditional_vN_entropy(rhoYE, rhoE)
    S_YE = vonNeumann_entropy(rhoYE);
    S_E  = vonNeumann_entropy(rhoE);
    H = S_YE - S_E;
end

function H_tilde = conditional_SandDown_entropy(rhoYE, rhoE, a)
    if a <= 0 || abs(a-1) < 1e-12, error('Rényi order a must satisfy a > 0 and a ~= 1.'); end
    [dYE1, ~] = size(rhoYE); dimTot = dYE1;
    [dE1, ~] = size(rhoE); dimE = dE1;
    dimY = dimTot / dimE;
    rhoYE = (rhoYE + rhoYE')/2; rhoYE = rhoYE / trace(rhoYE);
    rhoE = (rhoE + rhoE')/2; rhoE = rhoE / trace(rhoE);
    beta = (1 - a) / (2*a);
    [U_E, S_E, ~] = svd(rhoE); 
    lambdaE = diag(S_E);
    tol = 1e-13; lambdaE(lambdaE < tol) = 0;
    supp = lambdaE > 0;
    lambdaE_beta = zeros(size(lambdaE));
    lambdaE_beta(supp) = lambdaE(supp).^beta;
    rhoE_beta = U_E * diag(lambdaE_beta) * U_E';
    rhoE_beta = (rhoE_beta + rhoE_beta')/2;
    IY = eye(dimY);
    R  = kron(IY, rhoE_beta);
    if any(isnan(R(:))) || any(isinf(R(:))), H_tilde = NaN; return; end
    A = R * rhoYE * R;
    A = (A + A')/2;
    try lambdaA = eig(A); catch, lambdaA = svd(A); end
    lambdaA = real(lambdaA); lambdaA(lambdaA < 1e-15) = 0;
    nz = lambdaA > 0;
    if ~any(nz)
        H_tilde = NaN; 
    else
        trace_Aa = sum( lambdaA(nz).^a );
        H_tilde = (1/(1 - a)) * log2(trace_Aa);
    end
end

function H = Analytic_SandDown(eta, alpha, a)
    kappa = exp(2 .* alpha.^2 .* (eta - 1));
    r = erf(sqrt(2 .* eta) .* alpha);
    phi = atanh(kappa);
    phi_a = phi ./ a;
    Delta = sqrt(r.^2 + sinh(phi_a).^2);
    cosh_phi_a = cosh(phi_a);
    term_inner = (cosh_phi_a - Delta).^a + (cosh_phi_a + Delta).^a;
    log_arg = (2.^a) .* sech(phi) .* term_inner;
    term1 = -(2 .* a) ./ (1 - a);
    term2 = (1 ./ (1 - a)) .* log2(log_arg); 
    H = term1 + term2;
end

function rhoE = compute_rho_E_from_rhoYE(rhoYE, dimY)
    dimTot = size(rhoYE, 1);
    dimE = dimTot / dimY;
    rhoE = zeros(dimE, dimE);
    for y = 0:dimY-1
        idx = (y*dimE + 1):((y+1)*dimE);
        rhoE = rhoE + rhoYE(idx, idx);
    end
    rhoE = (rhoE + rhoE')/2;
    rhoE = rhoE / trace(rhoE);
end

function S = vonNeumann_entropy(rho)
    if any(isnan(rho(:))) || any(isinf(rho(:))), S = NaN; return; end
    rho = (rho + rho')/2;
    trR = real(trace(rho));
    if trR <= 1e-15, S = 0; return; end
    rho = rho / trR;
    try lambda = svd(rho); catch, lambda = real(eig(rho)); end
    lambda(lambda < 1e-15) = 0;
    sum_L = sum(lambda);
    if sum_L > 0, lambda = lambda / sum_L; end
    nz = lambda > 0;
    if ~any(nz), S = 0; else, S = -sum(lambda(nz) .* log2(lambda(nz))); end
end

function h = leakage_BPSK(p_y_given_x)
    pp  = p_y_given_x(1,1);
    pm = p_y_given_x(1,2);
    fact1 = -pp*log2(pp);
    fact2 = -pm*log2(pm);
    h= fact1 + fact2;
end

function h = leakage_QPSK(p_y_given_k)
    pp  = sqrt(p_y_given_k(1,1));
    pm  = sqrt(p_y_given_k(1,3));
    fact1 = -pp*log2(pp);
    fact2 = -pm*log2(pm);
    h= 2*(fact1 + fact2);
end

function h = BinEntropy(p)
    if (p==0 || p==1), h=0; else, h = -p*log2(p)-(1-p)*log2(1-p); end
end