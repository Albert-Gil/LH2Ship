%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT: LH2Ship_Compare_Scenarios.m
% DESCRIPTION: 
%   Comparative analysis of Summer (Critical Cooling) vs Winter (Heating)
%   scenarios. Generates 8 comparative figures for the thesis.
%
% INPUTS: 
%   - LH2Ship_Simulink_Summer_Results_080825_Clean.mat
%   - LH2Ship_Simulink_Winter_Results.mat
%
% OUTPUTS:
%   - Figures saved to "results/plots/" (optional)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

%% 1. INITIALIZATION & SETTINGS
% Graphics Standards
set(groot, 'defaultLineLineWidth', 2)
set(0, 'DefaultAxesFontSize', 14);
set(0, 'DefaultTextFontSize', 14);

% User Options
SavePlots = 1;                  % 1 to save figures
TimeUnits = 'hours';            % 'minutes' or 'hours'
PlotFolder = '../../results/plots/'; % Output folder (Adjust relative path as needed)

if SavePlots && ~exist(PlotFolder, 'dir')
    mkdir(PlotFolder);
end

% Color Palette
colors.yellow = [0.94 0.80 0.1250];
colors.orange = [0.89 0.5 0.1];
colors.purple = [0.4940 0.1840 0.5560];
colors.red    = [0.86 0.05 0.05];
colors.green  = [0.14 0.53 0.05];
colors.blue   = [0.01 0.45 0.75];

%% 2. DATA LOADING & MAPPING
% This section maps raw variables (e.g., S_Time) into structured structs (Summer.Time)

fprintf('Loading Simulation Data...\n');

% Define File Paths (Relative to src/scripts/)
summerFile = '../../data/results/LH2Ship_Simulink_Summer_Results_080825_Clean.mat';
winterFile = '../../data/results/LH2Ship_Simulink_Winter_Results.mat';

% Load & Map Summer Data
if exist(summerFile, 'file')
    rawS = load(summerFile);
    Summer = MapResultsToStruct(rawS, 'S_'); % Helper function at bottom
    fprintf('  [OK] Summer data loaded.\n');
else
    error('Summer results file not found: %s', summerFile);
end

% Load & Map Winter Data
if exist(winterFile, 'file')
    rawW = load(winterFile);
    % Note: Assuming Winter vars might start with 'W_' or 'S_' (if reused). 
    % We try to detect the prefix or map loosely.
    Winter = MapResultsToStruct(rawW, 'S_'); 
    % If your Winter file uses 'W_', change 'S_' to 'W_' above.
    fprintf('  [OK] Winter data loaded.\n');
else
    error('Winter results file not found: %s', winterFile);
end

%% 3. TIME VECTOR PREPARATION
[tS, xlimS] = toPlotTime(Summer.Time, TimeUnits);
[tW, xlimW] = toPlotTime(Winter.Time, TimeUnits);

%% 4. PLOTTING FIGURES

% --- FIGURE 1: POWER DEMANDS COMPARISON ---
figure('Name', 'Power Demands'); 
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.1 0.1 0.6 0.8]);

subplot(2,1,1);
    title('Power Demands — Summer (Cooling Dominant)'); hold on; grid on;
    plotxy(tS, Summer.PropPower_W,     'Color', colors.blue,   'DisplayName','Propulsion');
    plotxy(tS, Summer.AuxElPower_W,    'Color', colors.orange, 'DisplayName','Aux Electric');
    plotxy(tS, Summer.CoolPowerDemand_W, 'Color', colors.purple, 'DisplayName','Cooling Demand');
    plotxy(tS, Summer.TotalPowerDemand_W,'Color', colors.green,'DisplayName','Total Power');
    ylabel('Power [MW]'); xlim([0 xlimS]); legend('Location','best');
    
subplot(2,1,2);
    title('Power Demands — Winter (Heating Dominant)'); hold on; grid on;
    plotxy(tW, Winter.PropPower_W,     'Color', colors.blue,   'DisplayName','Propulsion');
    plotxy(tW, Winter.AuxElPower_W,    'Color', colors.orange, 'DisplayName','Aux Electric');
    plotxy(tW, Winter.HeatPowerDemand_W, 'Color', colors.red,    'DisplayName','Heating Demand');
    plotxy(tW, Winter.TotalPowerDemand_W,'Color', colors.green,'DisplayName','Total Power');
    xlabel(['Time [' TimeUnits ']']); ylabel('Power [MW]'); xlim([0 xlimW]); legend('Location','best');

if SavePlots, savePlot(gcf, PlotFolder, "1_PowerDemands_Compare"); end


% --- FIGURE 2: GENERATED POWER BREAKDOWN ---
figure('Name', 'Generated Power'); 
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.1 0.1 0.6 0.8]);

subplot(2,1,1);
    title('Generation — Summer'); hold on; grid on;
    plotxy(tS, Summer.TotalPower_W,     'Color', colors.blue,   'DisplayName','Total Gen');
    plotxy(tS, Summer.BoilerHeat_W,     'Color', colors.red,    'DisplayName','Burner Heat');
    plotxy(tS, Summer.FCGenHeat_W,      'Color', colors.orange, 'DisplayName','FC Heat Recov');
    ylabel('Power [MW]'); xlim([0 xlimS]); legend('Location','best');

subplot(2,1,2);
    title('Generation — Winter'); hold on; grid on;
    plotxy(tW, Winter.TotalPower_W,     'Color', colors.blue,   'DisplayName','Total Gen');
    plotxy(tW, Winter.BoilerHeat_W,     'Color', colors.red,    'DisplayName','Burner Heat');
    plotxy(tW, Winter.FCGenHeat_W,      'Color', colors.orange, 'DisplayName','FC Heat Recov');
    xlabel(['Time [' TimeUnits ']']); ylabel('Power [MW]'); xlim([0 xlimW]); legend('Location','best');

if SavePlots, savePlot(gcf, PlotFolder, "2_PowerGeneration_Compare"); end


% --- FIGURE 3: EFFICIENCY & H2 UTILIZATION ---
figure('Name', 'Efficiency'); 
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.1 0.1 0.6 0.8]);

subplot(2,1,1);
    title('System Efficiency — Summer'); hold on; grid on;
    yyaxis left
    plotxy(tS, Summer.H2utilization, 'Color', colors.green, 'DisplayName','H_2 Utilization');
    ylabel('Utilization [-]');
    yyaxis right
    plotxy(tS, Summer.TotalInstEff, 'Color', colors.blue, 'DisplayName','Inst. Efficiency');
    ylabel('Efficiency [-]');
    xlim([0 xlimS]); legend('Location','southwest');

subplot(2,1,2);
    title('System Efficiency — Winter'); hold on; grid on;
    yyaxis left
    plotxy(tW, Winter.H2utilization, 'Color', colors.green, 'DisplayName','H_2 Utilization');
    ylabel('Utilization [-]');
    yyaxis right
    plotxy(tW, Winter.TotalInstEff, 'Color', colors.blue, 'DisplayName','Inst. Efficiency');
    ylabel('Efficiency [-]');
    xlabel(['Time [' TimeUnits ']']); xlim([0 xlimW]);

if SavePlots, savePlot(gcf, PlotFolder, "3_Efficiency_Compare"); end


% --- FIGURE 4: BATTERY DYNAMICS (SUMMER) ---
figure('Name', 'Battery Dynamics'); 
tiledlayout(3,1);

nexttile; title('Summer: DC Bus Voltage'); hold on; grid on;
    plotxy(tS, Summer.BusVoltage, 'Color', colors.purple);
    ylabel('Voltage [V]'); xlim([0 xlimS]);

nexttile; title('Summer: Battery Power Flow'); hold on; grid on;
    plotxy(tS, Summer.PowerToBatt_W, 'Color', colors.red);
    ylabel('Power [MW]'); xlim([0 xlimS]); legend('(+) Charging / (-) Discharging');

nexttile; title('Summer: Battery SOC'); hold on; grid on;
    plotxy(tS, Summer.BatterySOC, 'Color', colors.green);
    ylabel('SOC [-]'); xlabel(['Time [' TimeUnits ']']); xlim([0 xlimS]); ylim([0 1]);

if SavePlots, savePlot(gcf, PlotFolder, "4_Battery_Summer"); end


% --- FIGURE 5: CUMULATIVE H2 CONSUMPTION ---
figure('Name', 'H2 Consumption'); 
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.15 0.15 0.6 0.6]);

title('Cumulative Hydrogen Consumption'); hold on; grid on;
[h2totS, ~, ~] = getCumH2(Summer, tS, TimeUnits);
[h2totW, ~, ~] = getCumH2(Winter, tW, TimeUnits);

plotxy(tS, h2totS, 'Color', colors.red,   'DisplayName','Summer Total');
plotxy(tW, h2totW, 'Color', colors.blue,  'DisplayName','Winter Total');

xlabel(['Time [' TimeUnits ']']); ylabel('H_2 Consumed [tonnes]'); 
legend('Location','best');
xlim([0 max(xlimS, xlimW)]);

if SavePlots, savePlot(gcf, PlotFolder, "5_Cumulative_H2"); end

fprintf('Plots generated successfully.\n');

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function S = MapResultsToStruct(rawData, prefix)
    % Maps variables from raw struct (e.g. S_Time) to Clean Struct (S.Time)
    % Handles exact matches or prefix removal.
    
    fields = fieldnames(rawData);
    S = struct();
    
    for i = 1:length(fields)
        val = rawData.(fields{i});
        name = fields{i};
        
        % Remove prefix if present (e.g. "S_Time" -> "Time")
        if startsWith(name, prefix)
            cleanName = extractAfter(name, length(prefix));
        else
            cleanName = name;
        end
        
        % Map common variations to standard names for the plotter
        switch lower(cleanName)
            case {'coolpowerdemand_w', 'coolpower_w'}
                S.CoolPowerDemand_W = val;
            case {'heatpowerdemand_w', 'heatpower_w', 'spaceheatpower_w'}
                S.HeatPowerDemand_W = val;
            case {'totalpowerdemand_w', 'totalpower_w'}
                S.TotalPowerDemand_W = val; % Demand side
                S.TotalPower_W = val;       % Gen side (often same in balance)
            case {'totalgenpower_w'}
                 S.TotalPower_W = val;
            case {'h2_consumption_rate', 'h2consrate_kgs'}
                S.H2instCons_kgs = val;
            % Add other mappings as needed based on your specific .mat files
            otherwise
                S.(cleanName) = val;
        end
        
        % Keep original just in case
        S.(cleanName) = val;
    end
end

function h = plotxy(x, y, varargin)
    % Robust plotting helper that handles Unit Conversion (W -> MW)
    % and reshaping
    if isempty(y) || isempty(x), return; end
    
    x = double(x(:));
    y = double(y(:));
    
    % Auto-convert Watts to MW if mean value > 1e5
    if mean(abs(y), 'omitnan') > 1e5
        y = y / 1e6;
    end
    
    % Auto-convert kg to Tonnes if max > 1000 (for cumulative)
    if max(abs(y), [], 'all') > 5000 
        y = y / 1000;
    end
    
    n = min(length(x), length(y));
    h = plot(x(1:n), y(1:n), varargin{:});
end

function [tPlot, xlimit] = toPlotTime(tRaw, timeUnits)
    if isempty(tRaw), tPlot=[]; xlimit=0; return; end
    tRaw = double(tRaw(:));
    tRaw = tRaw - tRaw(1);
    switch lower(timeUnits)
        case 'minutes', tPlot = tRaw/60;
        otherwise,      tPlot = tRaw/3600; % hours
    end
    xlimit = max(tPlot);
end

function [h2_total, h2_port, h2_sail] = getCumH2(S, tPlot, timeUnits)
    % Attempts to calculate cumulative H2 from rate or existing cumulative vars
    if isfield(S, 'H2instCons_kgs')
        rate = double(S.H2instCons_kgs(:));
        % Integrate: mass = rate * dt
        % Assume constant step or calculate dt
        tSec = tPlot * (strcmpi(timeUnits,'hours')*3600 + strcmpi(timeUnits,'minutes')*60);
        dt = [diff(tSec); tSec(end)-tSec(end-1)];
        h2_total = cumsum(rate .* dt);
    else
        h2_total = [];
    end
    h2_port = []; h2_sail = []; % Placeholder
end

function savePlot(figHandle, folder, name)
    saveas(figHandle, fullfile(folder, strcat(name, ".png")));
    % saveas(figHandle, fullfile(folder, strcat(name, ".fig"))); % Uncomment if needed
end