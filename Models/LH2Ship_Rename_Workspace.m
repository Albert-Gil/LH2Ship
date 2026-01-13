%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT: LH2Ship_Rename_Workspace.m
% PROJECT: Feasibility of Zero-Emission Cruise Ships
% DESCRIPTION: 
%   Utility script to organize the workspace after a Simulink run.
%   1. Gathers all loose variables (e.g., S_Time, S_Power).
%   2. Strips the specified prefix (e.g., "S_").
%   3. Packs them into a single struct (e.g., "Winter").
%   4. Clears the original loose variables to clean the workspace.
%
% USAGE:
%   Run this immediately after the Simulink simulation finishes, 
%   before running the plotting scripts.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1. CONFIGURATION
% Set the name of the output struct ('Winter' or 'Summer')
TargetStructName = 'Winter'; 

% Set the prefix to strip from variable names (usually 'S_')
PrefixToRemove   = 'S_'; 

%% 2. EXECUTION
fprintf('Packing workspace variables into struct "%s"...\n', TargetStructName);

% Get all variable names in the current workspace
vars = who; 

% Exclude the configuration variables themselves from being packed
varsToExclude = {'vars', 'TargetStructName', 'PrefixToRemove', 'i', 'varName', 'value', 'fieldName', 'varsToExclude'};
varsToPack = setdiff(vars, varsToExclude);

% Initialize the new structure
OutputStruct = struct; 

% Iterate through variables
for i = 1:length(varsToPack)
    varName = varsToPack{i};
    
    % Get the value from the base workspace
    value = evalin('base', varName); 
    
    % Remove prefix if present
    if startsWith(varName, PrefixToRemove)
        % 'S_Time' -> 'Time'
        fieldName = extractAfter(varName, length(PrefixToRemove)); 
    else
        % 'OtherVar' -> 'OtherVar'
        fieldName = varName; 
    end
    
    % Assign to the struct
    OutputStruct.(fieldName) = value;
end

%% 3. FINALIZE
% Assign the new struct to the Base Workspace
assignin('base', TargetStructName, OutputStruct);

% Clear the old loose variables
% We construct a string to clear only the packed variables, keeping the new struct
clearCommand = ['clear ', strjoin(varsToPack)];
evalin('base', clearCommand);

% Clean up temporary script variables
clear vars varsToExclude varsToPack i varName value fieldName PrefixToRemove clearCommand;

fprintf('Done. Access results via the "%s" struct.\n', TargetStructName);