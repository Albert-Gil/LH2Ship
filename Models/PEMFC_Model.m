%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% 1D PEMFC STATIC MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT: PEMFC_Model.m
% PROJECT: Feasibility of Zero-Emission Cruise Ships
% DESCRIPTION: 
%   1D Static Model characterizing the electrochemical behavior of the 
%   PEM Fuel Cell stack. Generates lookup tables for Simulink.
% OUTPUT: 
%   PEMFC_Results.mat
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

%% PEM FUEL CELL DATA

% Utilisations
% H2u=(0.05:0.03:0.95);       % Hydrogen utilisation factor
H2u=(0.05:0.1:0.95);       % Hydrogen utilisation factor
O2u=0.5;                    % Oxigen utilisation factor

% Size 
CellArea=50*25;             % Cell Area [cm^2]
Ncells=120;                 % Number of cells per stack
Nstacks=330;                % Number of total stacks

% Temperatures
Tamb=298.15;                % Ambient temperature [K]
Tinlet=Tamb;                % Inlet gases temperature [K]. We assume them to be at ambient temperature.
TexhaustMax=80+273.15;      % Maximum allowed exhaust gases temperature [K]
%Toutlet=343;               % Operating temperature if fixed manually [K]

% Working pressures
pC=3;                       % Cathode pressure (atm)
pA=3;                       % Anode pressure (atm)
pSat=0.307;                 % Saturation pressure of H20 (atm)

% Concentrations
xH2a=0.85;                  % Hydrogen molar fraction (anode inlet)
%xO2d=0.19;                 % Oxigen molar fraction (cathode)
xH2Oa=0.15;                 % Water molar fraction (anode outlet)
xH2Od=0.15;                 % Water molar fraction (cathode inlet)

% Thicknesses and geometric parameters
tM=100;                      % Electrolyte thickness [um]
tA=350;                     % Anode thickness [um]
tC=350;                     % Cathode thickness [um]
CathPor=0.4;                % Cathode porosity [%]
tortuosity=2.5;             % Cathode tortuosity

% Water cooling parameters
WaterCooling='y';                   % Use 'y' to enable water cooling ('yes' will also work). Leave blank or use any other expression to disable water cooling.
CoolingConvCoeff_minVel=1000;       % Estimated (by formulae) convection coefficient for water cooling when minimum flow.[W/(m2*K)]. Estimation for flow of water in a pipe.
CoolingChannSurf=pi*(5*tA-2*tA)*10^-6*(0.5+0.1)*10*(Ncells+1);    % Active surface for water cooling [m2]
CoolingWatTempIn=Tamb+5;            % Inlet temperature of FC cooling water
Re1=100;                            % Reynolds number considering the desired operating point (velocity, flow and diameter) for the minimum flow.
Re2=6000;                           % Reynolds number considering the desired operating point (velocity, flow and diameter) for the maximum flow.

% Auxiliary systems parameters
BoilerEff=0.85;
PowerElectronicsEff=0.9;
HeatStorageEff=0.9;
HeatStorageCapacity=4.5e10;
HeatStorageIC=1;

% Waste heat exchanger parameters
HXeff_exh=0.85;                     % Exhaust gases heat exchanger efficiency
SpaceHeatingAirTemp=35+273.15;      % Outlet temperature of space heating air, heated by the exhaust gases.
Text=Tamb-10;                       % Inlet temperature of space heating air. Assumed to be at outside, exterior temperature.
Tout_exh=40+273.15;                 % Exhaust gases temperature after waste heat recovery
HXeff_wat=0.85;                     % Water cooling heat exchanger efficiency
WaterHeatingTemp=55+273.15;         % Outlet temperature of hot sanitary water, heated by the FC cooling water
Tout_wat=58+273.15;                 % Cooling water temperature after waste heat recovery

% Activation parameters
TransfCoeff=0.5;                    % Transference coefficient (alpha)
j0O2=0.0001.*(0.5./O2u);             % Exchange current for cathode [A/cm2]
j0H2=0.025.*(0.9./H2u);                % Exchange current for anode [A/cm2]
jleak=0;                            % Leakage current [A/cm2]

%% SIMULATION PARAMETERS
% Nsampling=3000;              % Number of samples of current density from 0 to limiting current density
Nsampling=1000;              % Number of samples of current density from 0 to limiting current density
TempAccuracy=0.01;           % Accuracy of outlet gases temperature iteration loop [K]
CalcTempPonderation=0.85;    % Ponderation of the calculated exhaust temperature to determine the next iteration temperature [0-0.9]. High values can lead to stability problems.

Toutlet=Tamb+30;            % Initial seed of iteration to estimate operating temperature (K)   

%% Constants
R=8.314;                % Universal gas constant [J/(mol*K)]
Rlit=0.0820578;         % Universal gas constant [L*atm/(mol*K)]
F=96485;                % Faraday's constant
H2hhv=141.7*10^6;       % Hydrogen fuel high heating value [J/kg]. = 286000 J/mol
H2lhv=119.96*10^6;      % Hydrogen fuel low heating value [J/kg].
xO2air=0.21;            % Molar fraction of oxigen in air
xN2air=0.79;            % Molar fraction of nytrogen in air
O2FormEnthSTD=0;        % Formation enthalpy of O2 (gas) at standard conditions [J/mol]
O2FormEntrSTD=205;      % Formation entropy of O2 (gas) at standard conditions [J/mol]
H2FormEnthSTD=0;        % Formation enthalpy of H2 (gas) at standard conditions [J/mol]
H2FormEntrSTD=130.68;   % Formation entropy of H2 (gas) at standard conditions [J/mol]
H2OFormEnthSTD=-285830; % Formation enthalpy of H2O (liquid) at standard conditions [J/mol]
H2OFormEntrSTD=69.95;   % Formation entropy of H2O (liquid) at standard conditions [J/mol]
CpO2T=(28.91+30.30)/2;  % O2 (gas) molar heat capacity. We assume it as constant, using cp's mean value from 298.15K to 380K. [J/mol*K]
CpH2T=(28.84+29.15)/2;  % H2 (gas) molar heat capacity. We assume it as constant, using cp's mean value from 298.15K to 380K. [J/mol*K]
CpH2OT=(75.37+75.99)/2; % H2O (liquid) molar heat capacity. We assume it as constant, using cp's mean value from 298.15K to 373.15K. [J/mol*K]
CpH2OTkg=(4180+4220)/2; % H2O (liquid) heat capacity [J/(kg*K)]. We assume it as constant, using cp's mean value from 298.15K to 373.15K. 
CpH2OliqT=(75.37+75.99)/2;  % H2O (liquid) molar heat capacity. We assume it as constant, using cp's mean value from 298.15K to 373K. [J/mol*K]
CpH2Ogas=35.22;         % H2O (gas) molar heat capacity at 500K. We assume it as constant. [J/mol*K]
HvapH2O=40660;          % H2O molar heat of vaporization
CpN2T=(28.87+29.25)/2;  % N2 (gas) molar heat capacity. We assume it as constant, using cp's mean value from 298.15K to 380K. [J/mol*K]
n=2;
DeffH2=0.149;           % Effective diffusion H2 in H2O
DeffO2=0.0295;          % Effective diffusion O2 in H2O
Dnaf=3.81e-6;           % Effective water diffusion in Nafion
MO2=31.999;             % O2 molecular weight [g/mol]
TcO2=154.4;             % O2 critical temperature [K]
pcO2=49.7;              % O2 critical pressure [atm]
MN2=28.013;             % N2 molecular weight [g/mol]
TcN2=126.2;             % N2 critical temperature [K]
pcN2=33.5;              % N2 critical pressure [atm]

ad=2.745e-4;
bd=1.823;
xO2dref=0.2;
xH2aref=0.2;

%% Resultant parameters
lambdaO2=1/O2u;
lambdaH2=1./H2u;
watermol=xH2Od/(1-xH2Od);
watertoO2=watermol/xO2air;
N2toO2=xN2air/xO2air;
xO2d=(lambdaO2-1)/((1+N2toO2+watertoO2)*lambdaO2-1);
DO2N2=(ad*(Toutlet/(sqrt(TcO2*TcN2)))^bd*(pcO2*pcN2)^(1/3)*(TcO2*TcN2)^(5/12)*(1/MO2+1/MN2)^(1/2))/pC; % Diffusivity of O2 through N2 [cm^2/s]
DeffO2N2=CathPor^tortuosity*DO2N2;
jlO2=n*2*F*DeffO2N2*xO2dref/tC;     % Limiting current density
jlH2=n*F*DeffH2/1*xH2aref/tA.*(0.9./H2u);
j=0.001:jlO2/Nsampling:jlO2;      % Sampling current densities

%% Conversion factors
H2O_kg_mol=18.01528/1000; % Conversion of liquid water: 1 mol is equal to 0.1801528 kg
H2_kg_mol=2*1.00794/1000;
O2_kg_mol=2*15.9994/1000;

%% Initialization of variables
O2FormEnthT=zeros(size(H2u,2),size(j,2));
O2FormEntrT=zeros(size(H2u,2),size(j,2));
H2FormEnthT=zeros(size(H2u,2),size(j,2));
H2FormEntrT=zeros(size(H2u,2),size(j,2));
H2OFormEnthT=zeros(size(H2u,2),size(j,2));
H2OFormEntrT=zeros(size(H2u,2),size(j,2));
H2O2RxnEnthT=zeros(size(H2u,2),size(j,2));
H2O2RxnEntrT=zeros(size(H2u,2),size(j,2));
GibbsRxn=zeros(size(H2u,2),size(j,2));
E0=zeros(size(H2u,2),size(j,2));
Eh=zeros(size(H2u,2),size(j,2));
C1=zeros(size(H2u,2),size(j,2));
alphaH2O=zeros(size(H2u,2),size(j,2));
Rmemb=zeros(size(H2u,2),size(j,2));
vOhmic=zeros(size(H2u,2),size(j,2));
vAct=zeros(size(H2u,2),size(j,2));
vActivation=zeros(size(H2u,2),size(j,2));
vConcentration=zeros(size(H2u,2),size(j,2));
VNernst=zeros(size(H2u,2),size(j,2));
Vcellout=zeros(size(H2u,2),size(j,2));
Pdensout=zeros(size(H2u,2),size(j,2));
iCell=zeros(1,size(j,2));
Vout=zeros(size(H2u,2),size(j,2));
Pout=zeros(size(H2u,2),size(j,2));
H2SupplyMolRate=zeros(size(H2u,2),size(j,2));
O2SupplyMolRate=zeros(1,size(j,2));
N2SupplyMolRate=zeros(1,size(j,2));
H2ConsMolRate=zeros(1,size(j,2));
O2ConsMolRate=zeros(1,size(j,2));
H2OProdMolRate=zeros(1,size(j,2));
H2OMembMolRate=zeros(size(H2u,2),size(j,2));
H2ExhaustMolRate=zeros(size(H2u,2),size(j,2));
O2ExhaustMolRate=zeros(1,size(j,2));
N2ExhaustMolRate=zeros(1,size(j,2));
H2OAnodeOutMolRate=zeros(size(H2u,2),size(j,2));
H2OCathodeInMolRate=zeros(1,size(j,2));
H2OAnodeInMolRate=zeros(size(H2u,2),size(j,2));
H2OCathodeOutMolRate=zeros(size(H2u,2),size(j,2));
Heatgen=zeros(size(H2u,2),size(j,2));
Texhaust=zeros(size(H2u,2),size(j,2));
Texhaust1=zeros(size(H2u,2),size(j,2));
Texhaust2=zeros(size(H2u,2),size(j,2));
Tmean=zeros(size(H2u,2),size(j,2));
HeatTo373=zeros(size(H2u,2),size(j,2));
H2OVapHeat=zeros(size(H2u,2),size(j,2));
TitleH2O=zeros(size(H2u,2),size(j,2));
HeatCoolingWater=zeros(size(H2u,2),size(j,2));
CoolingWatTempOut=zeros(size(H2u,2),size(j,2));
CoolingWatFlow=zeros(size(H2u,2),size(j,2));
CoolingConvCoeff=zeros(1,size(j,2));
vConc_Cathode=zeros(size(H2u,2),size(j,2));
vConc_Anode=zeros(size(H2u,2),size(j,2));
iPmax=zeros(1,size(H2u,2));
Re=zeros(1,size(j,2));

%% Loop to determine the exhaust gases temperature
Toutlet=ones(size(H2u,2),size(j,2))*Toutlet;  % Seed Outlet temperatures (before iteration process)

wb = waitbar(0,sprintf('Please wait... %2.2f%%',0));  % Creation of progress bar

for u=1:1:size(H2u,2) % Loop to run the calculations for all the H2 utilization values
    for i=1:1:size(j,2) % Loop to run the calculations for all the range of current densities
        
            %% Iteration to obtain the exhaust gases temperature at each sampling current density
            while (abs(Toutlet(u,i)-Texhaust(u,i)))>TempAccuracy
            % Preallocation
            Wcont=zeros(1,1,'sym');
            Mcond=zeros(1,1,'sym');
            xH2=zeros(1,1,'sym');
            xH2Oanode=zeros(1,1,'sym');
            xO2=zeros(1,1,'sym');
            xH2Ocathode=zeros(1,1,'sym');
            
            % Calculation of temperature dependant thermodynamic parameters
            Tmean(u,i)=(Toutlet(u,i)+Tinlet)/2;                                         % Estimation of reaction's mean temperature (K)
            O2FormEnthT(u,i)=O2FormEnthSTD+CpO2T*(Toutlet(u,i)-298.15);                 % Formation enthalpy of O2 (gas) at non-standard temperature [J/mol]
            O2FormEntrT(u,i)=O2FormEntrSTD+CpO2T*(Toutlet(u,i)-298.15)/Tmean(u,i);        % Formation entropy of O2 (gas) at non-standard temperature [J/mol]
            H2FormEnthT(u,i)=H2FormEnthSTD+CpH2T*(Toutlet(u,i)-298.15);                 % Formation enthalpy of H2 (gas) at non-standard temperature [J/mol]
            H2FormEntrT(u,i)=H2FormEntrSTD+CpH2T*(Toutlet(u,i)-298.15)/Tmean(u,i);        % Formation entropy of H2 (gas) at non-standard temperature [J/mol]
            H2OFormEnthT(u,i)=H2OFormEnthSTD+CpH2OT*(Toutlet(u,i)-298.15);              % Formation enthalpy of H2O (liquid) at non-standard temperature [J/mol]
            H2OFormEntrT(u,i)=H2OFormEntrSTD+CpH2OT*(Toutlet(u,i)-298.15)/Tmean(u,i);     % Formation entropy of H2O (liquid) at non-standard temperature [J/mol]
            H2O2RxnEnthT(u,i)=(1*H2OFormEnthT(u,i))-(1*H2FormEnthSTD+1/2*O2FormEnthSTD);% H2+(1/2)O2->H2O reaction's enthalpy [J/molH2]. WE ASSUME REACTANTS TO BE AT AMBIENT TEMPERATURE.
            H2O2RxnEntrT(u,i)=(1*H2OFormEntrT(u,i))-(1*H2FormEntrSTD+1/2*O2FormEntrSTD);% H2+(1/2)O2->H2O reaction's entropy [J/(molH2*K)]. WE ASSUME REACTANTS TO BE AT AMBIENT TEMPERATURE.

            GibbsRxn(u,i)=H2O2RxnEnthT(u,i)-Tmean(u,i)*H2O2RxnEntrT(u,i);   % Reaction's gibbs free energy accounting for the operating temperature
            E0(u,i)=-GibbsRxn(u,i)/(n*F);                               % Thermodynamic potential accounting for the operating temperature
            Eh(u,i)=abs(H2O2RxnEnthT(u,i))/(n*F);                       % Ideal enthalpy potential at operating temperature

            %% Cell membrane water content and ohmic losses
            syms alphae C z 
            % Governing equations for electrolyte water content
            eqns=[14*pA/pSat*(xH2Oa-tA*10^-4*alphae*j(i)*R*Tmean(u,i)/(2*F*pA*101325*DeffH2))==4.4*alphae+C,12.6+1.4*pC/pSat*(xH2Od-tC*10^-4*(1+alphae)*j(i)*R*Tmean(u,i)/(2*F*pC*101325*DeffO2))==4.4*alphae+C*exp(0.000598*j(i)*tM*10^-4/Dnaf)];
            sol=solve(eqns,alphae,C);
            alphaH2O(u,i)=double(sol.alphae);
            C1(u,i)=double(sol.C);
            alphaH2O(alphaH2O<0)=0;
            Wcont(z)=4.4*alphaH2O(u,i)+C1(u,i)*exp(0.000598*j(i)*z/Dnaf); % Water content in the membrane, as function of position (z)
            Mcond(z)=(0.005193*(4.4*alphaH2O(u,i)+C1(u,i)*exp(0.000598*j(i)*z/Dnaf))-0.00326)*exp(1268*(1/Tamb-1/Tmean(u,i))); % Membrane conductivity as function of z
            Rmemb(u,i)=double(vpaintegral(1./Mcond,0,tM*10^-4)); % Electrolyte ohmic resistance for the sampling current densities. Obtained by num. integrating (1/memb.cond(z)) from 0 to tM, for each of the sampling current densities. 
            vOhmic(u,i)=Rmemb(u,i)*j(i); % Ohmic loss at each of the sampling currents.

            %% Cell concentrations and activation and concentration losses
            xH2(z)=xH2a-z*(j(i)*R*Tmean(u,i))/(2*F*pA*101325*DeffH2); % Hydrogen concentration as function of anode distance
            xH2Oanode(z)=xH2Oa-z*(alphaH2O(u,i)*j(i)*R*Tmean(u,i))/(2*F*pA*101325*DeffH2); % Water vapor concentration as function of anode distance
            xO2(z)=xO2d-z*(j(i)*R*Tmean(u,i))/(4*F*pC*101325*DeffO2N2); % Oxygen concentration as function of cathode distance
            xH2Ocathode(z)=xH2Od+z*((1-alphaH2O(u,i))*j(i)*R*Tmean(u,i))/(2*F*pA*101325*DeffO2); % Water vapor concentration as function of cathode distance

            vAct(u,i)=(-R*Tmean(u,i)/(2*n*F*TransfCoeff)*log(j0O2)+R*Tmean(u,i)/(2*n*F*TransfCoeff)*log((j(i)+jleak)))+(-R*Tmean(u,i)/(n*F*(TransfCoeff+0.1))*log(j0H2(u))+R*Tmean(u,i)/(n*F*(TransfCoeff+0.1))*log((j(i)+jleak))); % Activation losses as function of current density
            vActivation(u,i)=double(vAct(u,i)); % Activation losses at the sampling density currents

            vConc_Cathode(u,i)=(R*Tmean(u,i)/(2*n*F))*(1+1/TransfCoeff)*log(jlO2/(jlO2-(j(i)+jleak))); % Concentration losses as function of current density and limiting current density
            vConc_Anode(u,i)=R*Tmean(u,i)/(n*F)*(1+1/TransfCoeff)*log(jlH2(u)/(jlH2(u)-(j(i)+jleak)));
            vConcentration(u,i)=vConc_Cathode(u,i)+vConc_Anode(u,i); % Concentration losses at the sampling density currents
       
            %% Cell Output voltage and power
            VNernst(u,i)=E0(u,i)-((R*Tmean(u,i))/(n*F))*log(1/(pA*xH2a*(pC*xO2air/(1+watermol))^0.5)); % Nerns equation: adding concentration effects to the thermodinamic potential.
            Vcellout(u,i)=VNernst(u,i)-vActivation(u,i)-vConcentration(u,i)-vOhmic(u,i); % Resultant output voltage considering the losses
            Pdensout(u,i)=Vcellout(u,i)*j(i);           % Power density output [W/cm2] per stack, accounting for all the losses
            iCell(i)=j(i)*CellArea;              % Current output per Fuel cell stack
            Vout(u,i)=Vcellout(u,i)*Ncells;          % Output voltage per Fuel cell stack
            Pout(u,i)=iCell(i)*Vcellout(u,i)*Ncells;   % Output power per stack

            %% Mass balance
            H2SupplyMolRate(u,i)=iCell(i)*(Ncells/(2*F)*lambdaH2(u));         % Total supply of H2 to the anode of the FC system [mol/s]
            O2SupplyMolRate(i)=iCell(i)*(Ncells/(4*F)*lambdaO2);         % Total supply of O2 to the cathode of the FC system [mol/s]
            N2SupplyMolRate(i)=O2SupplyMolRate(i)*N2toO2;                % Total supply of N2 (air mixture) to the cathode of the FC system [mol/s]

            H2ConsMolRate(i)=iCell(i)*(Ncells/(2*F));                    % H2 reaction consumption molar rate [mol/s]
            O2ConsMolRate(i)=iCell(i)*(Ncells/(4*F));                    % O2 reaction consumption molar rate [mol/s]
            H2OProdMolRate(i)=H2ConsMolRate(i);                          % H2O production molar rate due to H2+(1/2)O2->H2O reaction [mol/s]
            H2OMembMolRate(u,i)=iCell(i)*(Ncells/(2*F)*(1+alphaH2O(u,i)))-H2OProdMolRate(i);        % H2O diffusion molar rate through membrane [mol/s]

            H2ExhaustMolRate(u,i)=H2SupplyMolRate(u,i)-H2ConsMolRate(i);     % H2 total exhaust [mol/s]
            O2ExhaustMolRate(i)=O2SupplyMolRate(i)-O2ConsMolRate(i);     % O2 total exhaust [mol/s]
            N2ExhaustMolRate(i)=N2SupplyMolRate(i);                      % N2 total exhaust [mol/s]
            H2OAnodeOutMolRate(u,i)=(xH2Oa/xH2a)*H2SupplyMolRate(u,i);       % Total supply of H2O to the anode of the FC system [mol/s]
            H2OCathodeInMolRate(i)=(xH2Oa/xO2d)*O2ExhaustMolRate(i);     % H2O cathode total exhaust [mol/s]
            H2OAnodeInMolRate(u,i)=H2OAnodeOutMolRate(u,i)+H2OMembMolRate(u,i);% H2O anode total exhaust [mol/s]
            H2OCathodeOutMolRate(u,i)=H2OCathodeInMolRate(i)+H2OMembMolRate(u,i)+H2OProdMolRate(i); % H2O cathode inlet [mol/s]

            %% Thermal Balance
            Heatgen(u,i)=(Eh(u,i)-Vcellout(u,i))*iCell(i)*Ncells;

            Texhaust1(u,i)=(Heatgen(u,i)/(CpH2T.*H2ExhaustMolRate(u,i)+CpO2T*O2ExhaustMolRate(i)+CpH2OliqT*(H2OAnodeOutMolRate(u,i)+H2OCathodeOutMolRate(u,i))+CpN2T*N2ExhaustMolRate(i)))+Tinlet; % Calculation of exhaust temperature without considering H2O phase change
            Texhaust2(u,i)=(Heatgen(u,i)+Tinlet*(CpH2T*H2ExhaustMolRate(u,i)+CpO2T*O2ExhaustMolRate(i)+CpH2OliqT*(H2OAnodeOutMolRate(u,i)+H2OCathodeOutMolRate(u,i))+CpN2T*N2ExhaustMolRate(i))-(H2OAnodeOutMolRate(u,i)+H2OCathodeOutMolRate(u,i))*(373.15*CpH2Ogas+373.15*CpH2OliqT+HvapH2O))/(CpH2T*H2ExhaustMolRate(u,i)+CpO2T*O2ExhaustMolRate(i)+CpH2Ogas*(H2OAnodeOutMolRate(u,i)+H2OCathodeOutMolRate(u,i))+CpN2T*N2ExhaustMolRate(i)); % Calculation of exhaust temperature considering H2O phase change

            % Calculation of exhaust gases temperature and title
                if (Texhaust1(u,i)>373.15)&&(Texhaust2(u,i)<373.15)   % Without considering H2O phase change, Texhaust is >100C. Considering phase change, Texhaust is <100C. 
                                                            % Thus, we can assume that Texhaust=100C. Now, we must obtain the title of H2O (proportion of gas H2O over total H2O, at 373.15K)
                    Texhaust(u,i)=373.15;
                    HeatTo373(u,i)=(CpH2T.*H2ExhaustMolRate(u,i)+CpO2T.*O2ExhaustMolRate(i)+CpH2OliqT.*(H2OAnodeOutMolRate(u,i)+H2OCathodeOutMolRate(u,i))+CpN2T.*N2ExhaustMolRate(i)).*(373.15-Tinlet); % Power used to heat the exhaust species from inlet temperature to 373.15K
                    H2OVapHeat(u,i)=Heatgen(u,i)-HeatTo373(u,i); % Remaining power used to evaporate part of the exhaust water
                    TitleH2O(u,i)=H2OVapHeat(u,i)/((H2OAnodeOutMolRate(u,i)+H2OCathodeOutMolRate(u,i))*HvapH2O); % Title of exhaust H2O (proportion between gas H2O and total H2O) before going through the heat exchanger

                elseif Texhaust1(u,i)<373.15                    % Outlet gases temperature is below 100C. We have liquid H2O.
                    Texhaust(u,i)=Texhaust1(u,i);
                    TitleH2O(u,i)=0;
                else                                       % Outlet gases temperature is over 100C. We have gaseous H2O. 
                    Texhaust(u,i)=Texhaust2(u,i);
                    TitleH2O(u,i)=1;
                end

            Toutlet(u,i)=Texhaust(u,i)*CalcTempPonderation+Toutlet(u,i)*(1-CalcTempPonderation);% We adopt Toutlet as the mean value of old Toutlet and calculated Texhaust.
            
            end     % End of iteration to determine exhaust temperature without considering cooling water.

        %% Update of progress bar and results cleaning
        waitbar(i/size(j,2)*1/size(H2u,2)+(u-1)/size(H2u,2),wb,sprintf('Please wait... %2.2f%%',(i/size(j,2)*1/size(H2u,2)+(u-1)/size(H2u,2))*100)) % Progress bar update

        if 373.15-Toutlet(u,i)<TempAccuracy % Results cleaning
            Toutlet(u,i)=373.15;
        end
        symObj = syms;
        cellfun(@clear,symObj)
            
        %% Fuel Cell Water cooling calculation -if enabled-
        if (isequal(WaterCooling,'y'))||(isequal(WaterCooling,'Y'))||(isequal(WaterCooling,'yes'))||(isequal(WaterCooling,'Yes'))
            % Calculation of Cooling water flow and temperature to keep exhaust temperature below TexhaustMax
            if Toutlet(u,i)<=(TexhaustMax) % If exhaust temperature is below TexhaustMax, no water cooling is needed.
                CoolingWatFlow(u,i)=0;
                HeatCoolingWater(u,i)=0;
                CoolingWatTempOut(u,i)=CoolingWatTempIn;

            elseif Toutlet(u,i)>TexhaustMax    % Determination of cooling water flow when exhaust temperature is over TexhaustMax.
                Toutlet(u,i)=TexhaustMax;
                TitleH2O(u,i)=0;
                Re(i)=Re1*(Re2/Re1)^(i/size(j,2));
                CoolingConvCoeff(i)=CoolingConvCoeff_minVel/Re1*Re(i)^0.8;
                HeatCoolingWater(u,i)=-((Toutlet(u,i)-Tinlet)*(CpH2T.*H2ExhaustMolRate(u,i)+CpO2T*O2ExhaustMolRate(i)+CpH2OliqT*(H2OAnodeOutMolRate(u,i)+H2OCathodeOutMolRate(u,i))+CpN2T*N2ExhaustMolRate(i))-Heatgen(u,i)); % Heat that the cooling water has to remove to mantain the TexhaustMax [J/s]
                CoolingWatTempOut(u,i)=-(HeatCoolingWater(u,i)/(CoolingConvCoeff(i)*CoolingChannSurf)-Toutlet(u,i));   % Equation for convection heat exchange. We assume that FC surface temperature is equal to the exhaust temperature. Cooling water outlet temperature [K].
                CoolingWatFlow(u,i)=HeatCoolingWater(u,i)/((CoolingWatTempOut(u,i)-CoolingWatTempIn)*CpH2OTkg);   % Cooling water mass flow [kg/s] to mantain the exhaust gases at TexhaustMax.
            end
        end
    end
end
close(wb)

%% Heat exchanging calculations and system efficiencies
AirSupplyMolRate=O2SupplyMolRate+N2SupplyMolRate;
AirSupplyLPM=AirSupplyMolRate.*Rlit*Tinlet/pC*60;
BlowerPower=AirSupplyLPM/1000*pC*101325/17.4*(pC^0.283-1);  % Estimation of Blower consumption [W]. Equation from http://onlinembr.info/cost/blower-power-calculation/
ElectricalEff=(Pout-BlowerPower)./(H2SupplyMolRate*H2lhv*H2_kg_mol);

% Exhaust gases heat exchanger
Q_SpaceHeating=-((CpH2T*H2ExhaustMolRate+CpO2T*O2ExhaustMolRate+CpH2OliqT*(H2OAnodeOutMolRate+H2OCathodeOutMolRate)+CpN2T*N2ExhaustMolRate)*HXeff_exh.*(Tout_exh-Toutlet));
Q_SpaceHeating(Q_SpaceHeating<0)=0;
SpaceHeatingMolRate=Q_SpaceHeating./((CpO2T*xO2air+CpN2T*xN2air)*(SpaceHeatingAirTemp-Text));
SpaceHeatingLPM=SpaceHeatingMolRate*Rlit*SpaceHeatingAirTemp/1*60;
SpaceHeatingEff=Q_SpaceHeating./(H2SupplyMolRate*H2lhv*H2_kg_mol);

% Cooling water heat exchanger
Q_WaterHeating=-((CpH2OTkg.*CoolingWatFlow)*HXeff_wat.*(Tout_wat-CoolingWatTempOut));
Q_WaterHeating(Q_WaterHeating<0)=0;
WaterHeatingLPM=Q_WaterHeating./((CpH2OTkg)*(WaterHeatingTemp-Tamb))*60;
WaterHeatingEff=Q_WaterHeating./(H2SupplyMolRate*H2lhv*H2_kg_mol);

HeatEff=SpaceHeatingEff+WaterHeatingEff;
CHPEff=ElectricalEff+HeatEff;

NetElectricalPowerOut=Pout-BlowerPower;
NetHeatOut=Q_SpaceHeating+Q_WaterHeating;
NetTotalPowerOut=NetElectricalPowerOut+NetHeatOut;


%% PLOTS
set(groot, 'defaultLineLineWidth', 2)
set(0, 'DefaultAxesFontSize', 16);       % Axes tick labels
set(0, 'DefaultTextFontSize', 16);       % Titles, labels, annotations
set(0, 'DefaultColorbarFontSize', 16);   % Colorbar ticks (since R2018b)

H2utilization_plot=0.9; % Hydrogen utilization that plots will be based on
uplot=round((H2utilization_plot-H2u(1))/(H2u(end)-H2u(1))*(size(H2u,2)-1)+1);

figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen.
plot(j,vOhmic(uplot,:));
title('Ohmic loss')
xlabel('Current density [A/cm2]')
ylabel('Ohmic loss [V]')
ylim([0 inf])
xlim([0 jlO2])

figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen.
plot(j,vActivation(uplot,:));
title('Activation loss')
xlabel('Current density [A/cm2]')
ylabel('Activation loss [V]')
ylim([0 inf])
xlim([0 jlO2])

figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen.
plot(j,vConcentration(uplot,:));
title('Concentration loss')
xlabel('Current density [A/cm2]')
ylabel('Concentration loss [V]')
ylim([0 inf])
xlim([0 jlO2])

figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen.
plot(j,Vcellout(uplot,:))
title('j-V cell curve')
xlabel('Current density [A/cm2]')
ylabel('Cell voltage [V]')
ylim([0 inf])
xlim([0 jlO2])

%polarization and power density of single cell
figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen.
yyaxis left
xticks(0:0.2:jlO2)
plot(j,Vcellout(uplot,:))
xlabel('Current density [A/cm^2]')
ylabel('Cell voltage [V]')
ylim([0 inf])
xlim([0 jlO2])
yyaxis right
plot(j,Vcellout(uplot,:).*j)
ylabel('Power density [W/cm^2]')
% ylim([0 250])


figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen.
yyaxis left
plot(iCell,Vout(uplot,:))
title('i-V FC and Power output curve')
xlabel('Current [A]')
ylabel('Cell voltage [V]')
ylim([0 inf])
xlim([0 iCell(end)])
yyaxis right
plot(iCell,NetElectricalPowerOut(uplot,:)/1000)
ylabel('Power output [kW]')
ylim([0 250])

figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen.
plot(j,Pdensout(uplot,:))
title('Power density curve')
xlabel('Current density [A/cm2]')
ylabel('Power density [W/cm2]')
ylim([0 1.8])
xlim([0 jlO2])


if (isequal(WaterCooling,'y'))||(isequal(WaterCooling,'Y'))||(isequal(WaterCooling,'yes'))||(isequal(WaterCooling,'Yes'))
    figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen.
    yyaxis left
    plot(j,Toutlet(uplot,:)-273)
    ylabel('Exhaust temperature [C]')
    xlabel('Current density [A/cm2]')
    ylim([0 100])
    xlim([0 jlO2])
    title('Exhaust gases temperature and air flow rate')
    yyaxis right
    plot(j,AirSupplyLPM)
    ylim([0 inf])
    ylabel('Air flow rate [l/min]')
    legend('Exhaust temperature','Air flow rate','Location','northwest')
    
    figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen.
    yyaxis left
    plot(j,CoolingWatTempOut(uplot,:)-273.15)
    ylabel('Cooling water out temperature [C]')
    xlabel('Current density [A/cm2]')
    ylim([0 100])
    xlim([0 jlO2])
    yyaxis right
    plot(j,CoolingWatFlow(uplot,:)*60)
    ylabel('Cooling water flow rate [l/min]')
    ylim([0 inf])
    title('Cooling water temperature and flow rate')
    legend('Cooling water temp','Flow rate','Location','northwest')
else
    figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen.
    yyaxis left
    plot(j,Toutlet(uplot,:)-273)
    ylabel('Exhaust temperature [C]')
    xlabel('Current density [A/cm2]')
    ylim([0 130])
    xlim([0 jlO2])
    yyaxis right
    plot(j,TitleH2O(uplot,:))
    ylabel('H2O Title')
    ylim([0 1])
    title('Exhaust temperature and H2O title')
    legend('Exhaust temperature','H2O title')
end

figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen.
yyaxis left
plot(j,AirSupplyLPM)
title('Cathode Air supply rate and Blower estimated power')
xlabel('Current density [A/cm2]')
ylabel('Cathode Air supply [LPM]')
ylim([0 inf])
xlim([0 jlO2])
yyaxis right
plot(j,BlowerPower/1000)
ylabel('Blower estimated power [kW]')
ylim([0 inf])
legend('Cathode Air supply','Blower est. power','Location','northwest')

figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen.
plot(j,ElectricalEff(uplot,:))
title('FC Electrical efficiency (H2 lhv)')
xlabel('Current density [A/cm2]')
ylabel('Electrical efficiency')
ylim([0 1])
xlim([0 jlO2])

figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen.
plot(j,WaterHeatingEff(uplot,:),j,SpaceHeatingEff(uplot,:),j,HeatEff(uplot,:))
title('FC Heat efficiencies (H2 lhv)')
xlabel('Current density [A/cm2]')
ylabel('Efficiency')
ylim([0 1])
xlim([0 jlO2])
legend('Water Heating efficiency','Space Heating efficiency','Total Heat efficiency')

figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen.
plot(j,ElectricalEff(uplot,:),j,HeatEff(uplot,:),j,CHPEff(uplot,:))
title('FC efficiencies (H2 lhv)')
xlabel('Current density [A/cm2]')
ylabel('Efficiency')
ylim([0 1])
xlim([0 jlO2])
legend('Electrical efficiency','Heat efficiency','Total cogeneration efficiency')

figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen.
yyaxis left
plot(j,NetTotalPowerOut(uplot,:)/1000)
title('FC CHP power output and CHP efficiency')
xlabel('Current density [A/cm2]')
ylabel('Total power [kW]')
ylim([0 350])
xlim([0 jlO2])
yyaxis right
plot(j,CHPEff(uplot,:))
ylabel('CHP efficiency (H2 lhv)')
ylim([0 1])
legend('Total power output','Total cogeneration efficiency','location','southeast')

figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen.
yyaxis left
plot(j,NetElectricalPowerOut(uplot,:))
title('Electrical power output and Electrical efficiency')
xlabel('Current density [A/cm2]')
ylabel('Electrical power [kW]')
ylim([0 inf])
xlim([0 jlO2])
yyaxis right
plot(j,ElectricalEff(uplot,:))
ylabel('Electrical efficiency (H2 lhv)')
ylim([0 1])
legend('Electrical power output','Electrical efficiency')

figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen.
yyaxis left
plot(j,Q_WaterHeating(uplot,:)./1000,j,Q_SpaceHeating(uplot,:)./1000,j,NetHeatOut(uplot,:)./1000)
title('Heat output and Heat efficiency')
xlabel('Current density [A/cm2]')
ylabel('Heat [kW]')
ylim([0 inf])
xlim([0 jlO2])
yyaxis right
plot(j,HeatEff(uplot,:))
ylabel('Heat efficiency (H2 lhv)')
ylim([0 1])
legend('Water Heating output','Space Heating output', 'Total heat output','Heat efficiency')

figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen. 
plot(j,NetHeatOut(uplot,:)./(Ncells*CellArea))
title('Heat output power density')
ylabel('Power density [W/cm2]')
xlabel('Current density [A/cm2]')
ylim([0 inf])
xlim([0 jlO2])
legend('Heat output power density')

figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen. 
plot(j,(NetHeatOut(uplot,:))./NetElectricalPowerOut(uplot,:))
ylabel('Heat-to-electricity ratio')
ylim([0 inf])
xlim([0 jlO2])
legend('Heat-to-electricity ratio')

figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen.
yyaxis left
plot(j,NetTotalPowerOut(uplot,:)./1000,j,NetElectricalPowerOut(uplot,:)/1000,j,NetHeatOut(uplot,:)./1000)
title('Output powers and Fuel Cell efficiencies')
xlabel('Current density [A/cm2]')
ylabel('Power [kW]')
ylim([0 300])
xlim([0 jlO2])
yyaxis right
plot(j,CHPEff(uplot,:),j,ElectricalEff(uplot,:),j,HeatEff(uplot,:))
ylabel('Efficiency (H2 lhv)')
ylim([0 1])
legend('Total Power','Electrical Power','Heat','CHP efficiency','Electrical efficiency','Heat efficiency','location','southoutside','orientation','horizontal','Numcolumns',3)

figure; set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.15 0.45 0.55]); % Enlarge figure to 50% of full screen. 
title('FC Heat Recovery System: Power per Stack')
plot(j,Q_SpaceHeating(uplot,:)/1000,j,Q_WaterHeating(uplot,:)/1000)
ylabel('Power per Stack [kW]')
ylim([0 inf])
xlim([0 jlO2])
legend('Coolant HX','Exhaust HX')


[Pmaxu,iPmaxu]=max(Pdensout(uplot,:)); % Maximum power density point
jPmaxu=j(iPmaxu); % Current density at maximum power density point
VPmaxu=Vcellout(iPmaxu); % Output voltage at maximum power density point
fprintf('\n\nFor a hydrogen utilization of %.2f, the maximum power density output value is %.4f W/cm^2.\nIt is achieved for a density current of %.4f A/cm^2, when the cell output voltage is %.4f V.\n',H2u(uplot),Pmaxu,jPmaxu,VPmaxu)
fprintf('The maximum net electrical power is %.2f kW per stack, recovering a heat of %.2f kW.\n',NetElectricalPowerOut(uplot,iPmaxu)/1000,NetHeatOut(uplot,iPmaxu)/1000)
fprintf('In these operating conditions, the efficiencies are: ElectricalEff=%.2f ; HeatEff=%.2f ; CHPEff=%.2f.\n\n',ElectricalEff(uplot,iPmaxu),HeatEff(uplot,iPmaxu),CHPEff(uplot,iPmaxu)) 

%% Results cleaning for Simulink model 
for k=1:1:size(H2u,2)
    [~,iPmax(k)]=max(NetElectricalPowerOut(k,:)); % Maximum power density point
end
iPmaxtot=min(iPmax);
H2Consumption_kg=H2SupplyMolRate(:,1:iPmaxtot).*H2_kg_mol;
H2Exhaust_kg=H2ExhaustMolRate(:,1:iPmaxtot).*H2_kg_mol;
ElectricalPower=NetElectricalPowerOut(:,1:iPmaxtot);
RecoveredHeat=NetHeatOut(:,1:iPmaxtot);
V_FC=Vout(:,1:iPmaxtot);
i_FC=iCell(1,1:iPmaxtot);