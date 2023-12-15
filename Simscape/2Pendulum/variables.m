% Use Simulink.Parameters to populate workspace variables for Simulink
% Run this script to initialize all normal behavior variables
% Then run an anomaly script or multiple ones to generate anomalies

% File export path
csvname = './data/NormalOperation.csv';

% Gloabal Variables
runtime = Simulink.Parameter(2000.0);
samplefrequency = 100.0;
sampletime = Simulink.Parameter(1/samplefrequency);  % T_s and T_0
M1 = Simulink.Parameter(100.0);
M2 = Simulink.Parameter(100.0);
M3 = Simulink.Parameter(100.0);
M4 = Simulink.Parameter(100.0);
M5 = Simulink.Parameter(100.0);
M6 = Simulink.Parameter(100.0);
M7 = Simulink.Parameter(100.0);
M8 = Simulink.Parameter(100.0);
