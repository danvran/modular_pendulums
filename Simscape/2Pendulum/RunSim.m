clear; % Clear workspace
anomalies = true;  % Set to true to simulate with anomalies
variables;  % Run script to initialize all model variables

if anomalies == true  % This must run after the "variables" script
    % Specify the directory path
    anomalyPath = './anomalies';
    anomalyFiles = dir(anomalyPath);
    for i = 1:length(anomalyFiles)
        if anomalyFiles(i).isdir == 0 % Assure it's not a directory
            filename = anomalyFiles(i).name;
            disp(['Processing file: ' filename]);
            variables;  % Run script to initialize all model variables
            runtime = 1000.0;  % half the runtime of the normal data
            scriptPath = fullfile(anomalyPath, filename);
            run(scriptPath);
            processFile(csvname);
        end
    end
else
    disp(['Processing file: ' csvname]);
    processFile(csvname)
end


% Function to process each file
function processFile(csvname)
    simOut = sim('two_pendulum_sl.slx');
    
    T = table(sin1, ...
    cos1, ...
    tf1, ...
    tt1, ...
    sin2, ...
    cos2, ...
    tf2, ...
    tt2);

    %T(1,:) = [];
    writetable(T, csvname)
    
end