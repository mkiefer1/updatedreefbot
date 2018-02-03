function [scoreOut]=mvg_scoreWindows(img,intialWindows,config)
% function [windowsOut,scoreOut]=mvg_runObjectDetection(img,config)
% returns a set of candidate windows which are prominent to contain 
% objects over broad variety of classes.
%
% Inputs:
% img, imgRow*imgCol*3, double, is the input color image.
% config, struct, contains the parameter settings. See below for formatting 
%                 details. If undefined, default configuration will be used.
% initialWindows, numWindows*4, double, is a matrix with the candidate
%                                       windows. Each row corresponds to
%                                       one window in format
%                                       initialWindows(i,:)=[xmin,ymin,xmax,ymax]
%
% Outputs:
% windowsOut, numWindows*4, double, is a matrix containing the candidate 
%                                   windows. Each row corresponds to one
%                                   window in format:
%                                   windowsOut(i,:)=[xmin,ymin,xmax,ymax];
% scoreOut, numWindows*1, double, is a vector of objectness scores associated
%                                 with each window in windowsOut. i:th entry
%                                 corresponds to i:th row in windowsOut.
%
% Example usage:
% >>img=imread('exampleImage.jpg');
% >>[windowsOut,scoreOut]=mvg_runObjectDetection(img);
% >>mvg_drawWindows(img,windowsOut((1:10),:));

% This program implements the method described in:
%
% Rahtu E. & Kannala J. & Blaschko M. B. 
% Learning a Category Independent Object Detection Cascade. 
% Proc. International Conference on Computer Vision (ICCV 2011).

% 2011 MVG, Oulu, Finland, Esa Rahtu and Juho Kannala 
% 2011 VGG, Oxford, UK, Matthew Blaschko

%% Default configuration (for details, please refer to the paper above)
% The parameters you most likely need to change are config.NMS.numberOfOutputWindows that
% defines the number of output windows and config.NMS.trhNMSb, which defines the non-maxima
% suppression threshold (0.4-0.6 gives better recall with lower overlaps, 0.6-> gives better reall with high overlaps).
if ~exist('config','var') || isempty(config)
    %% Initial window sampling parameters
    % General

    %% Feature parameters
    config.Features.loadFeatureScores=false; % false->nothing is loaded, true->load feature scores from the file given in storage parameter below (overrides all other initial feature settings)
    config.Features.featureTypes={'SS','WS','BE','BI'}; % Feature to be computed from initial windows
    config.Features.featureWeights=[1.685305e+00, 7.799898e-02, 3.020189e-01, -7.056292e-04]; % Relative feature weights (same ordering as with the features above)
    
    %% General parameters
    config.verbose=0; % 0->show nothing, 1->display progress 
end

%% Initialize
% General
timeStamp=clock;
config.Features.verbose=config.verbose;

% Check loading paramets
if config.Features.loadFeatureScores && ~config.InitialWindows.loadInitialWindows
    error('If you load feature scores, also corresponding windows must be loaded. Otherwise windows and scores do not match!');
end

% Ensure superpixel representation exists if it is needed anywhere in the algorithm 
if ((sum(strcmp('SS',config.Features.featureTypes))>eps || sum(strcmp('BI',config.Features.featureTypes))) && ~config.Features.loadFeatureScores)
    % Compute superpixels if they are not in config.superPixels
    if ~isfield(config,'superPixels')
        % Display
        if config.verbose>eps
            fprintf('\nComputing superpixels...\n');
            intermediateTimeStamp=clock;
        end
        % Compute superpixels
        config.superPixels=mvg_computeSuperpixels(img);
        % Display
        if config.verbose>eps
            fprintf('Superpixels computed! Time taken %1.2f sec.\n',etime(clock,intermediateTimeStamp));
        end
    end
    % Store superpixels for initial windows and feature scoring
    config.InitialWindows.superPixels=config.superPixels;
    config.Features.superPixels=config.superPixels;
end


%% Compute feature scores 
% Display
if config.verbose>eps
    fprintf('\nComputing feature scores...\n');
    intermediateTimeStamp=clock;
end
% Run score computation
featureScores=mvg_computeFeatureScores(img,intialWindows,config.Features);
% Display
if config.verbose>eps
    fprintf('Features computed! Time taken %1.2f sec.\n',etime(clock,intermediateTimeStamp));
end


%% Make combined score
% Display
if config.verbose>eps
    fprintf('\nComputing combined scores...\n');
    intermediateTimeStamp=clock;
end
% Compute linear combination
combinedScores=(config.Features.featureWeights*featureScores')';
% Display
if config.verbose>eps
    fprintf('Combined score computed! Time taken %1.2f sec.\n',etime(clock,intermediateTimeStamp));
end

scoreOut = combinedScores;

%% Print done
if config.verbose>eps
    fprintf('Done!\n');
    fprintf('Total time taken: %1.2f sec.\n',etime(clock,timeStamp));
end



