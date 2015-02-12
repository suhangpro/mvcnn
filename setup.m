function setup(doCompile, gpuMode)
% SETUP  Setup paths, dependencies, etc.
% 
%   doCompile:: false
%       Set to true to compile the libraries
%   gpuMode:: false
%       Set to true to enable GPU support

if nargin==0, 
    doCompile = false;
elseif nargin<2, 
    gpuMode = false; 
end

if doCompile && gpuDeviceCount()==0 && gpuMode, 
    fprintf('No supported gpu detected! ');
    gpuMode = false;
    reply = input('Continue w/ cpu mode? Y/N [Y]:','s');
	if ~isempty(reply) && reply~='Y', 
        return;
	end
end

addpath(genpath('dataset'));
addpath(genpath('utils'));

% -------------------------------------------------------------------------
%                                                           matlab-helpers
% -------------------------------------------------------------------------
addpath('dependencies/matlab-helpers');

% -------------------------------------------------------------------------
%                                                                   vlfeat
% -------------------------------------------------------------------------
if doCompile,
	!make -C dependencies/vlfeat/ clean
    !make -C dependencies/vlfeat/
end
run dependencies/vlfeat/toolbox/vl_setup.m

% -------------------------------------------------------------------------
%                                                               matconvnet
% -------------------------------------------------------------------------
if doCompile, 
    run dependencies/matconvnet/matlab/vl_setupnn.m
    vl_compilenn('enableGpu', gpuMode, 'enableImreadJpeg', gpuMode);
end
run dependencies/matconvnet/matlab/vl_setupnn.m
addpath dependencies/matconvnet/examples/