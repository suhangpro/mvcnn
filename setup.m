function setup(doCompile, matconvnetOpts)
% SETUP  Setup paths, dependencies, etc.
% 
%   doCompile:: false
%       Set to true to compile the libraries
%   matconvnetOpts:: struct('enableGpu',false)
%       Options for vl_compilenn

if nargin==0, 
    doCompile = false;
elseif nargin<2, 
    matconvnetOpts = struct('enableGpu', false); 
end

if doCompile && gpuDeviceCount()==0 ...
    && isfield(matconvnetOpts,'enableGpu') && matconvnetOpts.enableGpu, 
    fprintf('No supported gpu detected! ');
    return;
end

addpath(genpath('dataset'));
addpath(genpath('utils'));

% -------------------------------------------------------------------------
%                                                                liblinear
% -------------------------------------------------------------------------
if doCompile, 
    !make -C dependencies/liblinear-1.96/ clean
    !make -C dependencies/liblinear-1.96/
    run dependencies/liblinear-1.96/matlab/make.m
end
addpath('dependencies/liblinear-1.96/matlab');

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
    cd dependencies/matconvnet
    vl_compilenn(matconvnetOpts);
    cd ../..
end
run dependencies/matconvnet/matlab/vl_setupnn.m
