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
    if ispc
        cd('dependencies/liblinear-1.96/');
        % change to vcvars32 for 32-bit platforms
        !vcvars64 & nmake /f Makefile.win clean & nmake /f Makefile.win lib
        cd('../../');
    else
        !make -C dependencies/liblinear-1.96/ clean
        !make -C dependencies/liblinear-1.96/
    end
    run dependencies/liblinear-1.96/matlab/make.m
end
addpath('dependencies/liblinear-1.96/matlab');

% -------------------------------------------------------------------------
%                                                                   vlfeat
% -------------------------------------------------------------------------
if doCompile,
    if ispc
        cd('dependencies/vlfeat/');
        % change to vcvars32 for 32-bit platforms
        % if there is a crash here, you need to explicitly change
        % the paths and parameters in vlfeat's Makefile.mak        
        % also remove "-f $(MEXOPT)" from Makefile.mak for latest VS
        % versions. 
        !vcvars64 & nmake /f Makefile.mak clean & nmake /f Makefile.mak
        cd('../../');
    else    
    	!make -C dependencies/vlfeat/ clean
        !make -C dependencies/vlfeat/
    end
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
addpath('dependencies/matconvnet/examples/imagenet');
