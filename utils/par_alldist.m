function Dist = par_alldist(x1,x2,varargin)
% a parallelized version of vl_alldist2

opts.measure = 'L2';
opts.maxParts = 100;
opts.maxMem = [];
opts.numWorkers = 20;
opts.verbose = true;
opts = vl_argparse(opts,varargin);

if isempty(x2) || isequal(x1,x2),
    if isempty(x2), x2 = x1; end
    withSelf = true;
else
    withSelf = false;
end

if opts.numWorkers<=1, 
    Dist = vl_alldist(x1,x2,opts.measure);
    return;
end

typeX1 = class(x1); typeX2 = class(x2);
if ~strcmp(typeX1,typeX2), 
    error('Wrong input: x1 and x2 should be of same data type');
end

[D1, n1] = size(x1); [D2, n2] = size(x2);
if D1~=D2, 
    error('Wrong input: x1 and x2 don''t match in 1st dimention'); 
end; 
D = D1;

% see how much memory is available
tmp = x1(1);
info = whos('tmp');
numbytes = info.bytes;
if ~ispc
    [~,w] = unix('free | grep Mem');
    stats = str2double(regexp(w, '[0-9]*', 'match'));
    freemem = (stats(3)+stats(end))*1e3;
else
    stats = memory;
    freemem = stats.MemAvailableAllArrays;
end
if isempty(opts.maxMem), 
    maxmem = freemem * 0.8;
else
    maxmem = min(opts.maxMem,freemem);
end
maxNum = maxmem / numbytes;

% decide window width w.r.t. constraints
w = ceil(max(sqrt(n1*n2/opts.maxParts),2*n1*n2*D/(maxNum-2*n1*n2)));
npar = ceil([n1 n2]/w);
while npar(1)*npar(2)>opts.maxParts, 
    npar = max(1,npar - 1);
    w = ceil(max([n1 n2]./npar));
    npar = ceil([n1 n2]/w);
end
sz1 = [ones(1,npar(1)-1)*w n1-(npar(1)-1)*w];
sz2 = [ones(1,npar(2)-1)*w n2-(npar(2)-1)*w];

% partition data
t = npar(1)*npar(2);
x1cell = cell(1,t);
x2cell = cell(1,t);
distcell = cell(1,t);
if opts.verbose, 
    ts = tic;
    fprintf('[1/3] Partitioning into %dx ((%d,%d),(%d,%d)) parts \n', ...
        t, D,w,D,w);
end
for i = 1:t,
    i1 = mod(i-1,npar(1))+1;
    i2 = floor((i-1)/npar(1))+1;
    if withSelf && i2>i1, % skip top-right triangle for self-dist
        x1cell{i} = [];
        x2cell{i} = [];
    else
        x1cell{i} = x1(:,(i1-1)*w+(1:sz1(i1)));
        x2cell{i} = x2(:,(i2-1)*w+(1:sz2(i2)));
    end
    if opts.verbose, 
        if mod(i,10)==0, fprintf('.');end;
        if mod(i,200)==0, fprintf(' [%d/%d]\n',i,t); end;
    end
end
if opts.verbose, fprintf(' done! (%s)\n', timestr(toc(ts))); end;

% estimate speed
tmp = rand(1000,1000);
tt=tic;vl_alldist2(tmp,tmp,opts.measure);tc=toc(tt);
if withSelf, 
    estTime = (D*(n1*n2/2-n1/2)/1e9)*tc;
else
    estTime = (D*n1*n2/1e9)*tc;
end

% real work
%{-
% comment this block for ealier MATLAB versions 
pool = gcp('nocreate');
if isempty(pool) || pool.NumWorkers<opts.numWorkers, 
    if ~isempty(pool), delete(pool); end
    pool = parpool(opts.numWorkers);
end
if opts.verbose, 
    ts = tic;
    fprintf('[2/3] Computing distances using %d workers (~%s) ...', ...
    pool.NumWorkers, timestr(estTime/pool.NumWorkers));
end
%}
fprintf('\n'); 
parfor_progress(t); 
parfor i=1:t,
    distcell{i} = vl_alldist2(x1cell{i},x2cell{i},opts.measure);
    parfor_progress(); 
end
parfor_progress(0); 
if opts.verbose, fprintf(' done! (%s)\n', timestr(toc(ts))); end;

% assemble results
if opts.verbose, 
    ts = tic;
    fprintf('[3/3] Assembling distance matrix \n'); 
end;
Dist = zeros(n1,n2,typeX1);
for i = 1:t,
    i1 = mod(i-1,npar(1))+1;
    i2 = floor((i-1)/npar(1))+1;
    if withSelf && i2>i1, 
        dist_block = distcell{(i1-1)*npar(1)+i2}';
    else
        dist_block = distcell{i};
    end
    Dist((i1-1)*w+(1:sz1(i1)),(i2-1)*w+(1:sz2(i2))) = dist_block;
    if opts.verbose, 
        if mod(i,10)==0, fprintf('.'); end
        if mod(i,200)==0, fprintf(' [%d/%d]\n',i,t); end;
    end
end
if opts.verbose, fprintf(' done! (%s)\n', timestr(toc(ts))); end;

end

function str=timestr(nsecs)

muls = [60 60];
% label = {'H','M','S'};
label = {'','',''};
tvec = [nsecs];

for i=1:numel(muls)
    tvec = [floor(tvec(1)/muls(end-i+1)) tvec];
    tvec(2) = mod(tvec(2),muls(end-i+1));
end

if tvec(end)~=floor(tvec(end)) 
    str = sprintf('%02.2f%s',tvec(end),label{end});
else
    str = sprintf('%02d%s',tvec(end),label{end});
end
for i=2:numel(tvec), 
    str = sprintf('%02d%s:%s',tvec(end-i+1),label{end-i+1},str);
end

end
