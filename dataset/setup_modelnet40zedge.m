function imdb = setup_modelnet40zedge(datasetDir, varargin)

imdb = setup_modelnet_(datasetDir, 'invert', true, varargin{:}); 
