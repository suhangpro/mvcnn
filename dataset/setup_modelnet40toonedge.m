function imdb = setup_modelnet40toonedge(datasetDir, varargin)

imdb = setup_modelnet_(datasetDir, 'invert', true, varargin{:}); 
