function imdb = setup_modelnet40toonedge(datasetDir, varargin)

imdb = setup_modelnet(datasetDir, 'invert', true, varargin{:}); 