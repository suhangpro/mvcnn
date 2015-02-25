function imdb = setup_modelnet40toon(datasetDir, varargin)

imdb = setup_modelnet(datasetDir, 'invert', false, varargin{:}); 