function imdb = setup_modelnet30toon(datasetDir, varargin)

imdb = setup_modelnet_(datasetDir, 'invert', false, varargin{:}); 
