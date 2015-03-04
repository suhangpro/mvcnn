function imdb = setup_modelnet10toon(datasetDir, varargin)

imdb = setup_modelnet_(datasetDir, 'invert', false, varargin{:}); 
