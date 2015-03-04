function imdb = setup_modelnet10toonedge(datasetDir, varargin)

imdb = setup_modelnet_(datasetDir, 'invert', true, varargin{:}); 
