function imdb = setup_modelnet10toon(datasetDir, varargin)

imdb = setup_modelnet(datasetDir,'suffix','toon','invert',false,varargin{:}); 