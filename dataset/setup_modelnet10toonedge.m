function imdb = setup_modelnet10toonedge(datasetDir, varargin)

imdb = setup_modelnet(datasetDir,'suffix','toonedge','invert',true,varargin{:}); 