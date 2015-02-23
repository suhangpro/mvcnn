function imdb = setup_modelnet34toonedgev2(datasetDir, varargin)

imdb = setup_modelnet(datasetDir,'suffix','toonedge','invert',true,varargin{:}); 