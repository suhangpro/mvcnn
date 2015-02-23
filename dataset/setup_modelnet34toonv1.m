function imdb = setup_modelnet34toonv1(datasetDir, varargin)

imdb = setup_modelnet(datasetDir,'suffix','toon','invert',false,varargin{:}); 