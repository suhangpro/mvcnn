var Evaluator = require('./Evaluator.js');
var evaluator = new Evaluator({truthFile: 'val.csv'});
evaluator.evaluate('path/to/dir/with/results/');
