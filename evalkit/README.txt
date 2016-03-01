In order to run this evaluation code you must have node.js installed on your system (https://nodejs.org/en/).

Then create an evaluate.js file with the following contents:
var Evaluator = require('./Evaluator.js');
var evaluator = new Evaluator({truthFile: 'path/to/val.csv'});
evaluator.evaluate('path/to/dir/with/results/');

The 'path/to/val.csv' refers to the category ground truth file (can also be train.csv), while 'path/to/dir/with/results/' refers to the directory containing all ranked list result files.  The code will print out a csv format set of evaluation metric statistics for each category, as well as a micro-average and macro-average across all categories.  The same information is saved in a 'summary.csv' file in the working directory.  In addition, a 'PR.csv' file is saved with precision-recall values that can be used to generate a Precision-Recall plot.

Please contact the organizers at shrec2016shapenet@gmail.com if you have any questions or issues.

UPDATES:
2016-03-01 - Fixed aggregation bug for P@N, R@N, and F1@N when N (retrieval list length) varies. Also added writeOracleResults function to Evaluator