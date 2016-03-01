var fs = require('fs');
var Metrics = require('./Metrics.js');

function Evaluator(params) {
  this.truth = {};       // id -> {synset, subsynset}
  this.bySynset = {};    // synset -> [id]
  this.bySubSynset = {}; // subsynset -> [id]
  this.outDir = params.outDir || 'out/';
  var scope = this;
  fs.readFileSync(params.truthFile).toString().split('\n').forEach(function(line) {
    var tokens = line.split(',');
    var id = tokens[0];
    if (id === 'id') { return; }  // this is the header line so skip it
    var synset = tokens[1];
    var subsynset = tokens[2];
    scope.truth[id] = {synset: synset, subsynset: subsynset};
    if (!scope.bySynset[synset]) scope.bySynset[synset] = [];
    scope.bySynset[synset].push(id);
    if (!scope.bySubSynset[subsynset]) scope.bySubSynset[subsynset] = [];
    scope.bySubSynset[subsynset].push(id);
  });
}
Evaluator.prototype.constructor = Evaluator;

Evaluator.prototype.resultsToScores = function(queryId, results) {
  var query = this.truth[queryId];
  var scores = [];
  for (var i = 0; i < results.length; i++) {
    var result = this.truth[results[i]];
    if (query.synset !== result.synset) {  // parent synset mismatch
      scores[i] = 0;
    } else {  // parent synset match, check subsynset
      if (query.subsynset === result.subsynset) {  // subsynset match
        scores[i] = 3;
      } else if (query.synset === result.subsynset) {  // predicted subsynset is query synset
        scores[i] = 2;
      } else {  // must be a sibling subsynset since match
        scores[i] = 1;
      }
    }
  }
  //console.log(queryId + ":" + scores);
  return scores;
};

// Reads all query results from given dir
var readQueryResults = function(dir) {
  var files = fs.readdirSync(dir);
  var allQueryResults = {};

  // helper converts line in results file and pushes to results
  var lineToQueryResult = function(results, line) {
    if (line) {
      var tokens = line.split(' ');
      var id = tokens[0];
      //var dist = tokens[1];  // ignore dists for now
      results.push(id);
    }
  };

  // iterate over files and convert each to results list
  for (var iF = 0; iF < files.length; iF++) {
    var file = files[iF];
    var results = [];
    var line2result = lineToQueryResult.bind(undefined, results);
    fs.readFileSync(dir + file).toString().split('\n').forEach(line2result);
    allQueryResults[file] = results;
  }

  return allQueryResults;
};

// save average Precision-Recall values for each id in avgs to filename
var savePRs = function(avgs, filename) {
  var f = fs.createWriteStream(filename);
  f.on('error', function(err) { console.error(err); });
  f.write("class,P,R\n");
  for (var id in avgs) {
    if (avgs.hasOwnProperty(id)) {
      var a = avgs[id].getAverages();
      var P = a.P;  var R = a.R;
      for (var k = 0; k < P.length; k++) {
        var l = id + "," + a.P[k] + "," + a.R[k] + "\n";
        f.write(l);
      }
    }
  }
  f.end();
};

// save average metrics for each id in avgs
var saveAverages = function(avgs, filename) {
  var f = fs.createWriteStream(filename);
  f.on('error', function(err) { console.error(err); });
  var header = "class,P@N,R@N,F1@N,mAP,NDCG";
  console.log(header);  f.write(header + '\n');
  for (var id in avgs) {
    if (avgs.hasOwnProperty(id)) {
      var a = avgs[id].getAverages();
      var l = [id, a['P@N'], a['R@N'], a['F1@N'], a.mAP, a.NDCG].join(',');
      console.log(l);  f.write(l + '\n');
    }
  }
  f.end();
};

Evaluator.prototype.evaluate = function(dir) {
  var queries = readQueryResults(dir);
  var cutoff = 1000;  // only consider recall up to this retrieval list length
  var metrics = {'microALL': new Metrics.SummedMetrics()};
  for (var queryId in queries) {
    if (!queries.hasOwnProperty(queryId)) { continue; }
    var queryTruth = this.truth[queryId];
    var querySynset = queryTruth.synset;
    var results = queries[queryId];
    if (results.length > cutoff) {  // only accept up to cutoff results
      results = results.slice(0, cutoff);
    }
    var maxPossibleTrue = Math.min(this.bySynset[querySynset].length, cutoff);
    var scores = this.resultsToScores(queryId, results);
    metrics.microALL.addResult(scores, maxPossibleTrue);
    if (!metrics[querySynset]) { metrics[querySynset] = new Metrics.SummedMetrics(); }
    metrics[querySynset].addResult(scores, maxPossibleTrue);
  }

  // Computer macro average = average of per-synset micro averages
  metrics.macroALL = new Metrics.SummedMetrics();
  for (var synId in metrics) {
    if (metrics.hasOwnProperty(synId) && synId !== 'microALL') {
      metrics.macroALL.addSummedMetrics(metrics[synId]);
    }
  }

  saveAverages(metrics, 'summary.csv');
  savePRs(metrics, 'PR.csv');
};

// Writes oracle query results to given dir, cutting off lists at maxN
Evaluator.prototype.writeOracleResults = function(dir, maxN) {
  // for each id in truth table
  for (var modelId in this.truth) {
    if (!this.truth.hasOwnProperty(modelId) || modelId === '') { continue; }
    var model = this.truth[modelId];
    var synsetId = model.synset;
    // get all ids with same synset
    var sameSynsetModelIds = this.bySynset[synsetId];
    if (sameSynsetModelIds.length > maxN) {
      sameSynsetModelIds = sameSynsetModelIds.slice(0, maxN);
    }
    var file = dir + '/' + modelId;
    fs.writeFileSync(file, sameSynsetModelIds.join('\n'));
  }
};

module.exports = Evaluator;