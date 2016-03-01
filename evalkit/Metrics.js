// Discounted Cumulative Gain
var DCG = function(x_, k_) {
  var k = k_ || x_.length;
  var x = k_ ? x_.slice(0, k) : x_;
  var sum = x[0];
  for (var i = 1; i < x.length; i++) {
    var w = Math.log(2) / Math.log(i + 1);
    sum = sum + (w * x[i]);
  }
  return sum;
};

// Ideal Discounted Cumulative Gain
var IDCG = function(x_, k_) {
  var k = k_ || x_.length;
  var x = x_.slice(0, k); //NOTE: copy for sort mutation
  x = x.sort().reverse();
  return DCG(x);
};

// Normalized Discounted Cumulative Gain at 
var NDCG = function(x, k) {
  return DCG(x, k) / IDCG(x, k);
};

// Precisions at k=i for i=1 to x.length
var PrecisionVec = function(x) {
  var sum = 0;
  var P = [];
  for (var i = 0; i < x.length; i++) {
    if (x[i]) { sum++; }
    P[i] = sum / (i + 1);
  }
  return P;
};

// Precision at k
var Precision = function(x, k_) {
  var k = k_ || x.length - 1;
  return PrecisionVec(x)[k];
};

// Recalls at k=i for i=1 to x.length
var RecallVec = function(x, cutoff) {
  var sum = 0;
  var R = [];
  for (var i = 0; i < x.length; i++) {
    if (x[i]) { sum++; }
    R[i] = sum / cutoff;
  }
  return R;
};

// Recall at k_
var Recall = function(x, cutoff, k_) {
  var k = k_ || x.length - 1;
  return RecallVec(x, cutoff)[k];
};

// F1 score given precisions Pv and recalls Rv
var F1Vec = function(Pv, Rv) {
  var f1 = [];
  for (var i = 0; i < Pv.length; i++) {
    f1[i] = 2 * Pv[i] * Rv[i] / (Pv[i] + Rv[i]);
  }
  return f1;
};

// F1 score at k_
var F1 = function(x, cutoff, k_) {
  var k = k_ || x.length - 1;
  var Pv = PrecisionVec(x);
  var Rv = RecallVec(x, cutoff);
  return F1Vec(Pv, Rv)[k];
};

// Average precision over x
var AveragePrecision = function(x) {
  var sum = 0;
  var precisions = [];
  for (var i = 0; i < x.length; i++) {
    if (x[i]) {
      sum++;
      precisions.push(sum / (i + 1));
    }
  }
  var sumPrecisions = 0;
  for (var j = 0; j < precisions.length; j++) {
    sumPrecisions += precisions[j];
  }
  if (precisions.length > 0) {
    return sumPrecisions /precisions.length;
  } else {
    return 0;
  }
};

var MetricsSet = function(x, cutoff) {
  this.p    = PrecisionVec(x);
  this.r    = RecallVec(x, cutoff);
  this.f1   = F1Vec(this.p, this.r);
  this.ap   = AveragePrecision(x);
  this.ndcg = NDCG(x);
  this.num  = 1;
};

// Collection of retrieval performance summary stats
var SummedMetrics = function() {
  this.pSum    = [];
  this.rSum    = [];
  this.f1Sum   = [];
  this.pNSum   = 0;
  this.rNSum   = 0;
  this.f1NSum  = 0;
  this.apSum   = 0;
  this.ndcgSum = 0;
  this.numSum  = 0;
};

var addVec = function(x, y) {
  var o = [];
  for (var i = 0; i < Math.max(x.length, y.length); i++) {
    o[i] = (x[i] ? x[i] : 0) + (y[i] ? y[i] : 0);
  }
  return o;
};

SummedMetrics.prototype.addResult = function(x, cutoff) {
  var m = new MetricsSet(x, cutoff);
  this.addMetricsSet(m);
};

SummedMetrics.prototype.addMetricsSet = function(m) {
  this.pSum     = addVec(this.pSum, m.p);
  this.rSum     = addVec(this.rSum, m.r);
  this.f1Sum    = addVec(this.f1Sum, m.f1);
  this.pNSum   += m.p[m.p.length-1];
  this.rNSum   += m.r[m.r.length-1];
  this.f1NSum  += m.f1[m.f1.length-1];
  this.apSum   += m.ap;
  this.ndcgSum += m.ndcg;
  this.numSum  += m.num;
};

var mulVec = function(x, m) {
  var o = [];
  for (var i = 0; i < x.length; i++) {
    o[i] = x[i] * m;
  }
  return o;
};

SummedMetrics.prototype.addSummedMetrics = function(s) {
  var avg = s.getAverages();
  this.pSum     = addVec(this.pSum, avg.P);
  this.rSum     = addVec(this.rSum, avg.R);
  this.f1Sum    = addVec(this.f1Sum, avg.F1);
  this.pNSum   += avg['P@N'];
  this.rNSum   += avg['R@N'];
  this.f1NSum  += avg['F1@N'];
  this.apSum   += avg.mAP;
  this.ndcgSum += avg.NDCG;
  this.numSum++;  // add one for macro averaging since s is already summed
};

SummedMetrics.prototype.getAverages = function() {
  var norm = 1 / this.numSum;
  var p = mulVec(this.pSum, norm);
  var r = mulVec(this.rSum, norm);
  var f1 = mulVec(this.f1Sum, norm);
  var pn = this.pNSum * norm;
  var rn = this.rNSum * norm;
  var f1n = this.f1NSum * norm;
  var ap = this.apSum * norm;
  var ndcg = this.ndcgSum * norm;
  return {
    'P':    p,
    'R':    r,
    'F1':   f1,
    'P@N':  pn,
    'R@N':  rn,
    'F1@N': f1n,
    'mAP':  ap,
    'NDCG': ndcg,
    'num':  this.numSum
  };
};

module.exports = {
  DCG: DCG,
  IDCG: IDCG,
  NDCG: NDCG,
  Precision: Precision,
  Recall: Recall,
  PrecisionVec: PrecisionVec,
  RecallVec: RecallVec,
  F1Vec: F1Vec,
  F1: F1,
  AveragePrecision: AveragePrecision,
  SummedMetrics: SummedMetrics
};
