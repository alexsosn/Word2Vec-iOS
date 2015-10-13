
/*
A MagicNet takes data: a list of convnetjs.Vol(), and labels
which for now are assumed to be class indeces 0..K. MagicNet then:
- creates data folds for cross-validation
- samples candidate networks
- evaluates candidate networks on all data folds
- produces predictions by model-averaging the best networks
*/
class MagicNet {
    
    var data: AnyObject?
    var labels: AnyObject?
    var train_ratio: AnyObject?
    var num_folds: AnyObject?
    var num_candidates: AnyObject?
    var num_epochs: AnyObject?
    var ensemble_size: AnyObject?
    var batch_size_min: AnyObject?
    var batch_size_max: AnyObject?
    var l2_decay_min: AnyObject?
    var l2_decay_max: AnyObject?
    var learning_rate_min: AnyObject?
    var learning_rate_max: AnyObject?
    var momentum_min: AnyObject?
    var momentum_max: AnyObject?
    var neurons_min: AnyObject?
    var neurons_max: AnyObject?
    var folds: AnyObject?
    var candidates: AnyObject?
    var evaluated_candidates: AnyObject?
    var unique_labels: AnyObject?
    var iter: AnyObject?
    var foldix: AnyObject?
    var finish_fold_callback: AnyObject?
    var finish_batch_callback: AnyObject?
    
    init () {
        
    }
    
    convenience init(data:[AnyObject?], labels:[AnyObject?], opt:[String:AnyObject?]) {
        var opt = opt || {};
        if(data == null) { data = []; }
        if(labels == null) { labels = []; }
        
        // required inputs
        self.data = data; // store these pointers to data
        self.labels = labels;
        
        // optional inputs
        self.train_ratio = getopt(opt, "train_ratio", 0.7);
        self.num_folds = getopt(opt, "num_folds", 10);
        self.num_candidates = getopt(opt, "num_candidates", 50); // we evaluate several in parallel
        // how many epochs of data to train every network? for every fold?
        // higher values mean higher accuracy in final results, but more expensive
        self.num_epochs = getopt(opt, "num_epochs", 50);
        // number of best models to average during prediction. Usually higher = better
        self.ensemble_size = getopt(opt, "ensemble_size", 10);
        
        // candidate parameters
        self.batch_size_min = getopt(opt, "batch_size_min", 10);
        self.batch_size_max = getopt(opt, "batch_size_max", 300);
        self.l2_decay_min = getopt(opt, "l2_decay_min", -4);
        self.l2_decay_max = getopt(opt, "l2_decay_max", 2);
        self.learning_rate_min = getopt(opt, "learning_rate_min", -4);
        self.learning_rate_max = getopt(opt, "learning_rate_max", 0);
        self.momentum_min = getopt(opt, "momentum_min", 0.9);
        self.momentum_max = getopt(opt, "momentum_max", 0.9);
        self.neurons_min = getopt(opt, "neurons_min", 5);
        self.neurons_max = getopt(opt, "neurons_max", 30);
        
        // computed
        self.folds = []; // data fold indices, gets filled by sampleFolds()
        self.candidates = []; // candidate networks that are being currently evaluated
        self.evaluated_candidates = []; // history of all candidates that were fully evaluated on all folds
        self.unique_labels = arrUnique(labels);
        self.iter = 0; // iteration counter, goes from 0 -> num_epochs * num_training_data
        self.foldix = 0; // index of active fold
        
        // callbacks
        self.finish_fold_callback = null;
        self.finish_batch_callback = null;
        
        // initializations
        if(self.data.length > 0) {
            self.sampleFolds();
            self.sampleCandidates();
        }
    }
    
    // sets self.folds to a sampling of self.num_folds folds
    func sampleFolds() -> () {
        var N = self.data.length;
        var num_train = Math.floor(self.train_ratio * N);
        self.folds = []; // flush folds, if any
        for(var i=0;i<self.num_folds;i++) {
            var p = randperm(N);
            self.folds.push({train_ix: p.slice(0, num_train), test_ix: p.slice(num_train, N)});
        }
    }
    
    // returns a random candidate network
    func sampleCandidate() -> () {
        var input_depth = self.data[0].w.length;
        var num_classes = self.unique_labels.length;
        
        // sample network topology and hyperparameters
        var layer_defs = [];
        layer_defs.push({type:"input", out_sx:1, out_sy:1, out_depth: input_depth});
        var nl = weightedSample([0,1,2,3], [0.2, 0.3, 0.3, 0.2]); // prefer nets with 1,2 hidden layers
        for(var q=0;q<nl;q++) {
            var ni = randi(self.neurons_min, self.neurons_max);
            var act = ["tanh","maxout","relu"][randi(0,3)];
            if(randf(0,1)<0.5) {
                var dp = Math.random();
                layer_defs.push({type:"fc", num_neurons: ni, activation: act, drop_prob: dp});
            } else {
                layer_defs.push({type:"fc", num_neurons: ni, activation: act});
            }
        }
        layer_defs.push({type:"softmax", num_classes: num_classes});
        var net = Net();
        net.makeLayers(layer_defs);
        
        // sample training hyperparameters
        var bs = randi(self.batch_size_min, self.batch_size_max); // batch size
        var l2 = Math.pow(10, randf(self.l2_decay_min, self.l2_decay_max)); // l2 weight decay
        var lr = Math.pow(10, randf(self.learning_rate_min, self.learning_rate_max)); // learning rate
        var mom = randf(self.momentum_min, self.momentum_max); // momentum. Lets just use 0.9, works okay usually ;p
        var tp = randf(0,1); // trainer type
        var trainer_def;
        if(tp<0.33) {
            trainer_def = {method:"adadelta", batch_size:bs, l2_decay:l2};
        } else if(tp<0.66) {
            trainer_def = {method:"adagrad", learning_rate: lr, batch_size:bs, l2_decay:l2};
        } else {
            trainer_def = {method:"sgd", learning_rate: lr, momentum: mom, batch_size:bs, l2_decay:l2};
        }
        
        var trainer = Trainer(net, trainer_def);
        
        var cand = {};
        cand.acc = [];
        cand.accv = 0; // this will maintained as sum(acc) for convenience
        cand.layer_defs = layer_defs;
        cand.trainer_def = trainer_def;
        cand.net = net;
        cand.trainer = trainer;
        return cand;
    }
    
    // sets self.candidates with self.num_candidates candidate nets
    func sampleCandidates() -> () {
        self.candidates = []; // flush, if any
        for(var i=0;i<self.num_candidates;i++) {
            var cand = self.sampleCandidate();
            self.candidates.push(cand);
        }
    }
    
    func step() -> () {
        
        // run an example through current candidate
        self.iter++;
        
        // step all candidates on a random data point
        var fold = self.folds[self.foldix]; // active fold
        var dataix = fold.train_ix[randi(0, fold.train_ix.length)];
        for(var k=0;k<self.candidates.length;k++) {
            var x = self.data[dataix];
            var l = self.labels[dataix];
            self.candidates[k].trainer.train(x, l);
        }
        
        // process consequences: sample new folds, or candidates
        var lastiter = self.num_epochs * fold.train_ix.length;
        if(self.iter >= lastiter) {
            // finished evaluation of this fold. Get final validation
            // accuracies, record them, and go on to next fold.
            var val_acc = self.evalValErrors();
            for(var k=0;k<self.candidates.length;k++) {
                var c = self.candidates[k];
                c.acc.push(val_acc[k]);
                c.accv += val_acc[k];
            }
            self.iter = 0; // reset step number
            self.foldix++; // increment fold
            
            if(self.finish_fold_callback !== null) {
                self.finish_fold_callback();
            }
            
            if(self.foldix >= self.folds.length) {
                // we finished all folds as well! Record these candidates
                // and sample new ones to evaluate.
                for(var k=0;k<self.candidates.length;k++) {
                    self.evaluated_candidates.push(self.candidates[k]);
                }
                // sort evaluated candidates according to accuracy achieved
                self.evaluated_candidates.sort(function(a, b) {
                    return (a.accv / a.acc.length)
                        > (b.accv / b.acc.length)
                        ? -1 : 1;
                    });
                // and clip only to the top few ones (lets place limit at 3*ensemble_size)
                // otherwise there are concerns with keeping these all in memory
                // if MagicNet is being evaluated for a very long time
                if(self.evaluated_candidates.length > 3 * self.ensemble_size) {
                    self.evaluated_candidates = self.evaluated_candidates.slice(0, 3 * self.ensemble_size);
                }
                if(self.finish_batch_callback !== null) {
                    self.finish_batch_callback();
                }
                self.sampleCandidates(); // begin with new candidates
                self.foldix = 0; // reset this
            } else {
                // we will go on to another fold. reset all candidates nets
                for(var k=0;k<self.candidates.length;k++) {
                    var c = self.candidates[k];
                    var net = Net();
                    net.makeLayers(c.layer_defs);
                    var trainer = Trainer(net, c.trainer_def);
                    c.net = net;
                    c.trainer = trainer;
                }
            }
        }
    }
    
    func evalValErrors() -> () {
        // evaluate candidates on validation data and return performance of current networks
        // as simple list
        var vals = [];
        var fold = self.folds[self.foldix]; // active fold
        for(var k=0;k<self.candidates.length;k++) {
            var net = self.candidates[k].net;
            var v = 0.0;
            for(var q=0;q<fold.test_ix.length;q++) {
                var x = self.data[fold.test_ix[q]];
                var l = self.labels[fold.test_ix[q]];
                net.forward(x);
                var yhat = net.getPrediction();
                v += (yhat === l ? 1.0 : 0.0); // 0 1 loss
            }
            v /= fold.test_ix.length; // normalize
            vals.push(v);
        }
        return vals;
    }
    
    // returns prediction scores for given test data point, as Vol
    // uses an averaged prediction from the best ensemble_size models
    // x is a Vol.
    func predict_soft(data) -> () {
        // forward prop the best networks
        // and accumulate probabilities at last layer into a an output Vol
        
        var eval_candidates = [];
        var nv = 0;
        if(self.evaluated_candidates.length === 0) {
            // not sure what to do here, first batch of nets hasnt evaluated yet
            // lets just predict with current candidates.
            nv = self.candidates.length;
            eval_candidates = self.candidates;
        } else {
            // forward prop the best networks from evaluated_candidates
            nv = Math.min(self.ensemble_size, self.evaluated_candidates.length);
            eval_candidates = self.evaluated_candidates
        }
        
        // forward nets of all candidates and average the predictions
        var xout, n;
        for(var j=0;j<nv;j++) {
            var net = eval_candidates[j].net;
            var x = net.forward(data);
            if(j===0) {
                xout = x;
                n = x.w.length;
            } else {
                // add it on
                for(var d=0;d<n;d++) {
                    xout.w[d] += x.w[d];
                }
            }
        }
        // produce average
        for(var d=0;d<n;d++) {
            xout.w[d] /= nv;
        }
        return xout;
    }
    
    func predict(data) -> () {
        var xout = self.predict_soft(data);
        if(xout.w.length !== 0) {
            var stats = maxmin(xout.w);
            var predicted_label = stats.maxi;
        } else {
            var predicted_label = -1; // error out
        }
        return predicted_label;
        
    }
    
    func toJSON() -> () {
        // dump the top ensemble_size networks as a list
        var nv = Math.min(self.ensemble_size, self.evaluated_candidates.length);
        var json = {};
        json.nets = [];
        for(var i=0;i<nv;i++) {
            json.nets.push(self.evaluated_candidates[i].net.toJSON());
        }
        return json;
    }
    
    func fromJSON(json) -> () {
        self.ensemble_size = json.nets.length;
        self.evaluated_candidates = [];
        for(var i=0;i<self.ensemble_size;i++) {
            var net = Net();
            net.fromJSON(json.nets[i]);
            var dummy_candidate = {};
            dummy_candidate.net = net;
            self.evaluated_candidates.push(dummy_candidate);
        }
    }
    
    // callback functions
    // called when a fold is finished, while evaluating a batch
    func onFinishFold(f) -> () { self.finish_fold_callback = f; }
    // called when a batch of candidates has finished evaluating
    func onFinishBatch(f) -> () { self.finish_batch_callback = f; }
    
}

