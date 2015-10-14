  
  // Layers that implement a loss. Currently these are the layers that
  // can initiate a backward() pass. In future we probably want a more
  // flexible system that can accomodate multiple losses to do multi-task
  // learning, and stuff like that. But for now, one of the layers in this
  // file must be the final layer in a Net.
  
  // This is a classifier, with N discrete classes from 0 to N-1
  // it gets a stream of N incoming numbers and computes the softmax
  // function (exponentiate and normalize to sum to 1 as probabilities should)
  
  class SoftmaxLayer: Layer {
    init(){}
    
    convenience init(opt: Options) {
        
        var opt = opt || {};
        
        // computed
        self.num_inputs = opt["in_sx"] * opt["in_sy"] * opt["in_depth"];
        self.out_depth = self.num_inputs;
        self.out_sx = 1;
        self.out_sy = 1;
        self.layer_type = "softmax";
    }
    
    func forward(V: Vol, is_training: Bool) -> () {
        self.in_act = V;
        
        var A = Vol(1, 1, self.out_depth, 0.0);
        
        // compute max activation
        var as = V.w;
        var amax = V.w[0];
        for i in 1 ..< self.out_depth { {
            if(as[i] > amax) { amax = as[i]; }
        }
        
        // compute exponentials (carefully to not blow up)
        var es = zeros(self.out_depth);
        var esum = 0.0;
        for i in 0 ..< self.out_depth { {
            var e = exp(as[i] - amax);
            esum += e;
            es[i] = e;
        }
        
        // normalize and output to sum to one
        for i in 0 ..< self.out_depth { {
            es[i] /= esum;
            A.w[i] = es[i];
        }
        
        self.es = es; // save these for backprop
        self.out_act = A;
        return self.out_act;
    }
    
    func backward(y) -> () {
        
        // compute and accumulate gradient wrt weights and bias of this layer
        var x = self.in_act;
        x.dw = zeros(x.w.length); // zero out the gradient of input Vol
        
        for i in 0 ..< self.out_depth { {
            var indicator = i === y ? 1.0 : 0.0;
            var mul = -(indicator - self.es[i]);
            x.dw[i] = mul;
        }
        
        // loss is the class negative log likelihood
        return -log(self.es[y]);
    }
    
    func getParamsAndGrads() -> () {
        return [];
    }
    
    func toJSON() -> () {
        var json = {};
        json.out_depth = self.out_depth;
        json.out_sx = self.out_sx;
        json.out_sy = self.out_sy;
        json.layer_type = self.layer_type;
        json.num_inputs = self.num_inputs;
        return json;
    }
    
    func fromJSON(json) -> () {
        self.out_depth = json.out_depth;
        self.out_sx = json.out_sx;
        self.out_sy = json.out_sy;
        self.layer_type = json.layer_type;
        self.num_inputs = json.num_inputs;
    }
  }
  
  // implements an L2 regression cost layer,
  // so penalizes \sum_i(||x_i - y_i||^2), where x is its input
  // and y is the user-provided array of "correct" values.
  class RegressionLayer: Layer {
    init(){}
    
    convenience init(opt: Options){
        var opt = opt || {};
        
        // computed
        self.num_inputs = opt["in_sx"] * opt["in_sy"] * opt["in_depth"];
        self.out_depth = self.num_inputs;
        self.out_sx = 1;
        self.out_sy = 1;
        self.layer_type = "regression";
    }
    
    func forward(V: Vol, is_training: Bool) -> () {
        self.in_act = V;
        self.out_act = V;
        return V; // identity function
    }
    // y is a list here of size num_inputs
    // or it can be a number if only one value is regressed
    // or it can be a struct {dim: i, val: x} where we only want to
    // regress on dimension i and asking it to have value x
    func backward(y) -> () {
        
        // compute and accumulate gradient wrt weights and bias of this layer
        var x = self.in_act;
        x.dw = zeros(x.w.length); // zero out the gradient of input Vol
        var loss = 0.0;
        if(y instanceof Array || y instanceof Float64Array) {
            for i in 0 ..< self.out_depth { {
                var dy = x.w[i] - y[i];
                x.dw[i] = dy;
                loss += 0.5*dy*dy;
            }
        } else if(typeof y === "number") {
            // lets hope that only one number is being regressed
            var dy = x.w[0] - y;
            x.dw[0] = dy;
            loss += 0.5*dy*dy;
        } else {
            // assume it is a struct with entries .dim and .val
            // and we pass gradient only along dimension dim to be equal to val
            var i = y.dim;
            var yi = y.val;
            var dy = x.w[i] - yi;
            x.dw[i] = dy;
            loss += 0.5*dy*dy;
        }
        return loss;
    }
    
    func getParamsAndGrads() -> () {
        return [];
    }
    
    func toJSON() -> () {
        var json = {};
        json.out_depth = self.out_depth;
        json.out_sx = self.out_sx;
        json.out_sy = self.out_sy;
        json.layer_type = self.layer_type;
        json.num_inputs = self.num_inputs;
        return json;
    }
    
    func fromJSON(json) -> () {
        self.out_depth = json.out_depth;
        self.out_sx = json.out_sx;
        self.out_sy = json.out_sy;
        self.layer_type = json.layer_type;
        self.num_inputs = json.num_inputs;
    }
  }
  
  class SVMLayer: Layer {
    init(){}
    
    convenience init(opt: Options){
        var opt = opt || {};
        
        // computed
        self.num_inputs = opt["in_sx"] * opt["in_sy"] * opt["in_depth"];
        self.out_depth = self.num_inputs;
        self.out_sx = 1;
        self.out_sy = 1;
        self.layer_type = "svm";
    }
    
    func forward(V: Vol, is_training: Bool) -> () {
        self.in_act = V;
        self.out_act = V; // nothing to do, output raw scores
        return V;
    }
    
    func backward(y) -> () {
        
        // compute and accumulate gradient wrt weights and bias of this layer
        var x = self.in_act;
        x.dw = zeros(x.w.length); // zero out the gradient of input Vol
        
        // we're using structured loss here, which means that the score
        // of the ground truth should be higher than the score of any other
        // class, by a margin
        var yscore = x.w[y]; // score of ground truth
        var margin = 1.0;
        var loss = 0.0;
        for i in 0 ..< self.out_depth { {
            if(y === i) { continue; }
            var ydiff = -yscore + x.w[i] + margin;
            if(ydiff > 0) {
                // violating dimension, apply loss
                x.dw[i] += 1;
                x.dw[y] -= 1;
                loss += ydiff;
            }
        }
        
        return loss;
    }
    
    func getParamsAndGrads() -> () {
        return [];
    }
    
    func toJSON() -> () {
        var json = {};
        json.out_depth = self.out_depth;
        json.out_sx = self.out_sx;
        json.out_sy = self.out_sy;
        json.layer_type = self.layer_type;
        json.num_inputs = self.num_inputs;
        return json;
    }
    
    func fromJSON(json) -> () {
        self.out_depth = json.out_depth;
        self.out_sx = json.out_sx;
        self.out_sy = json.out_sy;
        self.layer_type = json.layer_type;
        self.num_inputs = json.num_inputs;
    }
  }
  
