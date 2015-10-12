
class Trainer {
    
    init(){
        
    }
    
    convenience init(net, options) {
        
        self.net = net;
        
        var options = options || {};
        self.learning_rate = options.learning_rate != null ? options.learning_rate : 0.01;
        self.l1_decay = options.l1_decay != null ? options.l1_decay : 0.0;
        self.l2_decay = options.l2_decay != null ? options.l2_decay : 0.0;
        self.batch_size = options.batch_size != null ? options.batch_size : 1;
        self.method = options.method != null ? options.method : "sgd"; // sgd/adam/adagrad/adadelta/windowgrad/netsterov
        
        self.momentum = options.momentum != null ? options.momentum : 0.9;
        self.ro = options.ro != null ? options.ro : 0.95; // used in adadelta
        self.eps = options.eps != null ? options.eps : 1e-8; // used in adam or adadelta
        self.beta1 = options.beta1 != null ? options.beta1 : 0.9; // used in adam
        self.beta2 = options.beta2 != null ? options.beta2 : 0.999; // used in adam
        
        self.k = 0; // iteration counter
        self.gsum = []; // last iteration gradients (used for momentum calculations)
        self.xsum = []; // used in adam or adadelta
        
        // check if regression is expected
        if(self.net.layers[self.net.layers.length - 1].layer_type === "regression")
        self.regression = true;
        else
        self.regression = false;
    }
    
    func train(x, y) -> () {
        
        var start = Date().getTime();
        self.net.forward(x, true); // also set the flag that lets the net know we're just training
        var end = Date().getTime();
        var fwd_time = end - start;
        
        var start = Date().getTime();
        var cost_loss = self.net.backward(y);
        var l2_decay_loss = 0.0;
        var l1_decay_loss = 0.0;
        var end = Date().getTime();
        var bwd_time = end - start;
        
        if(self.regression && y.constructor !== Array)
        console.log("Warning: a regression net requires an array as training output vector.");
        
        self.k++;
        if(self.k % self.batch_size === 0) {
            
            var pglist = self.net.getParamsAndGrads();
            
            // initialize lists for accumulators. Will only be done once on first iteration
            if(self.gsum.length === 0 && (self.method !== "sgd" || self.momentum > 0.0)) {
                // only vanilla sgd doesnt need either lists
                // momentum needs gsum
                // adagrad needs gsum
                // adam and adadelta needs gsum and xsum
                for(var i=0;i<pglist.length;i++) {
                    self.gsum.push(global.zeros(pglist[i].params.length));
                    if(self.method === "adam" || self.method === "adadelta") {
                        self.xsum.push(global.zeros(pglist[i].params.length));
                    } else {
                        self.xsum.push([]); // conserve memory
                    }
                }
            }
            
            // perform an update for all sets of weights
            for(var i=0;i<pglist.length;i++) {
                var pg = pglist[i]; // param, gradient, other options in future (custom learning rate etc)
                var p = pg.params;
                var g = pg.grads;
                
                // learning rate for some parameters.
                var l2_decay_mul = pg.l2_decay_mul != null ? pg.l2_decay_mul : 1.0;
                var l1_decay_mul = pg.l1_decay_mul != null ? pg.l1_decay_mul : 1.0;
                var l2_decay = self.l2_decay * l2_decay_mul;
                var l1_decay = self.l1_decay * l1_decay_mul;
                
                var plen = p.length;
                for(var j=0;j<plen;j++) {
                    l2_decay_loss += l2_decay*p[j]*p[j]/2; // accumulate weight decay loss
                    l1_decay_loss += l1_decay*Math.abs(p[j]);
                    var l1grad = l1_decay * (p[j] > 0 ? 1 : -1);
                    var l2grad = l2_decay * (p[j]);
                    
                    var gij = (l2grad + l1grad + g[j]) / self.batch_size; // raw batch gradient
                    
                    var gsumi = self.gsum[i];
                    var xsumi = self.xsum[i];
                    if(self.method === "adam") {
                        // adam update
                        gsumi[j] = gsumi[j] * self.beta1 + (1- self.beta1) * gij; // update biased first moment estimate
                        xsumi[j] = xsumi[j] * self.beta2 + (1-self.beta2) * gij * gij; // update biased second moment estimate
                        var biasCorr1 = gsumi[j] * (1 - Math.pow(self.beta1, self.k)); // correct bias first moment estimate
                        var biasCorr2 = xsumi[j] * (1 - Math.pow(self.beta2, self.k)); // correct bias second moment estimate
                        var dx =  - self.learning_rate * biasCorr1 / (Math.sqrt(biasCorr2) + self.eps);
                        p[j] += dx;
                    } else if(self.method === "adagrad") {
                        // adagrad update
                        gsumi[j] = gsumi[j] + gij * gij;
                        var dx = - self.learning_rate / Math.sqrt(gsumi[j] + self.eps) * gij;
                        p[j] += dx;
                    } else if(self.method === "windowgrad") {
                        // this is adagrad but with a moving window weighted average
                        // so the gradient is not accumulated over the entire history of the run.
                        // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
                        gsumi[j] = self.ro * gsumi[j] + (1-self.ro) * gij * gij;
                        var dx = - self.learning_rate / Math.sqrt(gsumi[j] + self.eps) * gij; // eps added for better conditioning
                        p[j] += dx;
                    } else if(self.method === "adadelta") {
                        gsumi[j] = self.ro * gsumi[j] + (1-self.ro) * gij * gij;
                        var dx = - Math.sqrt((xsumi[j] + self.eps)/(gsumi[j] + self.eps)) * gij;
                        xsumi[j] = self.ro * xsumi[j] + (1-self.ro) * dx * dx; // yes, xsum lags behind gsum by 1.
                        p[j] += dx;
                    } else if(self.method === "nesterov") {
                        var dx = gsumi[j];
                        gsumi[j] = gsumi[j] * self.momentum + self.learning_rate * gij;
                        dx = self.momentum * dx - (1.0 + self.momentum) * gsumi[j];
                        p[j] += dx;
                    } else {
                        // assume SGD
                        if(self.momentum > 0.0) {
                            // momentum update
                            var dx = self.momentum * gsumi[j] - self.learning_rate * gij; // step
                            gsumi[j] = dx; // back this up for next iteration of momentum
                            p[j] += dx; // apply corrected gradient
                        } else {
                            // vanilla sgd
                            p[j] +=  - self.learning_rate * gij;
                        }
                    }
                    g[j] = 0.0; // zero out gradient so that we can begin accumulating anew
                }
            }
        }
        
        // appending softmax_loss for backwards compatibility, but from now on we will always use cost_loss
        // in future, TODO: have to completely redo the way loss is done around the network as currently 
        // loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
        // and it should all be computed correctly and automatically. 
        return {fwd_time: fwd_time, bwd_time: bwd_time, 
            l2_decay_loss: l2_decay_loss, l1_decay_loss: l1_decay_loss,
            cost_loss: cost_loss, softmax_loss: cost_loss, 
            loss: cost_loss + l1_decay_loss + l2_decay_loss}
    }
}

//  global.Trainer = Trainer;
//  global.SGDTrainer = Trainer; // backwards compatibility
//

