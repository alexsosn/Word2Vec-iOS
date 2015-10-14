
// a bit experimental layer for now. I think it works but I'm not 100%
// the gradient check is a bit funky. I'll look into this a bit later.
// Local Response Normalization in window, along depths of volumes
class LocalResponseNormalizationLayer: Layer {
    init(){}
    convenience init(opt: Options) {
        var opt = opt || {};
        
        // required
        self.k = opt["k"];
        self.n = opt["n"];
        self.alpha = opt["alpha"];
        self.beta = opt["beta"];
        
        // computed
        self.out_sx = opt["in_sx"];
        self.out_sy = opt["in_sy"];
        self.out_depth = opt["in_depth"];
        self.layer_type = "lrn";
        
        // checks
        if(self.n%2 === 0) { print("WARNING n should be odd for LRN layer"); }
    }
    
    func forward(V: Vol, is_training: Bool) -> () {
        self.in_act = V;
        
        var A = V.cloneAndZero();
        self.S_cache_ = V.cloneAndZero();
        var n2 = floor(self.n/2);
        for x in 0 ..< V.sx { {
            for y in 0 ..< V.sy { {
                for i in 0 ..< V.depth { {
                    
                    var ai = V.get(x,y,i);
                    
                    // normalize in a window of size n
                    var den = 0.0;
                    for(var j=max(0,i-n2);j<=min(i+n2,V.depth-1);j++) {
                        var aa = V.get(x,y,j);
                        den += aa*aa;
                    }
                    den *= self.alpha / self.n;
                    den += self.k;
                    self.S_cache_.set(x,y,i,den); // will be useful for backprop
                    den = pow(den, self.beta);
                    A.set(x,y,i,ai/den);
                }
            }
        }
        
        self.out_act = A;
        return self.out_act; // dummy identity function for now
    }
    
    func backward() -> () {
        // evaluate gradient wrt data
        var V = self.in_act; // we need to set dw of this
        V.dw = zeros(V.w.length); // zero out gradient wrt data
        var A = self.out_act; // computed in forward pass
        
        var n2 = floor(self.n/2);
        for x in 0 ..< V.sx { {
            for y in 0 ..< V.sy { {
                for i in 0 ..< V.depth { {
                    
                    var chain_grad = self.out_act.get_grad(x,y,i);
                    var S = self.S_cache_.get(x,y,i);
                    var SB = pow(S, self.beta);
                    var SB2 = SB*SB;
                    
                    // normalize in a window of size n
                    for(var j=max(0,i-n2);j<=min(i+n2,V.depth-1);j++) {
                        var aj = V.get(x,y,j);
                        var g = -aj*self.beta*pow(S,self.beta-1)*self.alpha/self.n*2*aj;
                        if(j===i) { g+= SB; }
                        g /= SB2;
                        g *= chain_grad;
                        V.add_grad(x,y,j,g);
                    }
                    
                }
            }
        }
    }
    
    func getParamsAndGrads() -> () { return []; }
    
    func toJSON() -> () {
        var json = {};
        json.k = self.k;
        json.n = self.n;
        json.alpha = self.alpha; // normalize by size
        json.beta = self.beta;
        json.out_sx = self.out_sx;
        json.out_sy = self.out_sy;
        json.out_depth = self.out_depth;
        json.layer_type = self.layer_type;
        return json;
    }
    
    func fromJSON(json) -> () {
        self.k = json.k;
        self.n = json.n;
        self.alpha = json.alpha; // normalize by size
        self.beta = json.beta;
        self.out_sx = json.out_sx; 
        self.out_sy = json.out_sy;
        self.out_depth = json.out_depth;
        self.layer_type = json.layer_type;
    }
}

