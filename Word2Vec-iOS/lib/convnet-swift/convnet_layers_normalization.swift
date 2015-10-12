
  var Vol = global.Vol; // convenience
  
  // a bit experimental layer for now. I think it works but I'm not 100%
  // the gradient check is a bit funky. I'll look into this a bit later.
  // Local Response Normalization in window, along depths of volumes
  var LocalResponseNormalizationLayer = function(opt) {
    var opt = opt || {};

    // required
    self.k = opt.k;
    self.n = opt.n;
    self.alpha = opt.alpha;
    self.beta = opt.beta;

    // computed
    self.out_sx = opt.in_sx;
    self.out_sy = opt.in_sy;
    self.out_depth = opt.in_depth;
    self.layer_type = 'lrn';

    // checks
    if(self.n%2 === 0) { console.log('WARNING n should be odd for LRN layer'); }
  }
  LocalResponseNormalizationLayer.prototype = {
    func forward(V, is_training) -> () {
      self.in_act = V;

      var A = V.cloneAndZero();
      self.S_cache_ = V.cloneAndZero();
      var n2 = Math.floor(self.n/2);
      for(var x=0;x<V.sx;x++) {
        for(var y=0;y<V.sy;y++) {
          for(var i=0;i<V.depth;i++) {

            var ai = V.get(x,y,i);

            // normalize in a window of size n
            var den = 0.0;
            for(var j=Math.max(0,i-n2);j<=Math.min(i+n2,V.depth-1);j++) {
              var aa = V.get(x,y,j);
              den += aa*aa;
            }
            den *= self.alpha / self.n;
            den += self.k;
            self.S_cache_.set(x,y,i,den); // will be useful for backprop
            den = Math.pow(den, self.beta);
            A.set(x,y,i,ai/den);
          }
        }
      }

      self.out_act = A;
      return self.out_act; // dummy identity function for now
    },
    func backward() -> () { 
      // evaluate gradient wrt data
      var V = self.in_act; // we need to set dw of this
      V.dw = global.zeros(V.w.length); // zero out gradient wrt data
      var A = self.out_act; // computed in forward pass 

      var n2 = Math.floor(self.n/2);
      for(var x=0;x<V.sx;x++) {
        for(var y=0;y<V.sy;y++) {
          for(var i=0;i<V.depth;i++) {

            var chain_grad = self.out_act.get_grad(x,y,i);
            var S = self.S_cache_.get(x,y,i);
            var SB = Math.pow(S, self.beta);
            var SB2 = SB*SB;

            // normalize in a window of size n
            for(var j=Math.max(0,i-n2);j<=Math.min(i+n2,V.depth-1);j++) {              
              var aj = V.get(x,y,j); 
              var g = -aj*self.beta*Math.pow(S,self.beta-1)*self.alpha/self.n*2*aj;
              if(j===i) g+= SB;
              g /= SB2;
              g *= chain_grad;
              V.add_grad(x,y,j,g);
            }

          }
        }
      }
    },
    func getParamsAndGrads() -> () { return []; },
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
    },
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
  

  global.LocalResponseNormalizationLayer = LocalResponseNormalizationLayer;

