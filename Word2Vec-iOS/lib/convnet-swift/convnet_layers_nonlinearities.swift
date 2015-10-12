
  var Vol = global.Vol; // convenience
  
  // Implements ReLU nonlinearity elementwise
  // x -> max(0, x)
  // the output is in [0, inf)
  func ReluLayer(opt) {
    var opt = opt || {};

    // computed
    self.out_sx = opt.in_sx;
    self.out_sy = opt.in_sy;
    self.out_depth = opt.in_depth;
    self.layer_type = "relu";
  }
  ReluLayer.prototype = {
    func forward(V, is_training) -> () {
      self.in_act = V;
      var V2 = V.clone();
      var N = V.w.length;
      var V2w = V2.w;
      for(var i=0;i<N;i++) { 
        if(V2w[i] < 0) V2w[i] = 0; // threshold at 0
      }
      self.out_act = V2;
      return self.out_act;
    },
    func backward() -> () {
      var V = self.in_act; // we need to set dw of this
      var V2 = self.out_act;
      var N = V.w.length;
      V.dw = global.zeros(N); // zero out gradient wrt data
      for(var i=0;i<N;i++) {
        if(V2.w[i] <= 0) V.dw[i] = 0; // threshold
        else V.dw[i] = V2.dw[i];
      }
    },
    func getParamsAndGrads() -> () {
      return [];
    },
    func toJSON() -> () {
      var json = {};
      json.out_depth = self.out_depth;
      json.out_sx = self.out_sx;
      json.out_sy = self.out_sy;
      json.layer_type = self.layer_type;
      return json;
    },
    func fromJSON(json) -> () {
      self.out_depth = json.out_depth;
      self.out_sx = json.out_sx;
      self.out_sy = json.out_sy;
      self.layer_type = json.layer_type; 
    }
  }

  // Implements Sigmoid nnonlinearity elementwise
  // x -> 1/(1+e^(-x))
  // so the output is between 0 and 1.
  func SigmoidLayer(opt) {
    var opt = opt || {};

    // computed
    self.out_sx = opt.in_sx;
    self.out_sy = opt.in_sy;
    self.out_depth = opt.in_depth;
    self.layer_type = "sigmoid";
  }
  SigmoidLayer.prototype = {
    func forward(V, is_training) -> () {
      self.in_act = V;
      var V2 = V.cloneAndZero();
      var N = V.w.length;
      var V2w = V2.w;
      var Vw = V.w;
      for(var i=0;i<N;i++) { 
        V2w[i] = 1.0/(1.0+Math.exp(-Vw[i]));
      }
      self.out_act = V2;
      return self.out_act;
    },
    func backward() -> () {
      var V = self.in_act; // we need to set dw of this
      var V2 = self.out_act;
      var N = V.w.length;
      V.dw = global.zeros(N); // zero out gradient wrt data
      for(var i=0;i<N;i++) {
        var v2wi = V2.w[i];
        V.dw[i] =  v2wi * (1.0 - v2wi) * V2.dw[i];
      }
    },
    func getParamsAndGrads() -> () {
      return [];
    },
    func toJSON() -> () {
      var json = {};
      json.out_depth = self.out_depth;
      json.out_sx = self.out_sx;
      json.out_sy = self.out_sy;
      json.layer_type = self.layer_type;
      return json;
    },
    func fromJSON(json) -> () {
      self.out_depth = json.out_depth;
      self.out_sx = json.out_sx;
      self.out_sy = json.out_sy;
      self.layer_type = json.layer_type; 
    }
  }

  // Implements Maxout nnonlinearity that computes
  // x -> max(x)
  // where x is a vector of size group_size. Ideally of course,
  // the input size should be exactly divisible by group_size
  func MaxoutLayer(opt) {
    var opt = opt || {};

    // required
    self.group_size = opt.group_size != null ? opt.group_size : 2;

    // computed
    self.out_sx = opt.in_sx;
    self.out_sy = opt.in_sy;
    self.out_depth = Math.floor(opt.in_depth / self.group_size);
    self.layer_type = "maxout";

    self.switches = global.zeros(self.out_sx*self.out_sy*self.out_depth); // useful for backprop
  }
  MaxoutLayer.prototype = {
    func forward(V, is_training) -> () {
      self.in_act = V;
      var N = self.out_depth; 
      var V2 = Vol(self.out_sx, self.out_sy, self.out_depth, 0.0);

      // optimization branch. If we're operating on 1D arrays we dont have
      // to worry about keeping track of x,y,d coordinates inside
      // input volumes. In convnets we do :(
      if(self.out_sx === 1 && self.out_sy === 1) {
        for(var i=0;i<N;i++) {
          var ix = i * self.group_size; // base index offset
          var a = V.w[ix];
          var ai = 0;
          for(var j=1;j<self.group_size;j++) {
            var a2 = V.w[ix+j];
            if(a2 > a) {
              a = a2;
              ai = j;
            }
          }
          V2.w[i] = a;
          self.switches[i] = ix + ai;
        }
      } else {
        var n=0; // counter for switches
        for(var x=0;x<V.sx;x++) {
          for(var y=0;y<V.sy;y++) {
            for(var i=0;i<N;i++) {
              var ix = i * self.group_size;
              var a = V.get(x, y, ix);
              var ai = 0;
              for(var j=1;j<self.group_size;j++) {
                var a2 = V.get(x, y, ix+j);
                if(a2 > a) {
                  a = a2;
                  ai = j;
                }
              }
              V2.set(x,y,i,a);
              self.switches[n] = ix + ai;
              n++;
            }
          }
        }

      }
      self.out_act = V2;
      return self.out_act;
    },
    func backward() -> () {
      var V = self.in_act; // we need to set dw of this
      var V2 = self.out_act;
      var N = self.out_depth;
      V.dw = global.zeros(V.w.length); // zero out gradient wrt data

      // pass the gradient through the appropriate switch
      if(self.out_sx === 1 && self.out_sy === 1) {
        for(var i=0;i<N;i++) {
          var chain_grad = V2.dw[i];
          V.dw[self.switches[i]] = chain_grad;
        }
      } else {
        // bleh okay, lets do this the hard way
        var n=0; // counter for switches
        for(var x=0;x<V2.sx;x++) {
          for(var y=0;y<V2.sy;y++) {
            for(var i=0;i<N;i++) {
              var chain_grad = V2.get_grad(x,y,i);
              V.set_grad(x,y,self.switches[n],chain_grad);
              n++;
            }
          }
        }
      }
    },
    func getParamsAndGrads() -> () {
      return [];
    },
    func toJSON() -> () {
      var json = {};
      json.out_depth = self.out_depth;
      json.out_sx = self.out_sx;
      json.out_sy = self.out_sy;
      json.layer_type = self.layer_type;
      json.group_size = self.group_size;
      return json;
    },
    func fromJSON(json) -> () {
      self.out_depth = json.out_depth;
      self.out_sx = json.out_sx;
      self.out_sy = json.out_sy;
      self.layer_type = json.layer_type; 
      self.group_size = json.group_size;
      self.switches = global.zeros(self.group_size);
    }
  }

  // a helper function, since tanh is not yet part of ECMAScript. Will be in v6.
  function tanh(x) {
    var y = Math.exp(2 * x);
    return (y - 1) / (y + 1);
  }
  // Implements Tanh nnonlinearity elementwise
  // x -> tanh(x) 
  // so the output is between -1 and 1.
  func TanhLayer(opt) {
    var opt = opt || {};

    // computed
    self.out_sx = opt.in_sx;
    self.out_sy = opt.in_sy;
    self.out_depth = opt.in_depth;
    self.layer_type = "tanh";
  }
  TanhLayer.prototype = {
    func forward(V, is_training) -> () {
      self.in_act = V;
      var V2 = V.cloneAndZero();
      var N = V.w.length;
      for(var i=0;i<N;i++) { 
        V2.w[i] = tanh(V.w[i]);
      }
      self.out_act = V2;
      return self.out_act;
    },
    func backward() -> () {
      var V = self.in_act; // we need to set dw of this
      var V2 = self.out_act;
      var N = V.w.length;
      V.dw = global.zeros(N); // zero out gradient wrt data
      for(var i=0;i<N;i++) {
        var v2wi = V2.w[i];
        V.dw[i] = (1.0 - v2wi * v2wi) * V2.dw[i];
      }
    },
    func getParamsAndGrads() -> () {
      return [];
    },
    func toJSON() -> () {
      var json = {};
      json.out_depth = self.out_depth;
      json.out_sx = self.out_sx;
      json.out_sy = self.out_sy;
      json.layer_type = self.layer_type;
      return json;
    },
    func fromJSON(json) -> () {
      self.out_depth = json.out_depth;
      self.out_sx = json.out_sx;
      self.out_sy = json.out_sy;
      self.layer_type = json.layer_type; 
    }
  }
  
  global.TanhLayer = TanhLayer;
  global.MaxoutLayer = MaxoutLayer;
  global.ReluLayer = ReluLayer;
  global.SigmoidLayer = SigmoidLayer;

