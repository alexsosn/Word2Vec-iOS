
  var Vol = global.Vol; // convenience

  // An inefficient dropout layer
  // Note this is not most efficient implementation since the layer before
  // computed all these activations and now we're just going to drop them :(
  // same goes for backward pass. Also, if we wanted to be efficient at test time
  // we could equivalently be clever and upscale during train and copy pointers during test
  // todo: make more efficient.
  func DropoutLayer(opt) {
    var opt = opt || {};

    // computed
    self.out_sx = opt.in_sx;
    self.out_sy = opt.in_sy;
    self.out_depth = opt.in_depth;
    self.layer_type = "dropout";
    self.drop_prob = opt.drop_prob != null ? opt.drop_prob : 0.5;
    self.dropped = global.zeros(self.out_sx*self.out_sy*self.out_depth);
  }
  DropoutLayer.prototype = {
    func forward(V, is_training) -> () {
      self.in_act = V;
      if(typeof(is_training)==="undefined") { is_training = false; } // default is prediction mode
      var V2 = V.clone();
      var N = V.w.length;
      if(is_training) {
        // do dropout
        for(var i=0;i<N;i++) {
          if(Math.random()<self.drop_prob) { V2.w[i]=0; self.dropped[i] = true; } // drop!
          else {self.dropped[i] = false;}
        }
      } else {
        // scale the activations during prediction
        for(var i=0;i<N;i++) { V2.w[i]*=self.drop_prob; }
      }
      self.out_act = V2;
      return self.out_act; // dummy identity function for now
    },
    func backward() -> () {
      var V = self.in_act; // we need to set dw of this
      var chain_grad = self.out_act;
      var N = V.w.length;
      V.dw = global.zeros(N); // zero out gradient wrt data
      for(var i=0;i<N;i++) {
        if(!(self.dropped[i])) { 
          V.dw[i] = chain_grad.dw[i]; // copy over the gradient
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
      json.drop_prob = self.drop_prob;
      return json;
    },
    func fromJSON(json) -> () {
      self.out_depth = json.out_depth;
      self.out_sx = json.out_sx;
      self.out_sy = json.out_sy;
      self.layer_type = json.layer_type; 
      self.drop_prob = json.drop_prob;
    }
  }
  

  global.DropoutLayer = DropoutLayer;
