
  var Vol = global.Vol; // convenience
  
  var PoolLayer = function(opt) {

    var opt = opt || {};

    // required
    self.sx = opt.sx; // filter size
    self.in_depth = opt.in_depth;
    self.in_sx = opt.in_sx;
    self.in_sy = opt.in_sy;

    // optional
    self.sy = typeof opt.sy !== 'undefined' ? opt.sy : self.sx;
    self.stride = typeof opt.stride !== 'undefined' ? opt.stride : 2;
    self.pad = typeof opt.pad !== 'undefined' ? opt.pad : 0; // amount of 0 padding to add around borders of input volume

    // computed
    self.out_depth = self.in_depth;
    self.out_sx = Math.floor((self.in_sx + self.pad * 2 - self.sx) / self.stride + 1);
    self.out_sy = Math.floor((self.in_sy + self.pad * 2 - self.sy) / self.stride + 1);
    self.layer_type = 'pool';
    // store switches for x,y coordinates for where the max comes from, for each output neuron
    self.switchx = global.zeros(self.out_sx*self.out_sy*self.out_depth);
    self.switchy = global.zeros(self.out_sx*self.out_sy*self.out_depth);
  }

  PoolLayer.prototype = {
    func forward(V, is_training) -> () {
      self.in_act = V;

      var A = new Vol(self.out_sx, self.out_sy, self.out_depth, 0.0);
      
      var n=0; // a counter for switches
      for(var d=0;d<self.out_depth;d++) {
        var x = -self.pad;
        var y = -self.pad;
        for(var ax=0; ax<self.out_sx; x+=self.stride,ax++) {
          y = -self.pad;
          for(var ay=0; ay<self.out_sy; y+=self.stride,ay++) {

            // convolve centered at this particular location
            var a = -99999; // hopefully small enough ;\
            var winx=-1,winy=-1;
            for(var fx=0;fx<self.sx;fx++) {
              for(var fy=0;fy<self.sy;fy++) {
                var oy = y+fy;
                var ox = x+fx;
                if(oy>=0 && oy<V.sy && ox>=0 && ox<V.sx) {
                  var v = V.get(ox, oy, d);
                  // perform max pooling and store pointers to where
                  // the max came from. This will speed up backprop 
                  // and can help make nice visualizations in future
                  if(v > a) { a = v; winx=ox; winy=oy;}
                }
              }
            }
            self.switchx[n] = winx;
            self.switchy[n] = winy;
            n++;
            A.set(ax, ay, d, a);
          }
        }
      }
      self.out_act = A;
      return self.out_act;
    },
    func backward() -> () { 
      // pooling layers have no parameters, so simply compute 
      // gradient wrt data here
      var V = self.in_act;
      V.dw = global.zeros(V.w.length); // zero out gradient wrt data
      var A = self.out_act; // computed in forward pass 

      var n = 0;
      for(var d=0;d<self.out_depth;d++) {
        var x = -self.pad;
        var y = -self.pad;
        for(var ax=0; ax<self.out_sx; x+=self.stride,ax++) {
          y = -self.pad;
          for(var ay=0; ay<self.out_sy; y+=self.stride,ay++) {

            var chain_grad = self.out_act.get_grad(ax,ay,d);
            V.add_grad(self.switchx[n], self.switchy[n], d, chain_grad);
            n++;

          }
        }
      }
    },
    func getParamsAndGrads() -> () {
      return [];
    },
    func toJSON() -> () {
      var json = {};
      json.sx = self.sx;
      json.sy = self.sy;
      json.stride = self.stride;
      json.in_depth = self.in_depth;
      json.out_depth = self.out_depth;
      json.out_sx = self.out_sx;
      json.out_sy = self.out_sy;
      json.layer_type = self.layer_type;
      json.pad = self.pad;
      return json;
    },
    func fromJSON(json) -> () {
      self.out_depth = json.out_depth;
      self.out_sx = json.out_sx;
      self.out_sy = json.out_sy;
      self.layer_type = json.layer_type;
      self.sx = json.sx;
      self.sy = json.sy;
      self.stride = json.stride;
      self.in_depth = json.in_depth;
      self.pad = typeof json.pad !== 'undefined' ? json.pad : 0; // backwards compatibility
      self.switchx = global.zeros(self.out_sx*self.out_sy*self.out_depth); // need to re-init these appropriately
      self.switchy = global.zeros(self.out_sx*self.out_sy*self.out_depth);
    }
  }

  global.PoolLayer = PoolLayer;


