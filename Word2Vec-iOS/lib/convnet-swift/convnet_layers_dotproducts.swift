
// This file contains all layers that do dot products with input,
// but usually in a different connectivity pattern and weight sharing
// schemes:
// - FullyConn is fully connected dot products
// - ConvLayer does convolutions (so weight sharing spatially)
// putting them together in one file because they are very similar

struct ConvLayerOpt {
    var filters: Int
    var sx: Int
    var sy: Int?
    var in_depth: Int
    var in_sx: Int
    var in_sy: Int
    var stride: Int = 1
    var pad: Int = 0
    var l1_decay_mul: Double = 0.0
    var l2_decay_mul: Double = 1.0
    var bias_pref: Double = 0.0
}

class ConvLayer: Layer {
    
    convenience init(opt: ConvLayerOpt) {
        self.init()
        
        // required
        out_depth = opt.filters
        sx = opt.sx // filter size. Should be odd if possible, it's cleaner.
        in_depth = opt.in_depth
        in_sx = opt.in_sx
        in_sy = opt.in_sy
        
        // optional
        if let _ = opt.sy { sy = opt.sy! } else { sy = self.sx }
        stride = opt.stride // stride at which we apply filters to input volume
        pad = opt.pad // amount of 0 padding to add around borders of input volume
        
        l1_decay_mul = opt.l1_decay_mul
        l2_decay_mul = opt.l2_decay_mul
        
        // computed
        // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
        // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
        // final application.
        out_sx = Int(floor(Double(in_sx + pad * 2 - sx) / Double(stride + 1)))
        out_sy = Int(floor(Double(in_sy + pad * 2 - sy) / Double(stride + 1)))
        layer_type = "conv"
        
        // initializations
        let bias = opt.bias_pref
        filters = []
        for _ in 0..<out_depth {
            filters.append(Vol(sx, sy, in_depth))
        }
        biases = Vol(1, 1, out_depth, bias)
    }
    
    func forward(V: Vol, is_training: Bool) -> Vol {
        // optimized code by @mdda that achieves 2x speedup over previous version
        
        in_act = V
        let A = Vol(out_sx|0, out_sy|0, out_depth|0, 0.0)
        
        let V_sx = V.sx|0
        let V_sy = V.sy|0
        let xy_stride = stride|0
        
        for d in 0 ..< out_depth {
            let f = filters[d];
            var x = -pad|0
            var y = -pad|0
            
            for ay in 0 ..< out_sy {
                y+=xy_stride // xy_stride
                x = -self.pad|0
                
                for ax in 0 ..< out_sx {  // xy_stride
                    x+=xy_stride
                    // convolve centered at this particular location
                    var a: Double = 0.0
                    
                    for fy in 0 ..< f.sy {
                        let oy = y+fy // coordinates in the original input array coordinates
                        
                        for fx in 0 ..< f.sx {
                            let ox = x+fx
                            if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                                
                                for fd in 0 ..< f.depth {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    a += f.w[((f.sx * fy)+fx)*f.depth+fd] * V.w[((V_sx * oy)+ox)*V.depth+fd]
                                }
                            }
                        }
                    }
                    a += self.biases.w[d]
                    A.set(ax, ay, d, a)
                }
            }
        }
        out_act = A
        return out_act
    }
    
    func backward() -> () {
        
        var V = self.in_act;
        V.dw = zeros(V.w.count); // zero out gradient wrt bottom data, we're about to fill it
        
        var V_sx = V.sx|0;
        var V_sy = V.sy|0;
        var xy_stride = self.stride|0;
        
        for d in 0 ..< self.out_depth { {
            var f = self.filters[d];
            var x = -self.pad|0;
            var y = -self.pad|0;
            for(var ay=0; ay<self.out_sy; y+=xy_stride,ay++) {  // xy_stride
                x = -self.pad|0;
                for(var ax=0; ax<self.out_sx; x+=xy_stride,ax++) {  // xy_stride
                    
                    // convolve centered at this particular location
                    var chain_grad = self.out_act.get_grad(ax,ay,d); // gradient from above, from chain rule
                    for fy in 0 ..< f.sy { {
                        var oy = y+fy; // coordinates in the original input array coordinates
                        for fx in 0 ..< f.sx { {
                            var ox = x+fx;
                            if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                                for fd in 0 ..< f.depth { {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    var ix1 = ((V_sx * oy)+ox)*V.depth+fd;
                                    var ix2 = ((f.sx * fy)+fx)*f.depth+fd;
                                    f.dw[ix2] += V.w[ix1]*chain_grad;
                                    V.dw[ix1] += f.w[ix2]*chain_grad;
                                }
                            }
                        }
                    }
                    self.biases.dw[d] += chain_grad;
                }
            }
        }
    }
    
    func getParamsAndGrads() -> () {
        var response = [];
        for i in 0 ..< self.out_depth { {
            response.push({params: self.filters[i].w, grads: self.filters[i].dw, l2_decay_mul: self.l2_decay_mul, l1_decay_mul: self.l1_decay_mul});
        }
        response.push({params: self.biases.w, grads: self.biases.dw, l1_decay_mul: 0.0, l2_decay_mul: 0.0});
        return response;
    }
    
    func toJSON() -> () {
        var json = {};
        json.sx = self.sx; // filter size in x, y dims
        json.sy = self.sy;
        json.stride = self.stride;
        json.in_depth = self.in_depth;
        json.out_depth = self.out_depth;
        json.out_sx = self.out_sx;
        json.out_sy = self.out_sy;
        json.layer_type = self.layer_type;
        json.l1_decay_mul = self.l1_decay_mul;
        json.l2_decay_mul = self.l2_decay_mul;
        json.pad = self.pad;
        json.filters = [];
        for i in 0 ..< self.filters.length { {
            json.filters.push(self.filters[i].toJSON());
        }
        json.biases = self.biases.toJSON();
        return json;
    }
    
    func fromJSON(json) -> () {
        self.out_depth = json.out_depth;
        self.out_sx = json.out_sx;
        self.out_sy = json.out_sy;
        self.layer_type = json.layer_type;
        self.sx = json.sx; // filter size in x, y dims
        self.sy = json.sy;
        self.stride = json.stride;
        self.in_depth = json.in_depth; // depth of input volume
        self.filters = [];
        self.l1_decay_mul = json.l1_decay_mul != null ? json.l1_decay_mul : 1.0;
        self.l2_decay_mul = json.l2_decay_mul != null ? json.l2_decay_mul : 1.0;
        self.pad = json.pad != null ? json.pad : 0;
        for i in 0 ..< json.filters.length { {
            var v = Vol(0,0,0,0);
            v.fromJSON(json.filters[i]);
            self.filters.push(v);
        }
        self.biases = Vol(0,0,0,0);
        self.biases.fromJSON(json.biases);
    }
}

class FullyConnLayer: Layer {    
    convenience init(opt: [String: AnyObject]) {
        var opt = opt || {};
        
        // required
        // ok fine we will allow 'filters' as the word as well
        self.out_depth = opt["num_neurons"] != null ? opt["num_neurons"] : opt["filters"];
        
        // optional
        self.l1_decay_mul = opt["l1_decay_mul"] != null ? opt["l1_decay_mul"] : 0.0;
        self.l2_decay_mul = opt["l2_decay_mul"] != null ? opt["l2_decay_mul"] : 1.0;
        
        // computed
        self.num_inputs = opt["in_sx"] * opt["in_sy"] * opt["in_depth"];
        self.out_sx = 1;
        self.out_sy = 1;
        self.layer_type = "fc";
        
        // initializations
        var bias = opt["bias_pref"] != null ? opt["bias_pref"] : 0.0;
        self.filters = [];
        for i in 0 ..< self.out_depth  { { self.filters.push(Vol(1, 1, self.num_inputs)); }
        self.biases = Vol(1, 1, self.out_depth, bias);
    }
    
    func forward(V: Vol, is_training: Bool) -> () {
        self.in_act = V;
        var A = Vol(1, 1, self.out_depth, 0.0);
        var Vw = V.w;
        for i in 0 ..< self.out_depth { {
            var a = 0.0;
            var wi = self.filters[i].w;
            for d in 0 ..< self.num_inputs { {
                a += Vw[d] * wi[d]; // for efficiency use Vols directly for now
            }
            a += self.biases.w[i];
            A.w[i] = a;
        }
        self.out_act = A;
        return self.out_act;
    }
    
    func backward() -> () {
        var V = self.in_act;
        V.dw = zeros(V.w.length); // zero out the gradient in input Vol
        
        // compute gradient wrt weights and data
        for i in 0 ..< self.out_depth { {
            var tfi = self.filters[i];
            var chain_grad = self.out_act.dw[i];
            for d in 0 ..< self.num_inputs { {
                V.dw[d] += tfi.w[d]*chain_grad; // grad wrt input data
                tfi.dw[d] += V.w[d]*chain_grad; // grad wrt params
            }
            self.biases.dw[i] += chain_grad;
        }
    }
    
    func getParamsAndGrads() -> [AnyObject] {
        var response = [];
        for i in 0 ..< self.out_depth { {
            response.append(["params": self.filters[i].w, "grads": self.filters[i].dw, "l1_decay_mul": self.l1_decay_mul, "l2_decay_mul": self.l2_decay_mul]);
        }
        response.append(["params": self.biases.w, "grads": self.biases.dw, "l1_decay_mul": 0.0, "l2_decay_mul": 0.0]);
        return response;
    }
    
    func toJSON() -> () {
        var json = {};
        json.out_depth = self.out_depth;
        json.out_sx = self.out_sx;
        json.out_sy = self.out_sy;
        json.layer_type = self.layer_type;
        json.num_inputs = self.num_inputs;
        json.l1_decay_mul = self.l1_decay_mul;
        json.l2_decay_mul = self.l2_decay_mul;
        json.filters = [];
        for i in 0 ..< self.filters.length { {
            json.filters.push(self.filters[i].toJSON());
        }
        json.biases = self.biases.toJSON();
        return json;
    }
    
    func fromJSON(json: [String: AnyObject]) -> () {
        self.out_depth = json.out_depth;
        self.out_sx = json.out_sx;
        self.out_sy = json.out_sy;
        self.layer_type = json.layer_type;
        self.num_inputs = json.num_inputs;
        self.l1_decay_mul = json.l1_decay_mul != null ? json.l1_decay_mul : 1.0;
        self.l2_decay_mul = json.l2_decay_mul != null ? json.l2_decay_mul : 1.0;
        self.filters = [];
        for i in 0 ..< json.filters.length { {
            var v = Vol(0,0,0,0);
            v.fromJSON(json.filters[i]);
            self.filters.push(v);
        }
        self.biases = Vol(0,0,0,0);
        self.biases.fromJSON(json.biases);
    }
}

