
// Net manages a set of layers
// For now constraints: Simple linear order of layers, first layer input last layer a cost layer

class Net {
    var layers: [AnyObject?] = []
    
    // takes a list of layer definitions and creates the network layer objects
    func makeLayers(defs: [[String:AnyObject?]]) -> () {
        
        // few checks
        assert(defs.length >= 2, "Error! At least one input layer and one loss layer are required.")
        assert(defs[0].type === "input", "Error! First layer must be the input layer, to declare size of inputs")
        
        // desugar layer_defs for adding activation, dropout layers etc
        func desugar() {
            var new_defs = []
            for(var i=0;i<defs.length;i++) {
                var def = defs[i]
                
                if(def["type"]==="softmax" || def["type"]==="svm") {
                    // add an fc layer here, there is no reason the user should
                    // have to worry about this and we almost always want to
                    new_defs.append(["type":"fc", "num_neurons": def["num_classes"]])
                }
                
                if(def["type"]==="regression") {
                    // add an fc layer here, there is no reason the user should
                    // have to worry about this and we almost always want to
                    new_defs.append(["type":"fc", "num_neurons": def["num_neurons"]])
                }
                
                if((def["type"]==="fc" || def["type"]==="conv")
                    && typeof(def["bias_pref"]) === "undefined"){
                        def["bias_pref"] = 0.0
                        if(def["activation"] && def["activation"] === "relu") {
                            def["bias_pref"] = 0.1 // relus like a bit of positive bias to get gradients early
                            // otherwise it's technically possible that a relu unit will never turn on (by chance)
                            // and will never get any gradient and never contribute any computation. Dead relu.
                        }
                }
                
                new_defs.append(def)
                
                if(def["activation"]) {
                    if(def["activation"]==="relu") { new_defs.append(["type":"relu"]) }
                    else if (def["activation"]==="sigmoid") { new_defs.append(["type":"sigmoid"]) }
                    else if (def["activation"]==="tanh") { new_defs.append(["type":"tanh"]) }
                    else if (def["activation"]==="maxout") {
                        // create maxout activation, and pass along group size, if provided
                        var gs = def["group_size"] !== "undefined" ? def["group_size"] : 2
                        new_defs.append(["type":"maxout", "group_size":gs])
                    }
                    else { console.log("ERROR unsupported activation " + def["activation"]) }
                }
                if(def["drop_prob"] && def["type"] !== "dropout") {
                    new_defs.append(["type":"dropout", "drop_prob": def["drop_prob"]])
                }
                
            }
            return new_defs
        }
        
        defs = desugar(defs)
        
        // create the layers
        self.layers = []
        for(var i=0;i<defs.length;i++) {
            var def = defs[i]
            if(i>0) {
                var prev = self.layers[i-1]
                def["in_sx"] = prev.out_sx
                def["in_sy"] = prev.out_sy
                def["in_depth"] = prev.out_depth
            }
            
            switch(def["type"]) {
            case "fc": self.layers.append(FullyConnLayer(def))
            case "lrn": self.layers.append(LocalResponseNormalizationLayer(def))
            case "dropout": self.layers.append(DropoutLayer(def))
            case "input": self.layers.append(InputLayer(def))
            case "softmax": self.layers.append(SoftmaxLayer(def))
            case "regression": self.layers.append(RegressionLayer(def))
            case "conv": self.layers.append(ConvLayer(def))
            case "pool": self.layers.append(PoolLayer(def))
            case "relu": self.layers.append(ReluLayer(def))
            case "sigmoid": self.layers.append(SigmoidLayer(def))
            case "tanh": self.layers.append(TanhLayer(def))
            case "maxout": self.layers.append(MaxoutLayer(def))
            case "svm": self.layers.append(SVMLayer(def))
            default: println("ERROR: UNRECOGNIZED LAYER TYPE: " + def["type"])
            }
            
        }
    }
    
    // forward prop the network.
    // The trainer class passes is_training = true, but when this function is
    // called from outside (not from the trainer), it defaults to prediction mode
    func forward(V: Vol, is_training: Bool?) {
        if _ = is_training {
            
        } else {
            is_training = false
        }
        var act = self.layers[0].forward(V, is_training)
        for(var i=1;i<self.layers.length;i++) {
            act = self.layers[i].forward(act, is_training)
        }
        return act
    }
    
    func getCostLoss(V: Vol, y: AnyObject?) {
        self.forward(V, false)
        var N = self.layers.length
        var loss = self.layers[N-1].backward(y)
        return loss
    }
    
    // backprop: compute gradients wrt all parameters
    func backward(y: AnyObject?) {
        var N = self.layers.length
        var loss = self.layers[N-1].backward(y) // last layer assumed to be loss layer
        for(var i=N-2;i>=0;i--) { // first layer assumed input
            self.layers[i].backward()
        }
        return loss
    }
    
    func getParamsAndGrads() {
        // accumulate parameters and gradients for the entire network
        var response = []
        for(var i=0;i<self.layers.length;i++) {
            var layer_reponse = self.layers[i].getParamsAndGrads()
            for(var j=0;j<layer_reponse.length;j++) {
                response.append(layer_reponse[j])
            }
        }
        return response
    }
    
    func getPrediction() -> () {
        // this is a convenience function for returning the argmax
        // prediction, assuming the last layer of the net is a softmax
        var S = self.layers[self.layers.length-1]
        assert(S.layer_type === "softmax", "getPrediction function assumes softmax as last layer of the net!")
        
        var p = S.out_act.w
        var maxv = p[0]
        var maxi = 0
        for(var i=1;i<p.length;i++) {
            if(p[i] > maxv) {
                maxv = p[i]
                maxi = i
            }
        }
        return maxi // return index of the class with highest class probability
    }
    
    func toJSON() -> () {
        var json = {}
        json.layers = []
        for(var i=0;i<self.layers.length;i++) {
            json.layers.append(self.layers[i].toJSON())
        }
        return json
    }
    
    func fromJSON(json: [String: AnyObject?]) -> () {
        self.layers = []
        for(var i=0;i<json.layers.length;i++) {
            var Lj = json.layers[i]
            var t = Lj.layer_type
            var L
            if(t==="input") { L = InputLayer() }
            if(t==="relu") { L = ReluLayer() }
            if(t==="sigmoid") { L = SigmoidLayer() }
            if(t==="tanh") { L = TanhLayer() }
            if(t==="dropout") { L = DropoutLayer() }
            if(t==="conv") { L = ConvLayer() }
            if(t==="pool") { L = PoolLayer() }
            if(t==="lrn") { L = LocalResponseNormalizationLayer() }
            if(t==="softmax") { L = SoftmaxLayer() }
            if(t==="regression") { L = RegressionLayer() }
            if(t==="fc") { L = FullyConnLayer() }
            if(t==="maxout") { L = MaxoutLayer() }
            if(t==="svm") { L = SVMLayer() }
            L.fromJSON(Lj)
            self.layers.append(L)
        }
    }
}

