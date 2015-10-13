

class InputLayer: Layer {
    init(){}
    
    convenience init(opt: Options){
        var opt = opt || {};
        
        // required: depth
        self.out_depth = getopt(opt, ["out_depth", "depth"], 0);
        
        // optional: default these dimensions to 1
        self.out_sx = getopt(opt, ["out_sx", "sx", "width"], 1);
        self.out_sy = getopt(opt, ["out_sy", "sy", "height"], 1);
        
        // computed
        self.layer_type = "input";
    }
    
    func forward(V: Vol, is_training: Bool) -> () {
        self.in_act = V;
        self.out_act = V;
        return self.out_act; // simply identity function for now
    }
    func backward() -> () { }
    func getParamsAndGrads() -> () {
        return [];
    }
    func toJSON() -> () {
        var json = {};
        json.out_depth = self.out_depth;
        json.out_sx = self.out_sx;
        json.out_sy = self.out_sy;
        json.layer_type = self.layer_type;
        return json;
    }
    func fromJSON(json) -> () {
        self.out_depth = json.out_depth;
        self.out_sx = json.out_sx;
        self.out_sy = json.out_sy;
        self.layer_type = json.layer_type; 
    }
}

