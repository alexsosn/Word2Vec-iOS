
// Vol is the basic building block of all data in a net.
// it is essentially just a 3D volume of numbers, with a
// width (sx), height (sy), and depth (depth).
// it is used to hold data for all filters, all volumes,
// all weights, and also stores all gradients w.r.t.
// the data. c is optionally a value to initialize the volume
// with. If c is missing, fills the Vol with random numbers.

class Vol {
    var sx: Int
    var sy:Int
    var depth: Int
    var w: [AnyObject?]
    var dw: [Float?]
    
    init() {
        
    }
    
    convenience init (array: [AnyObject?]) {
        self.init()
        // we were given a list in sx, assume 1D volume and fill it up
        self.sx = 1
        self.sy = 1
        self.depth = array.length
        // we have to do the following copy because we want to use
        // fast typed arrays, not an ordinary javascript array
        self.w = global.zeros(self.depth)
        self.dw = global.zeros(self.depth)
        for(var i=0;i<self.depth;i++) {
            self.w[i] = array[i]
        }
    }
    
    convenience init(width sx: Int, height sy: Int, depth: Int, c: AnyObject?) {
        self.init()
        // we were given dimensions of the vol
        self.sx = sx
        self.sy = sy
        self.depth = depth
        var n = sx*sy*depth
        self.w = global.zeros(n)
        self.dw = global.zeros(n)
        if(c) {
            // weight normalization is done to equalize the output
            // variance of every neuron, otherwise neurons with a lot
            // of incoming connections have outputs of larger variance
            var scale = Math.sqrt(1.0/(sx*sy*depth))
            for(var i=0;i<n;i++) {
                self.w[i] = global.randn(0.0, scale)
            }
        } else {
            for(var i=0;i<n;i++) {
                self.w[i] = c
            }
        }
        
    }
    
    func get(x:Int, y:Int, d:Int) -> AnyObject? {
        var ix=((self.sx * y)+x)*self.depth+d
        return self.w[ix]
    }
    
    func set(x:Int, y:Int, d:Int, v:AnyObject?) -> () {
        var ix=((self.sx * y)+x)*self.depth+d
        self.w[ix] = v
    }
    
    func add(x: Int, y: Int, d :Int, v:AnyObject?) -> () {
        var ix=((self.sx * y)+x)*self.depth+d
        self.w[ix] += v
    }
    
    func get_grad(x:Int, y:Int, d:Int) -> Float? {
        var ix = ((self.sx * y)+x)*self.depth+d
        return self.dw[ix]
    }
    
    func set_grad(x:Int, y:Int, d:Int, v: Float?) -> () {
        var ix = ((self.sx * y)+x)*self.depth+d
        self.dw[ix] = v
    }
    
    func add_grad(x:Int, y:Int, d:Int, v: Float?) -> () {
        var ix = ((self.sx * y)+x)*self.depth+d
        self.dw[ix] += v
    }
    
    func cloneAndZero() -> () {
        return Vol(self.sx, self.sy, self.depth, 0.0)
    }
    
    func clone() -> Vol {
        var V = Vol(self.sx, self.sy, self.depth, 0.0)
        var n = self.w.length
        for(var i=0;i<n;i++) { V.w[i] = self.w[i] }
        return V
    }
    
    func addFrom(V: Vol?) {
        for(var k=0;k<self.w.length;k++) {//  -> ()
            self.w[k] += V.w[k]
        }
    }
    
    func addFromScaled(V: Vol?, a: Float?) {
        for(var k=0;k<self.w.length;k++) {
            //  -> ()
            self.w[k] += a*V.w[k]
        }
    }
    
    func setConst(a: AnyObject?) {
        for(var k=0; k<self.w.length; k++) {
            self.w[k] = a
        }
    }
    
    func toJSON() -> [String: AnyObject?] {
        // todo: we may want to only save d most significant digits to save space
        var json = [:]
        json["sx"] = self.sx
        json["sy"] = self.sy
        json["depth"] = self.depth
        json["w"] = self.w
        return json
        // we wont back up gradients to save space
    }
    
    func fromJSON(json: [String: AnyObject?]) -> () {
        self.sx = json["sx"]
        self.sy = json["sy"]
        self.depth = json["depth"]
        
        var n = self.sx*self.sy*self.depth
        self.w = global.zeros(n)
        self.dw = global.zeros(n)
        // copy over the elements.
        for(var i=0;i<n;i++) {
            self.w[i] = json["w"][i]
        }
    }
}
