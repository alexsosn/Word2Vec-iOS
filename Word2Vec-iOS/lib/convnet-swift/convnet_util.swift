

// Random number utilities
var return_v = false;
var v_val = 0.0;

func gaussRandom() -> Double {
    if(return_v) {
        return_v = false;
        return v_val;
    }
    
    var u = 2*random()-1;
    var v = 2*Math.random()-1;
    var r = u*u + v*v;
    if(r == 0 || r > 1) { return gaussRandom(); }
    var c = Math.sqrt(-2*Math.log(r)/r);
    v_val = v*c; // cache this
    return_v = true;
    return u*c;
}

func randf(a: Double, b: Double) -> Double {
    return Math.random()*(b-a)+a;
}
func randi(a: Int, b: Int) -> Int {
    return Math.floor(Math.random()*(b-a)+a);
}

func randn(mu: Double, std: Double) -> Double {
    return mu+gaussRandom()*std;
}

// Array utilities
func zeros(n: Int) -> [Double] {
    //    if(typeof(n)==="undefined" || isNaN(n)) { return []; }
    //    if(ArrayBuffer == null) {
    //      // lacking browser support
    //      var arr = Array(n);
    //      for(var i=0;i<n;i++) { arr[i]= 0; }
    //      return arr;
    //    } else {
    //      return Float64Array(n);
    //    }
    return [Double](count: n, repeatedValue: 0.0);
}

//func arrContains(arr, elt) {
//    for(var i=0,n=arr.length;i<n;i++) {
//        if(arr[i]===elt) return true;
//    }
//    return false;
//}

func arrUnique(arr: [AnyObject]) -> [AnyObject] {
    return Array(Set(arr));
}

// return max and min of a given non-empty array.
func maxmin(w: [Double]) -> [String: Double]{
    if(w.length === 0) { return {}; } // ... ;s
    var maxv = w[0];
    var minv = w[0];
    var maxi = 0;
    var mini = 0;
    var n = w.length;
    for(var i=1;i<n;i++) {
        if(w[i] > maxv) { maxv = w[i]; maxi = i; }
        if(w[i] < minv) { minv = w[i]; mini = i; }
    }
    return ["maxi": maxi, "maxv": maxv, "mini": mini, "minv": minv, "dv":maxv-minv];
}

// create random permutation of numbers, in range [0...n-1]
func randperm(n: Int) -> [Int]{
    var i = n,
    j = 0,
    temp;
    var array = [];
    for(var q=0;q<n;q++){array[q]=q;}
    while (i--) {
        j = Math.floor(Math.random() * (i+1));
        temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    return array;
}

// sample from list lst according to probabilities in list probs
// the two lists are of same size, and probs adds up to 1
func weightedSample(lst: [AnyObject], probs: [Double]) -> [AnyObject]{
    var p = randf(0, 1.0);
    var cumprob = 0.0;
    for(var k=0,n=lst.length;k<n;k++) {
        cumprob += probs[k];
        if(p < cumprob) { return lst[k]; }
    }
}

// syntactic sugar function for getting default parameter values
func getopt(opt: [String: AnyObject], field_name: String, default_value: AnyObject) -> AnyObject {
        // case of single string
        return (opt[field_name] != null) ? opt[field_name] : default_value
}

func getopt(opt: [String: AnyObject], field_name: [String], default_value: AnyObject) -> AnyObject{
    // assume we are given a list of string instead
    var ret = default_value;
    for(var i=0;i<field_name.length;i++) {
        var f = field_name[i];
        if (opt[f] != null) {
            ret = opt[f]; // overwrite return value
        }
    }
    return ret;
}

