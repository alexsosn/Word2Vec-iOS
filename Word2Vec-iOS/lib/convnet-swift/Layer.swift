//
//  Layer.swift
//  Word2Vec-iOS
//

class Layer: NSObject {
    override init() {
        //
    }
    
    var sx: Int
    var sy: Int
    
    var in_sx: Int
    var in_sy: Int
    var in_depth: Int
    
    var out_sx: Int
    var out_sy: Int
    var out_depth: Int
    
    var stride: Int
    var pad: Int
    
    var l1_decay_mul: Float
    var l2_decay_mul: Float
    
    var layer_type: String
    var filters = []
    
    var biases: Vol
    var in_act: Vol
}
