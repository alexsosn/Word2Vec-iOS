//
//  Word2Vec_iOSTests.swift
//  Word2Vec-iOSTests
//
//  Created by Tanya on 10/8/15.
//  Copyright © 2015 OWL. All rights reserved.
//

import XCTest
@testable import Word2Vec_iOS

class Word2Vec_iOSTests: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
    }
    
    func test2LayerNNPerformance() {
        // Here's a minimum example of defining a 2-layer neural network and training it on a single data point:
        self.measureBlock {
            // species a 2-layer neural network with one hidden layer of 20 neurons
            var layer_defs: [AnyObject?] = []
            // input layer declares size of input. here: 2-D data
            // ConvNetJS works on 3-Dimensional volumes (sx, sy, depth), but if you're not dealing with images
            // then the first two dimensions (sx, sy) will always be kept at size 1
            layer_defs.append(["type":"input", "out_sx":1, "out_sy":1, "out_depth":2])
            // declare 20 neurons, followed by ReLU (rectified linear unit non-linearity)
            layer_defs.append(["type":"fc", "num_neurons":20, "activation":"relu"])
            // declare the linear classifier on top of the previous hidden layer
            layer_defs.append(["type":"softmax", "num_classes":10])
            
            var net = Net()
            net.makeLayers(layer_defs)
            
            // forward a random data point through the network
            var x = Vol([0.3, -0.5])
            var prob = net.forward(x)
            
            // prob is a Vol. Vols have a field .w that stores the raw data, and .dw that stores gradients
            print("probability that x is class 0: " + prob.w[0]) // prints 0.50101
            
            var trainer = SGDTrainer(net, ["learning_rate":0.01, "l2_decay":0.001])
            trainer.train(x, 0); // train the network, specifying that x is class zero
            
            var prob2 = net.forward(x);
            print("probability that x is class 0: " + prob2.w[0])
            // now prints 0.50374, slightly higher than previous 0.50101: the networks
            // weights have been adjusted by the Trainer to give a higher probability to
            // the class we trained the network with (zero)
        }
    }
    
    func testConvolutionalNN() {
        // Small Convolutional Neural Network if you wish to predict on images
        self.measureBlock {
            var layer_defs: [AnyObject?] = []
            layer_defs.append(["type":"input", "out_sx":32, "out_sy":32, "out_depth":3]) // declare size of input
            // output Vol is of size 32x32x3 here
            layer_defs.append(["type":"conv", "sx":5, "filters":16, "stride":1, "pad":2, "activation":"relu"])
            // the layer will perform convolution with 16 kernels, each of size 5x5.
            // the input will be padded with 2 pixels on all sides to make the output Vol of the same size
            // output Vol will thus be 32x32x16 at this point
            layer_defs.append(["type":"pool", "sx":2, "stride":2])
            // output Vol is of size 16x16x16 here
            layer_defs.append(["type":"conv", "sx":5, "filters":20, "stride":1, "pad":2, "activation":"relu"])
            // output Vol is of size 16x16x20 here
            layer_defs.append(["type":"pool", "sx":2, "stride":2])
            // output Vol is of size 8x8x20 here
            layer_defs.append(["type":"conv", "sx":5, "filters":20, "stride":1, "pad":2, "activation":"relu"])
            // output Vol is of size 8x8x20 here
            layer_defs.append(["type":"pool", "sx":2, "stride":2])
            // output Vol is of size 4x4x20 here
            layer_defs.append(["type":"softmax", "num_classes":10])
            // output Vol is of size 1x1x10 here
            
            let net = Net()
            net.makeLayers(layer_defs)
            
            // helpful utility for converting images into Vols is included
            var x = img_to_vol(document.getElementById("#some_image"))
            var output_probabilities_vol = net.forward(x)
        }
    }
}
