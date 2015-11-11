//
//  Word2Vec_iOSTests.swift
//  Word2Vec-iOSTests
//
//  Created by Tanya on 10/8/15.
//  Copyright © 2015 OWL. All rights reserved.
//

import XCTest
import AVFoundation

//@testable import Word2Vec_iOS

class Word2Vec_iOSTests: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testAnalogy() {
        let paths = NSSearchPathForDirectoriesInDomains(.DocumentDirectory, .UserDomainMask, true)
        let documentsDirectory = paths[0]
        let gooUrl = NSBundle.mainBundle().URLForResource("text8", withExtension: nil)
        let binUrl = NSBundle.mainBundle().URLForResource("out.bin", withExtension: nil)
        let url = NSURL(fileURLWithPath: documentsDirectory).URLByAppendingPathComponent("out.bin")
        //        let exists = NSFileManager.defaultManager().fileExistsAtPath(url.path!)
        //        print(url.path)
        //
        
        let model = Word2VecModel()
        model.outputFile = binUrl
        
        //        if !exists {
        //            model.trainFile = gooUrl
        //            model.train()
        //        }
        
        //        model.outputFile = NSBundle.mainBundle().URLForResource("GoogleNews-vectors-negative300", withExtension: "bin")
        let synth = AVSpeechSynthesizer()
        
        let init_word = "airplane car"
        var acc : [String] = [init_word]
        let result = model.analogy(init_word, numberOfClosest: 10)
        print(result)
        var closest = result?.reduce(("", 0.0), combine: {
            (prew: (String, Float), this: (String, Float)) -> (String, Float) in
            return max(prew.1, this.1) == prew.1 ? prew : this
        })
        acc.append(closest!.0)
        print(closest!.0)
        //        let voice = AVSpeechSynthesisVoice(language: "EN")
        let utterance = AVSpeechUtterance(string: acc.last!)
        //        utterance.voice = voice
        synth.speakUtterance(utterance)
        
        //        for _ in 0..<100 {
        //            var result = model.distance(closest!.0, numberOfClosest: 1)
        //            for _ in 0 ..< result!.count {
        //                closest = result!.reduce(("", 0.0), combine: {
        //                    (prew: (String, Float), this: (String, Float)) -> (String, Float) in
        //                    return max(prew.1, this.1) == prew.1 ? prew : this
        //                })
        //                let new_association = closest!.0
        //
        //                if acc.contains(new_association) {
        //                    result?.removeValueForKey(new_association)
        //                } else {
        //                    acc.append(new_association)
        //                    break
        //                }
        //            }
        //            print(acc.last!)
        //            dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_BACKGROUND, 0), { () -> Void in
        //                let utterance = AVSpeechUtterance(string: acc.last!)
        //                //            utterance.voice = voice
        //                synth.speakUtterance(utterance)
        //            })
        //
        //        }

    }
}
