//
//  Word2Vec_iOSTests.swift
//  Word2Vec-iOSTests
//
//  Created by Tanya on 10/8/15.
//  Copyright © 2015 OWL. All rights reserved.
//

import XCTest

class Word2Vec_iOSTests: XCTestCase {
    var model: Word2VecModel!
    
    override func setUp() {
        super.setUp()
        model = Word2VecModel()
        let binUrl = NSBundle.mainBundle().URLForResource("out.bin", withExtension: nil)
        model.outputFile = binUrl
    }
    
    override func tearDown() {
        let paths = NSSearchPathForDirectoriesInDomains(.DocumentDirectory, .UserDomainMask, true)
        let documentsDirectory = paths[0]
        let url = NSURL(fileURLWithPath: documentsDirectory).URLByAppendingPathComponent("out.bin")
        let exists = NSFileManager.defaultManager().fileExistsAtPath(url.path!)
        if exists {
            do {
                try NSFileManager.defaultManager().removeItemAtPath(url.path!)
            } catch {}        }
        
        super.tearDown()
    }
    
    func testTrain() {
        let paths = NSSearchPathForDirectoriesInDomains(.DocumentDirectory, .UserDomainMask, true)
        let documentsDirectory = paths[0]
        let gooUrl = NSBundle.mainBundle().URLForResource("pg2701", withExtension: "txt")
        XCTAssertNotNil(gooUrl)
        
        let url = NSURL(fileURLWithPath: documentsDirectory).URLByAppendingPathComponent("out.bin")
        var exists = NSFileManager.defaultManager().fileExistsAtPath(url.path!)
        XCTAssertFalse(exists)
        
        model.trainFile = gooUrl
        model.outputFile = url
        model.train()
        exists = NSFileManager.defaultManager().fileExistsAtPath(url.path!)
        XCTAssertTrue(exists)
    }
    
    func testDistance() {
        let result = model.distance("cat", numberOfClosest: 1)
        XCTAssertEqual(result?.keys.first, "dog")
    }
    
    func testPerformance() {
        measureBlock { () -> Void in
            let init_word = "bird"
            var acc : [String] = [init_word]
            let result = self.model.distance(init_word, numberOfClosest: 10)
            var closest = result?.reduce(("", 0.0), combine: {
                (prew: (String, Float), this: (String, Float)) -> (String, Float) in
                return max(prew.1, this.1) == prew.1 ? prew : this
            })
            acc.append(closest!.0)
            for _ in 0..<100 {
                var result = self.model.distance(closest!.0, numberOfClosest: 10)
                for _ in 0 ..< result!.count {
                    closest = result!.reduce(("", 0.0), combine: {
                        (prew: (String, Float), this: (String, Float)) -> (String, Float) in
                        return max(prew.1, this.1) == prew.1 ? prew : this
                    })
                    let new_association = closest!.0
                    
                    if acc.contains(new_association) {
                        result?.removeValueForKey(new_association)
                    } else {
                        acc.append(new_association)
                        break
                    }
                }
            }
        }
    }
    
    func testAnalogy() {
        
        let result = model.analogy("man woman king", numberOfClosest: 1)
        XCTAssertEqual(result?.keys.first, "queen")
        
        let result2 = model.analogy("pet toy", numberOfClosest: 1)
        XCTAssertEqual(result2?.keys.first, "eat")
        
    }
}
