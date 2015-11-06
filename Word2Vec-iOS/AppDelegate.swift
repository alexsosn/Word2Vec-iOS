//
//  AppDelegate.swift
//  Word2Vec-iOS
//
//  Created by Tanya on 10/8/15.
//  Copyright © 2015 OWL. All rights reserved.
//

import UIKit
import AVFoundation

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?


    func application(application: UIApplication, didFinishLaunchingWithOptions launchOptions: [NSObject: AnyObject]?) -> Bool {
        
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
        
        let init_word = "cat"
        var acc : [String] = [init_word]
        let result = model.distance(init_word, numberOfClosest: 10)
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
        
        // Override point for customization after application launch.
        return true
    }

    func applicationWillResignActive(application: UIApplication) {
        // Sent when the application is about to move from active to inactive state. This can occur for certain types of temporary interruptions (such as an incoming phone call or SMS message) or when the user quits the application and it begins the transition to the background state.
        // Use this method to pause ongoing tasks, disable timers, and throttle down OpenGL ES frame rates. Games should use this method to pause the game.
    }

    func applicationDidEnterBackground(application: UIApplication) {
        // Use this method to release shared resources, save user data, invalidate timers, and store enough application state information to restore your application to its current state in case it is terminated later.
        // If your application supports background execution, this method is called instead of applicationWillTerminate: when the user quits.
    }

    func applicationWillEnterForeground(application: UIApplication) {
        // Called as part of the transition from the background to the inactive state; here you can undo many of the changes made on entering the background.
    }

    func applicationDidBecomeActive(application: UIApplication) {
        // Restart any tasks that were paused (or not yet started) while the application was inactive. If the application was previously in the background, optionally refresh the user interface.
    }

    func applicationWillTerminate(application: UIApplication) {
        // Called when the application is about to terminate. Save data if appropriate. See also applicationDidEnterBackground:.
    }


}

