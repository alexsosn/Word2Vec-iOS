//
//  Word2VecModel.swift
//  Word2Vec-iOS
//
//  Created by Tanya on 10/8/15.
//  Copyright © 2015 OWL. All rights reserved.
//

import Foundation

class Word2VecModel: NSObject {
    /*
    
    WORD VECTOR estimation toolkit v 0.1c
    
    Options:
    Parameters for training:
    -train <file>
    Use text data from <file> to train the model
    -output <file>
    Use <file> to save the resulting word vectors / word clusters
    -size <int>
    Set size of word vectors; default is 100
    -window <int>
    Set max skip length between words; default is 5
    -sample <float>
    Set threshold for occurrence of words. Those that appear with higher frequency in the training data
    will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)
    -hs <int>
    Use Hierarchical Softmax; default is 0 (not used)
    -negative <int>
    Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)
    -threads <int>
    Use <int> threads (default 12)
    -iter <int>
    Run more training iterations (default 5)
    -min-count <int>
    his will discard words that appear less than <int> times; default is 5
    -alpha <float>
    Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
    -classes <int>
    Output word classes rather than word vectors; default number of classes is 0 (vectors are written)
    -debug <int>
    Set the debug mode (default = 2 = more info during training)
    -binary <int>
    Save the resulting vectors in binary moded; default is 0 (off)
    -save-vocab <file>
    the vocabulary will be saved to <file>
    -read-vocab <file>
    the vocabulary will be read from <file>, not constructed from the training data
    -cbow <int>
    Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)
    
    Examples:
    ./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3
    
    */
    
    var trainFile: NSURL!
    var outputFile: NSURL!
    var wordVectorSize: Int = 100
    var windowLength: Int = 5
    var wordsOccurrenceThreshold: Float = 1e-3
    var hierarchicalSoftmax: Bool = false
    var negativeExamples: Int = 5
    var threads: Int = 12
    var trainingIterations: Int = 5
    var minCount: Int = 5
    var startingLearningRate: Float = 0.025
    var classesNumber: Int = 0
    var debug: Int = 2
    var saveToBinary: Bool = true
    var saveVocabFile: NSURL?
    var readVocabFile: NSURL?
    var continuousBagOfWords: Bool = true
    var distanceModel: W2VDistance?
    
    func train() -> () {
        Word2Vec.prapareWithTrainFile(trainFile,
            outputFile: outputFile,
            saveVocabFile: saveVocabFile,
            readVocabFile: readVocabFile,
            wordVectorSize: wordVectorSize,
            debug: debug,
            saveToBinary: saveToBinary,
            continuousBagOfWords: continuousBagOfWords,
            startingLearningRate: startingLearningRate,
            windowLength: windowLength,
            wordsOccurrenceThreshold: wordsOccurrenceThreshold,
            hierarchicalSoftmax: hierarchicalSoftmax,
            negativeExamples: negativeExamples,
            threads: threads,
            trainingIterations: trainingIterations,
            minCount: minCount,
            classesNumber: classesNumber)

        Word2Vec.trainModel()
    }
    
    func distance(word: String, numberOfClosest: Int?) -> [String : Float]? {
        let error: NSErrorPointer = nil
        let closest = numberOfClosest ?? 40
        if distanceModel == nil {
            distanceModel = W2VDistance()
            distanceModel!.loadBinaryVectorFile(outputFile, error: error)
        }
        let distances = distanceModel!.closestToWord(word, numberOfClosest: closest)
        return distances as! [String : Float]?
    }
    
    func analogy (words: String, numberOfClosest: Int?) -> [String : Float]? {
        let error: NSErrorPointer = nil
        let closest = numberOfClosest ?? 40
        if distanceModel == nil {
            distanceModel = W2VDistance()
            distanceModel!.loadBinaryVectorFile(outputFile, error: error)
        }
        
        let distances = distanceModel!.analogyToPhrase(words, numberOfClosest: closest)
        return distances as! [String : Float]?
    }
}
