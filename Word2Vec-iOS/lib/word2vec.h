
#import <Foundation/Foundation.h>

@interface Word2Vec : NSObject

+ (void)prapareWithTrainFile:(NSURL * _Nonnull) trainFile
                  outputFile:(NSURL * _Nonnull) outputFile
               saveVocabFile:(NSURL * _Nullable) saveVocabFile
               readVocabFile:(NSURL * _Nullable) readVocabFile
              wordVectorSize:(NSNumber * _Nullable) wordVectorSize
                       debug:(NSNumber * _Nullable) debug
                saveToBinary:(NSNumber * _Nullable) saveToBinary
        continuousBagOfWords:(NSNumber * _Nullable) continuousBagOfWords
        startingLearningRate:(NSNumber * _Nullable) startingLearningRate
                windowLength:(NSNumber * _Nullable) windowLength
    wordsOccurrenceThreshold:(NSNumber * _Nullable) wordsOccurrenceThreshold
         hierarchicalSoftmax:(NSNumber * _Nullable) hierarchicalSoftmax
            negativeExamples:(NSNumber * _Nullable) negativeExamples
                     threads:(NSNumber * _Nullable) threads
          trainingIterations:(NSNumber * _Nullable) trainingIterations
                    minCount:(NSNumber * _Nullable) minCount
               classesNumber:(NSNumber * _Nullable) classesNumber;

+ (void)trainModel;

@end
