
#ifndef WORD2VEC_H
#define WORD2VEC_H

#import <Foundation/Foundation.h>

void Prepare(NSString *trainFile,
             NSString *outputFile,
             NSString * _Nullable saveVocabFile,
             NSString * _Nullable readVocabFile,
             NSNumber * _Nullable wordVectorSize,
             NSNumber * _Nullable debug,
             NSNumber * _Nullable saveToBinary,
             NSNumber * _Nullable continuousBagOfWords,
             NSNumber * _Nullable startingLearningRate,
             NSNumber * _Nullable windowLength,
             NSNumber * _Nullable wordsOccurrenceThreshold,
             NSNumber * _Nullable hierarchicalSoftmax,
             NSNumber * _Nullable negativeExamples,
             NSNumber * _Nullable threads,
             NSNumber * _Nullable trainingIterations,
             NSNumber * _Nullable minCount,
             NSNumber * _Nullable classesNumber
             ) ;

void TrainModel();

#endif
