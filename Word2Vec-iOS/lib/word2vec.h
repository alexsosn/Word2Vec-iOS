
#ifndef WORD2VEC_H
#define WORD2VEC_H

#import <Foundation/Foundation.h>

void Prepare(NSURL * _Nonnull trainFile,
             NSURL * _Nonnull outputFile,
             NSURL * _Nullable saveVocabFile,
             NSURL * _Nullable readVocabFile,
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
