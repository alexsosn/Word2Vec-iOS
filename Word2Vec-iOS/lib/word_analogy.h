
#ifndef WORD_ANALOGY_H
#define WORD_ANALOGY_H

#import <Foundation/Foundation.h>

NSDictionary <NSString *, NSNumber *>  * _Nullable  Analogy(NSURL * _Nonnull fileURL,
                                                            NSString * _Nonnull threeWords,
                                                            NSNumber * _Nullable numberOfClosest,
                                                            NSError ** _Nullable error);

#endif