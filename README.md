# Word2Vec-iOS
Word2Vec Swift wrapper.

Original Word2Vec implementation is [here](https://code.google.com/p/word2vec/).
Nice alghorithm description is available on [deeplearning4j](http://deeplearning4j.org/word2vec.html) documentation page.

## Basic usage

### Train the model

```swift
let inputURL = NSBundle.mainBundle().URLForResource("text8", withExtension: nil)

let paths = NSSearchPathForDirectoriesInDomains(.DocumentDirectory, .UserDomainMask, true)
let documentsDirectory = paths[0]
let outputURL = NSURL(fileURLWithPath: documentsDirectory).URLByAppendingPathComponent("out.bin")

let model = Word2VecModel()
model.trainFile = inputURL
model.outputFile = outputURL
model.train()
```
You can find some text corpuses and pretrained models (``*.bin``) in ``Word2Vec-iOS/res`` folder.

### Use model to find connected words:
```swift
let result = model.distance("cat", numberOfClosest: 10) // Look for 10 words semantically similar to "cat".
print(result)
```
Outputs words and similarity measure:

> Optional(["bird": 0.760521, "cow": 0.766533, "dog": 0.831517, "rat": 0.748557, "blonde": 0.763721, "pig": 0.751001, "goat": 0.798104, "hamster": 0.768635, "bee": 0.774112, "llama": 0.747295])

Blonde and cat, really? hmmm...

```swift
let result = model.distance("wedding", numberOfClosest: 10)
print(result)
```
> Optional(["banquet": 0.723855, "dinner": 0.711831, "funeral": 0.772873, "madonna": 0.688461, "bride": 0.721245, "diana": 0.685946, "bedroom": 0.691792, "reunion": 0.673289, "aunt": 0.703193, "maid": 0.66286])

### Find analogy:

```swift
let result = model.analogy("man woman king", numberOfClosest: 1) // (man - woman) == (king - ???)
print(result)
```
> Optional(["daughter": 0.613628, "queen": 0.707821, "empress": 0.62057, "prince": 0.611979, "elizabeth": 0.60867])

Man to woman is the same as king to queen.

Another example:
```swift
let result = model.analogy("pet toy", numberOfClosest: 1) // (pet - toy) = (???)
print(result)
```
> Optional(["partake": 0.481817, "eat": 0.502222, "allah": 0.457967, "preach": 0.49685, "marry": 0.449955])

Pet is a toy which eats. (Or the toy you can eat?)
