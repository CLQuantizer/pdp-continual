# Own. Wait. See.

*By Ez · 23 Nov 2026*

Backpropagation nets got deep on a short road. A residual connection lets each block pass on its input plus a small change, so a block that is not yet useful can be stepped around (He et al. 2016). The gradient flows down the road, and a hundred layers train.

On 15 Nov the question here was a different one. **What does a network get when no signal crosses a layer?** No backward pass, no gradient from above: every layer learns only from what it can see.

## The setting

A stack of small recurrent layers reads a text file (enwik8, the first 180,000 bytes) one byte at a time and predicts the next byte. Weights update as the bytes arrive; there is no training set and no test set, only the stream. Each layer learns with an exact local trace rule that stays inside the layer. The score is bits per byte, lower is better, on the late part of the stream (bytes 40,000 to 180,000) unless said otherwise. Every result is two seeds, with the pass mark written down before the run.

## What failed: one error for everyone

The first idea was the obvious one: make the stack behave like one model. Add the layers' outputs, and let every layer learn from the stack's error, or from what the layers below have left unexplained. That is boosting, a residual in prediction space.

It got worse with every layer. On an earlier run over 300,000 bytes, summed stacks of one, two and four layers scored 3.50, 3.57 and 3.88 over the whole stream. On the same run, a cascade, in which each layer corrects the sum below it, was 0.50 worse at four layers than one layer alone. Feeding the stack's error back to every layer through fixed random weights (feedback alignment) cost 0.28. Teaching every layer to copy the stack's own prediction (self-distillation) cost 0.05. Growing the stack one layer at a time cost 0.09 to 0.14.

**Every signal that made a layer answer for the whole stack's error made the stack worse.** A layer cannot fix a mistake it did not make and cannot see.

## Own

Give every layer its own loss. Each layer is a complete predictor of the next byte, and a gate mixes their predictions. The gate learns, but no signal from the mixture goes back into a layer.

Now depth pays. One, two, four and six layers score 3.275, 3.138, 3.091 and 3.081. Six layers of width 128 beat one layer of width 768, with about the same compute, by 0.108. A learned gate beats fixed equal weights by only 0.012: what matters is that each layer has a target of its own, not how they are mixed.

In backprop nets this is deep supervision, a loss at every hidden layer (Lee et al. 2015), used there as a helper beside the gradient. Here it is the only teacher.

But the start is expensive. Over the first 40,000 bytes, six layers score 7.80 against 4.51 for one layer. The untrained upper layers are in the prediction from the first byte, at full weight, while their inputs are still moving.

## Wait

Start the gate with all its weight on the first layer and none on the others. The stack begins life as one layer. An upper layer gets a share only when the gate finds that it helps.

Over the whole stream this saves 0.36 bits per byte at four layers and 0.75 at six. On the late stream the two starts are within 0.003. Over the first 20,000 bytes the four gates read 0.94, 0.02, 0.04, 0.04; over the last 20,000, 0.39, 0.21, 0.14, 0.19. The upper layers come in later. They do not start in.

Backprop nets found the same thing. Initialise the last scale of each residual branch at zero, and the branch starts as the identity (Goyal et al. 2017); SkipInit and ReZero start a multiplier on each branch at zero (De & Smith 2020; Bachlechner et al. 2020). A deep net that starts shallow trains faster. It holds without a backward pass too.

## See

With the gate fixed, depth still flattened: four, six and eight layers scored 3.093, 3.078 and 3.069. Eight layers gained only 0.009 on six.

The deep layers were not short of depth. They were short of input. Each layer read only the layer directly below it. Give each layer instead the **mean** of the features of all the layers below it. DenseNet concatenates them, and the width grows (Huang et al. 2017); the mean keeps the width fixed.

At six layers this is worth 0.062, more than two extra layers buy. And depth pays again: four, six and eight layers now score 3.056, 3.016 and 2.990, and eight gain 0.026 on six. Six layers that see everything below beat eight that see only their neighbour by 0.053.

## Three jobs

One last run separates the tricks. Take the six-layer stack that sees the mean, and start its gate fully open instead of at the first layer. On the late stream it lands within 0.006 of the zero start. Over the whole stream it costs 0.68 more.

So each trick has its own job:

- **Own** makes depth possible at all. Without it, layers fight over one error.
- **Wait** fixes the start. It barely touches the end.
- **See** fixes where learning ends up. It barely touches the start.

All three are about responsibility. A layer answers for its own prediction, gets a say only when it has earned it, and can see what the layers below have already worked out. The signals that failed all broke the first rule.

## Away from home

A trick found on one stream may only describe that stream. So the same five stacks, with every setting unchanged, read three new streams of 180,000 bytes: clean web text (C4), the King James Bible, and handwritten digits (MNIST) read as a stream of pixel bytes, row after row.

On both texts all three tricks held, on both seeds. Own: six layers beat one by 0.15 on web text and 0.17 on the Bible. Wait: 0.69 and 0.62 saved over the whole stream. See: 0.05 and 0.06.

On the digits only Wait held (0.72 saved over the whole stream). With the gate open from the start, six layers barely beat one even late in the stream (0.008). With the gate at zero they beat it by 0.06, so on images Wait does more than fix the start. And seeing every lower layer added only 0.005. A picture read as a line has its structure 28 bytes back, in the row above, and the first layer's memory already reaches that far; there is little for the layers above to share.

**Wait is general. Own and See held on every text, and not on images read as a line.**

What these runs do not show: whether the tricks hold on much longer streams or at much larger depth, what a layer needs to use the second dimension of an image, and whether some honest form of credit from above, not tried here, could add to them.

## Papers

- He, Zhang, Ren, Sun (2016). Deep residual learning for image recognition.
- Lee, Xie, Gallagher, Zhang, Tu (2015). Deeply-supervised nets.
- Goyal et al. (2017). Accurate, large minibatch SGD: training ImageNet in 1 hour.
- De, Smith (2020). Batch normalization biases residual blocks towards the identity function in deep networks.
- Bachlechner, Majumder, Mao, Cottrell, McAuley (2020). ReZero is all you need: fast convergence at large depth.
- Huang, Liu, van der Maaten, Weinberger (2017). Densely connected convolutional networks.
