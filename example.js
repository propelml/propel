const { dataset, experiment } = require("propel");

async function train(maxSteps) {
  // Load mnist asynchronously, with a batch size of 128 and
  // repeat the dataset for 100 epochs.
  const ds = dataset("mnist/train").batch(128).repeat(100);
  // Create or restore an experiment called exp001.
  // experiments manage checkpoints and logs. It's stored at
  // $HOME/.propel/exp001
  const exp = await experiment("exp001");
  // Loop over the elements of the dataset.
  // Alternatively use for await () here.
  for (const batchPromise of ds) {
    const { images, labels } = await batchPromise;
    // Perform an SGD step on the current parameters, with a learning rate of
    // 0.01. This bit is the most confusing part of the API, but it is
    // justified.
    // The callback given takes the current parameters (restored from disk, or
    // initialized from random) and must return the loss.
    exp.sgd({ lr: 0.01 }, (params) =>
      // Calculate and return the loss. This is the model definition.
      // The images tensor is [128, 28, 28] and uint8 dtype.
      // Before inputting it into the neural network, rescale the values
      // from [0, 255] to [-1, 1]. Zero centered tensors are usually the
      // easiest for the network to consume.
      images.rescale([0, 255], [-1, 1])
      // Apply three linear (densely connected) layers with a relu after each
      // except the last. The shape of the activations are:
      // [128, 28, 28] -> [128, 200] -> [128, 100] -> [128, 10] -> []
      // Note that the params object is explicitly passed to
      // every layer, and each layer is explicitly scoped. We contend that 
      // this explicitness makes for much saner models.
      .linear("L1", params, 200).relu()
      .linear("L2", params, 100).relu()
      .linear("L3", params, 10)
      // Using the logits, calculate a classification loss between the
      // labels. Labels's shape is [128]. This value is returned, and will be
      // backpropagated thru.
      // Note the final loss is a scalar.
      .softmaxLoss(labels));

    // Stop after maxSteps.
    // Note the step counter is stored with the experiment.
    if (maxSteps && exp.step >= maxSteps) break;
  }
}

train(3000);

