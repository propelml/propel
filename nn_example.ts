// Simple MNIST classifier.
// Adapted from
// https://github.com/HIPS/autograd/blob/master/examples/neural_net.py
import { $, Params, sgd, Tensor } from "./api";
import * as mnist from "./mnist";

// Hyperparameters
const learningRate = 0.001;
const batchSize = 256;
const layerSizes = [784, 200, 100, 10];
const reg = 0.0001;

console.log("Load MNIST...");
const datasetTrain = mnist.load("train", batchSize);
const datasetTest = mnist.load("test", batchSize);

// Implements a fully-connected network with ReLU activations.
// Returns logits.
// @param params A list of parameters.
// @param images An (N x 28 x 28) tensor.
function inference(params: Params, images: Tensor) {
  let inputs = images.cast("float32").div(255).reshape([-1, 28 * 28]);
  let outputs;
  for (let i = 0; i < layerSizes.length - 1; ++i) {
    const m = layerSizes[i];
    const n = layerSizes[i + 1];
    // Initialize or get weights and biases.
    const w = params.randn(`w${i}`, [m, n]);
    const b = params.randn(`b${i}`, [n]);
    outputs = inputs.matmul(w).add(b);
    inputs = outputs.relu();
  }
  return outputs;
}

// Define the training objective using softmax cross entropy loss.
function loss(params: Params): Tensor {
  const [images, labels] = datasetTrain.next();
  const labels1H = labels.oneHot(10);
  const logits = inference(params, images);
  const softmaxLoss = logits.softmaxCE(labels1H).reduceMean();
  return softmaxLoss.add(regLoss(params));
}

// Regularization loss. Computes L2 norm of all the params scaled by reg.
function regLoss(params: Params): Tensor {
  let s = $(0);
  params.forEach((p) => {
    s = s.add(p.square().reduceSum());
  });
  return s.mul(reg);
}

function accuracy(params: Params, dataset, nExamples = 500): number {
  let totalCorrect = $(0);
  let seen = 0;
  while (seen < nExamples) {
    const [images, labels] = dataset.next();

    const logits = inference(params, images);
    const predicted = logits.argmax(1).cast("uint8");
    const a = predicted.equal(labels).cast("float32").reduceSum();
    totalCorrect = totalCorrect.add(a);
    seen += images.shape[0];
  }
  const acc = totalCorrect.div(seen);
  return acc.getData()[0];
}

sgd({
  callback: (step, loss, params) => {
    if (step % 100 === 0) {
      const trainAcc = accuracy(params, datasetTrain, 2 * batchSize);
      console.log("step", step,
                  "loss", loss.toFixed(3),
                  "train accuracy", (100 * trainAcc).toFixed(1));
    }
    if (step % 1000 === 0) {
      const testAcc = accuracy(params, datasetTest, 10 * batchSize);
      console.log("test accuracy", (100 * testAcc).toFixed(1));
    }
  },
  learningRate,
  lossFn: loss,
  momentum: 0.9,
  steps: 10000,
});
