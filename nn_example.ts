// Simple MNIST classifier.
// Adapted from
// https://github.com/HIPS/autograd/blob/master/examples/neural_net.py
import { $, randn, sgd, Tensor } from "./api";
import * as mnist from "./mnist";

// Hyperparameters
const learningRate = 0.001;
const batchSize = 256;
const layerSizes = [784, 200, 100, 10];
const l2Reg = 0.0001;

// Build a list of [weights, biases] pairs, one for each layer in the net.
function initRandomParams(scale: number, layerSizes: number[]): Tensor[] {
  const params: Tensor[] = [];
  for (let i = 0; i < layerSizes.length - 1; ++i) {
    const m = layerSizes[i];
    const n = layerSizes[i + 1];
    params.push(randn([m, n]).mul(scale)); // weight
    params.push(randn([n]).mul(scale)); // bias
  }
  return params;
}

// Implements a fully-connected network with ReLU activations.
// Returns logits.
// @param params A list of parameters.
// @param images An (N x 28 x 28) tensor.
function inference(params: Tensor[], images: Tensor) {
  let inputs = images.cast("float32").div(255).reshape([-1, 28 * 28]);
  let outputs;
  for (let i = 0; i < params.length; i += 2) {
    const weight = params[i];
    const bias = params[i + 1];
    outputs = inputs.matmul(weight).add(bias);
    inputs = outputs.relu();
  }
  return outputs;
}

// Computes L2 norm of all the params.
function l2Norm(params: Tensor[]): Tensor {
  let s = $(0);
  for (const p of params) {
    s = s.add(p.square().reduceSum());
  }
  return s;
}

function accuracy(params: Tensor[], dataset, nExamples = 500): number {
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

function main() {
  console.log("Load MNIST...");
  const datasetTrain = mnist.load("train", batchSize);
  const datasetTest = mnist.load("test", batchSize);

  // Define training objective
  const lossFn = (...params: Tensor[]) => {
    const [images, labels] = datasetTrain.next();
    const labels1H = labels.oneHot(10);
    const regularizationLoss = l2Norm(params).mul(l2Reg);
    const logits = inference(params, images);
    const softmaxLoss = logits.softmaxCE(labels1H).reduceMean();
    return softmaxLoss.add(regularizationLoss);
  };

  const printPerf = (step, params, loss) => {
    if (step === 0) return;
    if (step % 100 === 0) {
      const trainAcc = accuracy(params, datasetTrain, 2 * batchSize);
      console.log("step", step,
                  "loss", loss.getData()[0].toFixed(3),
                  "train accuracy", (100 * trainAcc).toFixed(1));
    }
    if (step % 1000 === 0) {
      const testAcc = accuracy(params, datasetTest, 10 * batchSize);
      console.log("test accuracy", (100 * testAcc).toFixed(1));
    }
  };

  sgd({
    callback: printPerf,
    learningRate,
    lossFn,
    momentum: 0.9,
    params: initRandomParams(0.1, layerSizes),
    steps: 10000,
  });
}

main();
