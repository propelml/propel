// Simple MNIST classifier.
// Adapted from
// https://github.com/HIPS/autograd/blob/master/examples/neural_net.py
import { $, OptimizerSGD, Params, Tensor } from "./api";
import * as mnist from "./mnist";

let device;  // Set in train()

// Hyperparameters
const learningRate = 0.001;
const momentum = 0.9;
const batchSize = 256;
const layerSizes = [784, 200, 100, 10];
const reg = 0.0001;

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
    const w = params.randn(`w${i}`, [m, n], {device});
    const b = params.randn(`b${i}`, [n], {device});
    outputs = inputs.matmul(w).add(b);
    inputs = outputs.relu();
  }
  return outputs;
}

// Define the training objective using softmax cross entropy loss.
function loss(images, labels, params: Params): Tensor {
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

async function accuracy(params: Params, dataset,
                        nExamples = 500): Promise<number> {
  let totalCorrect = $(0);
  let seen = 0;
  while (seen < nExamples) {
    const {images, labels} = await dataset.next();
    const logits = inference(params, images);
    const predicted = logits.argmax(1).cast("int32");
    const a = predicted.equal(labels).cast("float32").reduceSum();
    totalCorrect = totalCorrect.add(a);
    seen += images.shape[0];
  }
  const acc = totalCorrect.div(seen);
  return acc.getData()[0];
}

export async function train(useGPU = false, maxSteps = 10000) {
  device = useGPU ? "GPU:0" : "CPU:0";
  console.log("Load MNIST...");
  const datasetTrain = mnist.load("train", batchSize, useGPU);
  const datasetTest = mnist.load("test", batchSize, useGPU);

  const opt = new OptimizerSGD();
  while (opt.steps < maxSteps) {
    const {images, labels} = await datasetTrain.next();

    // Take a step of SGD. Update the parameters opt.params.
    const l = opt.step(learningRate, momentum, (params: Params) => {
      return loss(images, labels, params);
    });

    if (opt.steps % 100 === 0) {
      const trainAcc = await accuracy(opt.params, datasetTrain, 2 * batchSize);
      console.log("step", opt.steps,
                  "loss", l.toFixed(3),
                  "train accuracy", (100 * trainAcc).toFixed(1));
    }

    if (opt.steps % 1000 === 0) {
      const testAcc = await accuracy(opt.params, datasetTest, 10 * batchSize);
      console.log("test accuracy", (100 * testAcc).toFixed(1));
    }
  }
}
