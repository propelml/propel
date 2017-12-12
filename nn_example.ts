// A multi-layer perceptron for classification of MNIST handwritten digits.
// Adapted from
// https://github.com/HIPS/autograd/blob/master/examples/neural_net.py
import { $, multigradAndVal, randn, Tensor } from "./api";
import * as mnist from "./mnist";
import { assert, assertShapesEqual } from "./util";

interface LayerParams {
  weight: Tensor;
  bias: Tensor;
}
type Params = LayerParams[];

// Build a list of [weights, biases] pairs, one for each layer in the net.
function initRandomParams(scale: number, layerSizes: number[]): Params {
  const params: Params = [];
  for (let i = 0; i < layerSizes.length - 1; ++i) {
    const m = layerSizes[i];
    const n = layerSizes[i + 1];
    params.push({
      bias: randn(n).mul(scale),
      weight: randn(m, n).mul(scale),
    });
  }
  return params;
}

function flattenParams(params: Params): Tensor[] {
  const paramList: Tensor[] = [];
  for (const layerParams of params) {
    paramList.push(layerParams.weight);
    paramList.push(layerParams.bias);
  }
  return paramList;
}

function unflattenParams(paramList: Tensor[]): Params {
  const params: Params = [];
  for (let i = 0; i < paramList.length - 1; i += 2) {
    const weight = paramList[i];
    const bias = paramList[i + 1];
    params.push({bias, weight});
  }
  return params;
}

// Implements a deep neural network for classification.  params is a list of
// (weights, bias) tuples.  inputs is an (N x D) matrix.  returns normalized
// class log-probabilities.
function neuralNetPredict(params: Params, inputs) {
  let outputs;
  for (const {weight, bias} of params) {
    outputs = inputs.matmul(weight).add(bias);
    inputs = outputs.tanh();
  }
  return outputs.sub(outputs.reduceLogSumExp([1], true));
}

// Computes l2 norm of params by flattening them into a vector.
function l2Norm(params: Params): Tensor {
  let norm = $(0);
  for (const {weight, bias} of params) {
    const weight_ = weight.flatten(); // make weight a vector.
    norm = norm.add(weight_.dot(weight_));
    norm = norm.add(bias.dot(bias));
  }
  assert(norm.rank === 0);
  return norm;
}

function negLogPosterior(params, inputs, targets, l2Reg) {
  const reg = l2Norm(params).mul(l2Reg);
  const nll = neuralNetPredict(params, inputs).mul(targets).reduceSum().neg();
  return nll.add(reg);
}

function accuracy(params, inputs, targets) {
  const targetClass = targets.argmax(1);
  const predictedClass = neuralNetPredict(params, inputs).argmax(1);
  return predictedClass.equals(targetClass).mean();
}

interface ArgsSGD {
  objectiveGrad: (...params: Tensor[]) => [Tensor[], Tensor];
  initParams: Params;
  callback: (...args: any[]) => void;
  learningRate: number;
  steps: number;
  momentum: number;
}

// Stochastic gradient descent with momentum.
function sgd(args: ArgsSGD) {
  const params = flattenParams(args.initParams);
  const velocity = params.map((p) => p.zerosLike());
  const m = args.momentum;
  assert(0 <= m && m <= 1.0);
  // Training loop.
  for (let step = 0; step < args.steps; step++) {
    // Forward/Backward pass
    const [pGrads, loss] = args.objectiveGrad(...params);
    assert(loss.rank === 0);
    assert(pGrads.length === params.length);
    // Update each param tensor.
    for (let j = 0; j < params.length; j++) {
      assertShapesEqual(params[j].shape, pGrads[j].shape);
      velocity[j] = velocity[j].mul(m).sub(pGrads[j].mul(1 - m));
      params[j] = params[j].add(velocity[j].mul(args.learningRate));
    }
    if (args.callback) args.callback(step, loss);
  }
  return params;
}

function main() {
  // Model parameters
  const layerSizes = [784, 200, 100, 10];
  const l2Reg = 1.0;

  // Training parameters
  const batchSize = 256;

  const initParams = initRandomParams(0.1, layerSizes);

  // Fake it til you make it.
  console.log("Loading training data...");
  const dataset = mnist.load("train", batchSize);

  // Define training objective
  const objective = (...paramList: Tensor[]) => {
    let [images, labels] = dataset.next();
    images = images.cast("float32").div(255).reshape([-1, 28 * 28]);
    labels = labels.oneHot(10);
    assertShapesEqual(images.shape, [batchSize, 784]);
    assertShapesEqual(labels.shape, [batchSize, 10]);
    const params = unflattenParams(paramList);
    const l = negLogPosterior(params, images, labels, l2Reg);
    assert(l.rank === 0);
    return l;
  };

  // Get gradient of objective using autograd.
  const objectiveGrad = multigradAndVal(objective, [0, 1, 2, 3]);

  const printPerf = (step, loss) => {
    // const trainAcc = accuracy(params, trainImages, trainLabels);
    if (step % 50 === 0) {
      console.log("step", step, "loss", loss.getData()[0]);
    }
  };

  // The optimizers provided can optimize lists, tuples, or dicts of parameters.
  const optimizedParams = sgd({
    callback: printPerf,
    initParams,
    learningRate: 0.001,
    momentum: 0.9,
    objectiveGrad,
    steps: 2000,
  });
}

main();
