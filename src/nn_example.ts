/*!
   Copyright 2018 Propel http://propel.site/.  All rights reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */
// Simple MNIST classifier.
// Adapted from
// https://github.com/HIPS/autograd/blob/master/examples/neural_net.py
import { OptimizerSGD, Params, T, Tensor } from "./api";
import * as plt from "./matplotlib";
import * as mnist from "./mnist";
import { assert, IS_WEB } from "./util";

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
    assert(w.device === device);
    assert(b.device === device);
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
  let s: number | Tensor = 0;
  params.forEach(p => {
    s = p.square().reduceSum().add(s);
  });
  return T(s).mul(reg);
}

async function accuracy(params: Params, dataset,
                        nExamples = 500): Promise<number> {
  let totalCorrect = T(0);
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

export class Trainer {
  datasetTrain: any;
  datasetTest: any;
  opt: OptimizerSGD;
  durationSum = 0;
  durationCount = 0;
  lossSum = 0;
  lossCount = 0;

  constructor(useGPU = false) {
    device = useGPU ? "GPU:0" : "CPU:0";
    this.datasetTrain = mnist.load("train", batchSize, useGPU);
    this.datasetTest = mnist.load("test", batchSize, useGPU);
    this.opt = new OptimizerSGD();

    if (IS_WEB) {
      plt.register(() => {
        return document.querySelector("#output");
      });
    }
  }

  readonly lossSteps: number[] = [];
  readonly lossValues: number[] = [];

  async step() {
    const {images, labels} = await this.datasetTrain.next();
    const start = new Date();

    // Take a step of SGD. Update the parameters opt.params.
    const l = this.opt.step(learningRate, momentum, (params: Params) => {
      return loss(images, labels, params);
    });

    this.lossSum += l;
    this.lossCount += 1;

    this.durationSum += ((new Date()).getTime() - start.getTime());
    this.durationCount += 1;

    // Check that the params are on the right device.
    this.opt.params.forEach((t, name) => {
      assert(t.device === device, `param ${name} is on device ${t.device}`);
    });

    if (this.opt.steps % 10 === 0) {
      const lossAvg = this.lossSum / this.lossCount;
      this.lossSum = this.lossCount = 0;
      this.lossSteps.push(this.opt.steps);
      this.lossValues.push(lossAvg);
      const x = T(this.lossSteps);
      const y = T(this.lossValues);
      plt.plot(x, y);
    }

    if (this.opt.steps % 100 === 0) {
      const rate = this.durationCount / (this.durationSum / 1000);
      this.durationSum = this.durationCount = 0;

      const trainAcc = await accuracy(this.opt.params, this.datasetTrain,
                                      2 * batchSize);
      console.log("step", this.opt.steps,
                  "loss", l.toFixed(3),
                  "train accuracy", (100 * trainAcc).toFixed(1),
                  "steps/sec", rate.toFixed(1));
    }

    if (this.opt.steps % 1000 === 0) {
      const testAcc = await accuracy(this.opt.params, this.datasetTest,
                                     10 * batchSize);
      console.log("test accuracy", (100 * testAcc).toFixed(1));
    }
  }
}
