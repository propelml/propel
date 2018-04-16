// Script for training cifar10.
import * as pr from "../src/api";
import { IS_NODE } from "../src/util";
import * as resnet from "./resnet";

export async function train(expName: string): Promise<void> {
  const ds = pr.dataset("cifar10/train").batch(16).repeat(100);
  const exp = await pr.experiment(expName);
  const n = 10;  // Number of resnet blocks.
  for (const batchPromise of ds) {
    const { images, labels } = await batchPromise;
    exp.sgd({ lr: 0.01 }, (params) => {
      const x = images.rescale([0, 255], [-1, 1]);
      const logits = resnet.resnetSimple(x, params, n);
      return resnet.loss(params, logits, labels);
    });
  }
}

if (IS_NODE && require.main === module) {
  train(process.argv[2] || "resnet_cifar10_001");
}
