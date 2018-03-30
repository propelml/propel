// Run this program from the command-line using ts-node
//
//   npm install -g ts-node
//   ts-node example.ts
//
import * as pr from "./api";
import { IS_NODE } from "./util";

export async function train(maxSteps = 3000) {
  const ds = pr.dataset("mnist/train").batch(128).repeat(100);
  const exp = await pr.experiment("exp001");
  for (const batchPromise of ds) {
    const { images, labels } = await batchPromise;
    exp.sgd({ lr: 0.01 }, (params) =>
      images.rescale([0, 255], [-1, 1])
        .linear("L1", params, 200).relu()
        .linear("L2", params, 100).relu()
        .linear("L3", params, 10)
        .softmaxLoss(labels));
    if (maxSteps && exp.step >= maxSteps) break;
  }
}

if (IS_NODE && require.main === module) {
  train(Number(process.argv[2]));
}
