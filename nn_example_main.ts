// Use this to run nn_example on node.
import { listDevices } from "./api";
import { Trainer } from "./nn_example";

const useGPU = (listDevices().length > 1);
async function main() {
  const trainer = new Trainer(useGPU);
  while (trainer.opt.steps < 10000) {
    await trainer.step();
  }
}

main();
