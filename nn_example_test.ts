import { listDevices } from "./api";
import { Trainer } from "./nn_example";
import { test } from "./test";

test(async function testNNExample() {
  const useGPU = (listDevices().length > 1);
  const trainer = new Trainer(useGPU);
  await trainer.step();
  await trainer.step();
});
