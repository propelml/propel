import { listDevices } from "./api";
import { Trainer } from "./nn_example";

function testSmoke() {
  const useGPU = (listDevices().length > 1);
  const trainer = new Trainer(useGPU);
  trainer.step();
  trainer.step();
}

testSmoke();
