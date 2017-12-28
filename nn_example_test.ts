import { listDevices } from "./api";
import { train } from "./nn_example";

function testSmoke() {
  const useGPU = (listDevices().length > 1);
  train(useGPU, 2);  // Train for two steps.
}

testSmoke();
