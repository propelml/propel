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
