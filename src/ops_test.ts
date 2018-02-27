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
import { test } from "../tools/tester";
import { linspace } from "./api";
import * as ops from "./ops";
import { assertAllClose } from "./util";

// Basic Tests

test(async function api_linspace() {
  const x = linspace(-4, 4, 6);
  const y = linspace(-7, 3, 6);

  const result = ops.add_(x, y);
  console.log("...............................................");
  console.log(x);
  console.log(result);
  console.log("...............................................");
  assertAllClose(x, [-4., -2.4, -0.8,  0.8,  2.4, 4.]);
});
