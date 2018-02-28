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
import { assert } from "../src/util";
import * as gendoc from "./gendoc";
import { test } from "./tester";

test(async function gendoc_smoke() {
  const docs = gendoc.genJSON();
  console.log("length", docs.length);
  assert(docs.length > 5);
  const names = docs.map(e => e.name);
  // Check that a few of the names are correct.
  assert(names.indexOf("Tensor") >= 0);
  assert(names.indexOf("Tensor.add") >= 0);
  assert(names.indexOf("Tensor[Symbol.iterator]") >= 0);
});
