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

// Test our jasmine_shim with our own test framework.

import { assert } from "../src/util";
import { matchTesters } from "./jasmine_shim";
import { test } from "./tester";

test(async function test_matchTesters() {
  assert(matchTesters.toEqual("world", "world"));
  assert(!matchTesters.toEqual("hello", "world"));
  assert(matchTesters.toEqual(5, 5));
  assert(!matchTesters.toEqual(5, 6));
  assert(matchTesters.toEqual(NaN, NaN));
});
