/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as test_util from "../../test_util";
import { MathTests } from "../../test_util";
import { Scalar } from "../ndarray";

import { extractNDArraysFromScopeResult } from "./backend_engine";

// extractNDArraysFromScopeResult
{
  const tests: MathTests = it => {
    it("null input returns empty array", math => {
      const results = extractNDArraysFromScopeResult(null);

      expect(results).toEqual([]);
    });

    it("ndarray input returns one element array", math => {
      const x = Scalar.new(1);
      const results = extractNDArraysFromScopeResult(x);

      expect(results).toEqual([x]);
    });

    it("name array map returns flattened array", math => {
      const x1 = Scalar.new(1);
      const x2 = Scalar.new(3);
      const x3 = Scalar.new(4);
      const results = extractNDArraysFromScopeResult({x1, x2, x3});

      expect(results).toEqual([x1, x2, x3]);
    });
  };

  test_util.describeMathCPU(
      "tape_util.extractNDArraysFromScopeResult", [tests]);
}
