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

import * as test_util from "../test_util";
import { MathTests } from "../test_util";

import { Array1D, Array2D } from "./ndarray";

// softmax
{
  const tests: MathTests = it => {
    it("regular test", math => {
      const y = math.softmax(Array1D.new([2, 1, 3]));

      test_util.expectArraysClose(y, [0.24472847, 0.09003057, 0.66524095]);
      test_util.expectNumbersClose(y.get(0) + y.get(1) + y.get(2), 1);
    });

    it("overflow", math => {
      const y = math.softmax(Array1D.new([1000, 1000]));

      test_util.expectArraysClose(y, [0.5, 0.5]);
    });

    it("underflow", math => {
      const y = math.softmax(Array1D.new([-1000, -1000]));

      test_util.expectArraysClose(y, [0.5, 0.5]);
    });

    it("Huge difference between probabilities", math => {
      const y = math.softmax(Array1D.new([-1000, +1000]));

      test_util.expectArraysClose(y, [0, 1]);
    });

    it("Propagates NaNs", math => {
      const a = Array1D.new([2, 1, NaN]);
      const y = math.softmax(a);
      test_util.expectArraysClose(y, [NaN, NaN, NaN]);
    });

    it("2D, dim=1", math => {
      const y = math.softmax(Array2D.new([2, 3], [[2, 1, 3], [1, 3, 2]]), 1);
      const expected = [
        0.24472847, 0.09003057, 0.66524095, 0.09003057, 0.66524095, 0.24472847
      ];
      expect(y.rank).toBe(2);
      test_util.expectArraysClose(y, expected);
    });

    it("2D, implicit dim=1", math => {
      const y = math.softmax(Array2D.new([2, 3], [[2, 1, 3], [1, 3, 2]]));
      const expected = [
        0.24472847, 0.09003057, 0.66524095, 0.09003057, 0.66524095, 0.24472847
      ];
      expect(y.rank).toBe(2);
      test_util.expectArraysClose(y, expected);
    });

    it("2D, dim=0 throws error", math => {
      const f = () => {
        math.softmax(Array2D.new([2, 3], [[2, 1, 3], [1, 3, 2]]), 0);
      };
      expect(f).toThrowError();
    });
  };

  test_util.describeMathCPU("softmax", [tests]);
  test_util.describeMathGPU("softmax", [tests], [
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 2, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": false}
  ]);
}
{
  const tests: MathTests = it => {
    it("1D", math => {
      const logits = Array1D.new([1, 2, 3]);
      const label = Array1D.new([0.3, 0.6, 0.1]);
      const softmaxLogits = math.softmax(logits);

      const y = math.softmaxCrossEntropyWithLogits(label, logits);

      expect(y.shape).toEqual([]);
      test_util.expectNumbersClose(
          y.get(),
          -Math.log(softmaxLogits.get(0)) * label.get(0) +
              -Math.log(softmaxLogits.get(1)) * label.get(1) +
              -Math.log(softmaxLogits.get(2)) * label.get(2));
    });

    it("2D implicit dim", math => {
      const logits = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
      const label = Array2D.new([2, 3], [0.3, 0.6, 0.1, 0.2, 0.3, 0.5]);
      const softmaxLogits = math.softmax(logits);

      const y = math.softmaxCrossEntropyWithLogits(label, logits);

      expect(y.shape).toEqual([2]);
      test_util.expectArraysClose(y, [
        -Math.log(softmaxLogits.get(0, 0)) * label.get(0, 0) +
            -Math.log(softmaxLogits.get(0, 1)) * label.get(0, 1) +
            -Math.log(softmaxLogits.get(0, 2)) * label.get(0, 2),
        -Math.log(softmaxLogits.get(1, 0)) * label.get(1, 0) +
            -Math.log(softmaxLogits.get(1, 1)) * label.get(1, 1) +
            -Math.log(softmaxLogits.get(1, 2)) * label.get(1, 2)
      ]);
    });

    it("2D, dim=1", math => {
      const logits = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
      const label = Array2D.new([2, 3], [0.3, 0.6, 0.1, 0.2, 0.3, 0.5]);
      const dim = 1;
      const softmaxLogits = math.softmax(logits, dim);

      const y = math.softmaxCrossEntropyWithLogits(label, logits, dim);

      expect(y.shape).toEqual([2]);
      test_util.expectArraysClose(y, [
        -Math.log(softmaxLogits.get(0, 0)) * label.get(0, 0) +
            -Math.log(softmaxLogits.get(0, 1)) * label.get(0, 1) +
            -Math.log(softmaxLogits.get(0, 2)) * label.get(0, 2),
        -Math.log(softmaxLogits.get(1, 0)) * label.get(1, 0) +
            -Math.log(softmaxLogits.get(1, 1)) * label.get(1, 1) +
            -Math.log(softmaxLogits.get(1, 2)) * label.get(1, 2)
      ]);
    });

    it("2D, dim=0 throws error", math => {
      const logits = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
      const label = Array2D.new([2, 3], [0.3, 0.6, 0.1, 0.2, 0.3, 0.5]);
      const dim = 0;

      expect(() => math.softmaxCrossEntropyWithLogits(label, logits, dim))
          .toThrowError();
    });

    it("Propagates NaNs", math => {
      const logits = Array1D.new([1, 2, NaN]);
      const label = Array1D.new([0.3, 0.6, 0.1]);

      const y = math.softmaxCrossEntropyWithLogits(label, logits);

      expect(y.shape).toEqual([]);
      test_util.expectArraysClose(y, [NaN]);
    });
  };

  test_util.describeMathCPU("softmaxCrossEntropyWithLogits", [tests]);
  test_util.describeMathGPU("softmaxCrossEntropyWithLogits", [tests], [
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 2, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": false}
  ]);
}
