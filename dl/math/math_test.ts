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

import { Array1D, Array3D, Scalar } from "./ndarray";

// math.scope
{
  const gpuTests: MathTests = it => {
    it("scope returns NDArray", async math => {
      math.scope(() => {
        const a = Array1D.new([1, 2, 3]);
        let b = Array1D.new([0, 0, 0]);

        expect(math.getNumArrays()).toBe(2);
        math.scope(() => {
          const result = math.scope(() => {
            b = math.addStrict(a, b);
            b = math.addStrict(a, b);
            b = math.addStrict(a, b);
            return math.add(a, b);
          });

          // result is new. All intermediates should be disposed.
          expect(math.getNumArrays()).toBe(2 + 1);
          test_util.expectArraysClose(result, [4, 8, 12]);
        });

        // a, b are still here, result should be disposed.
        expect(math.getNumArrays()).toBe(2);
      });

      expect(math.getNumArrays()).toBe(0);
    });

    it("multiple disposes does not affect num arrays", math => {
      expect(math.getNumArrays()).toBe(0);
      const a = Array1D.new([1, 2, 3]);
      const b = Array1D.new([1, 2, 3]);
      expect(math.getNumArrays()).toBe(2);
      a.dispose();
      a.dispose();
      expect(math.getNumArrays()).toBe(1);
      b.dispose();
      expect(math.getNumArrays()).toBe(0);
    });

    it("scope returns NDArray[]", async math => {
      const a = Array1D.new([1, 2, 3]);
      const b = Array1D.new([0, -1, 1]);
      expect(math.getNumArrays()).toBe(2);

      math.scope(() => {
        const result = math.scope(() => {
          math.add(a, b);
          return [math.add(a, b), math.subtract(a, b)];
        });

        // the 2 results are new. All intermediates should be disposed.
        expect(math.getNumArrays()).toBe(4);
        test_util.expectArraysClose(result[0], [1, 1, 4]);
        test_util.expectArraysClose(result[1], [1, 3, 2]);
        expect(math.getNumArrays()).toBe(4);
      });

      // the 2 results should be disposed.
      expect(math.getNumArrays()).toBe(2);
      a.dispose();
      b.dispose();
      expect(math.getNumArrays()).toBe(0);
    });

    it("basic scope usage without return", math => {
      const a = Array1D.new([1, 2, 3]);
      let b = Array1D.new([0, 0, 0]);

      expect(math.getNumArrays()).toBe(2);

      math.scope(() => {
        b = math.addStrict(a, b);
        b = math.addStrict(a, b);
        b = math.addStrict(a, b);
        math.add(a, b);
      });

      // all intermediates should be disposed.
      expect(math.getNumArrays()).toBe(2);
    });

    it("scope returns Promise<NDArray>", async math => {
      const a = Array1D.new([1, 2, 3]);
      const b = Array1D.new([0, 0, 0]);

      expect(math.getNumArrays()).toBe(2);

      math.scope(() => {
        const result = math.scope(() => {
          let c = math.add(a, b);
          c = math.add(a, c);
          c = math.add(a, c);
          return math.add(a, c);
        });

        // result is new. All intermediates should be disposed.
        expect(math.getNumArrays()).toBe(3);
        test_util.expectArraysClose(result, [4, 8, 12]);
      });

      // result should be disposed. a and b are still allocated.
      expect(math.getNumArrays()).toBe(2);
      a.dispose();
      b.dispose();
      expect(math.getNumArrays()).toBe(0);
    });

    it("nested scope usage", async math => {
      const a = Array1D.new([1, 2, 3]);
      let b = Array1D.new([0, 0, 0]);

      expect(math.getNumArrays()).toBe(2);

      math.scope(() => {
        const result = math.scope(() => {
          b = math.addStrict(a, b);
          b = math.scope(() => {
            b = math.scope(() => {
              return math.addStrict(a, b);
            });
            // original a, b, and two intermediates.
            expect(math.getNumArrays()).toBe(4);

            math.scope(() => {
              math.addStrict(a, b);
            });
            // All the intermediates should be cleaned up.
            expect(math.getNumArrays()).toBe(4);

            return math.addStrict(a, b);
          });
          expect(math.getNumArrays()).toBe(4);

          return math.addStrict(a, b);
        });

        expect(math.getNumArrays()).toBe(3);
        test_util.expectArraysClose(result, [4, 8, 12]);
      });
      expect(math.getNumArrays()).toBe(2);
    });
  };

  test_util.describeMathGPU("scope", [gpuTests], [
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 2, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": false}
  ]);
}

// fromPixels & math
{
  const tests: MathTests = it => {
    it("debug mode does not error when no nans", math => {
      const pixels = new ImageData(2, 2);
      for (let i = 0; i < 8; i++) {
        pixels.data[i] = 100;
      }
      for (let i = 8; i < 16; i++) {
        pixels.data[i] = 250;
      }

      const a = Array3D.fromPixels(pixels, 4);
      const b = Scalar.new(20, "int32");

      const res = math.add(a, b);

      test_util.expectArraysEqual(res, [
        120, 120, 120, 120, 120, 120, 120, 120, 270, 270, 270, 270, 270, 270,
        270, 270
      ]);
    });
  };

  test_util.describeMathGPU("fromPixels + math", [tests], [
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 2, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": false}
  ]);
}
