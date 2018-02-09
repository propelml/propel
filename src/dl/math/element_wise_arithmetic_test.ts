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
import * as util from "../util";
import { Array1D, Array2D, Array3D, Scalar } from "./ndarray";

// divide
{
  const tests: MathTests = it => {
    it("divide", math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
      const c = Array2D.new([2, 3], [1, 2, 3, 4, 2, 5]);

      const r = math.divide(a, c);

      test_util.expectArraysClose(r, [1, 1, 1, 1, 2.5, 6 / 5]);
    });

    it("divide propagates NaNs", math => {
      const a = Array2D.new([2, 1], [1, 2]);
      const c = Array2D.new([2, 1], [3, NaN]);

      const r = math.divide(a, c);

      test_util.expectArraysClose(r, [1 / 3, NaN]);
    });

    it("divide broadcasting same rank NDArrays different shape", math => {
      const a = Array2D.new([2, 2], [1, 2, -3, -4]);
      const b = Array2D.new([2, 1], [2, 3]);

      const result = math.divide(a, b);

      expect(result.shape).toEqual([2, 2]);
      const expected = [1 / 2, 1, -1, -4 / 3];

      test_util.expectArraysClose(result, expected);
    });

    it("divide broadcast 2D + 1D", math => {
      const a = Array2D.new([2, 2], [1, 2, -3, -4]);
      const b = Array1D.new([1, 2]);

      const result = math.divide(a, b);

      expect(result.shape).toEqual([2, 2]);
      const expected = [1, 1, -3, -2];

      test_util.expectArraysClose(result, expected);
    });

    it("div throws when passed ndarrays of different shapes", math => {
      const a = Array2D.new([2, 3], [1, 2, -3, -4, 5, 6]);
      const b = Array2D.new([2, 2], [5, 3, 4, -7]);

      expect(() => math.divide(a, b)).toThrowError();
      expect(() => math.divide(b, a)).toThrowError();
    });

    it("scalar divided by array", math => {
      const c = Scalar.new(2);
      const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);

      const r = math.scalarDividedByArray(c, a);

      test_util.expectArraysClose(
          r, [2 / 1, 2 / 2, 2 / 3, 2 / 4, 2 / 5, 2 / 6]);
    });

    it("scalar divided by array propagates NaNs", math => {
      const c = Scalar.new(NaN);
      const a = Array2D.new([1, 3], [1, 2, 3]);

      const r = math.scalarDividedByArray(c, a);

      test_util.expectArraysEqual(r, [NaN, NaN, NaN]);
    });

    it("scalar divided by array throws when passed non scalar", math => {
      // tslint:disable-next-line:no-any
      const c: any = Array1D.new([1, 2, 3]);
      const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);

      expect(() => math.scalarDividedByArray(c, a)).toThrowError();
    });

    it("array divided by scalar", math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
      const c = Scalar.new(2);

      const r = math.arrayDividedByScalar(a, c);

      test_util.expectArraysClose(
          r, [1 / 2, 2 / 2, 3 / 2, 4 / 2, 5 / 2, 6 / 2]);
    });

    it("array divided by scalar propagates NaNs", math => {
      const a = Array2D.new([1, 3], [1, 2, NaN]);
      const c = Scalar.new(2);

      const r = math.arrayDividedByScalar(a, c);
      test_util.expectArraysClose(r, [1 / 2, 2 / 2, NaN]);
    });

    it("array divided by scalar throws when passed non scalar", math => {
      // tslint:disable-next-line:no-any
      const c: any = Array1D.new([1, 2, 3]);
      const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);

      expect(() => math.arrayDividedByScalar(a, c)).toThrowError();
    });

    it("scalar times ndarray", math => {
      const a = Array2D.new([3, 2], [2, -5, 1, 1, 4, 0]);
      const c = Scalar.new(2);

      const expected = [4, -10, 2, 2, 8, 0];
      const result = math.scalarTimesArray(c, a);

      expect(result.shape).toEqual([3, 2]);
      test_util.expectArraysClose(result, expected);
    });

    it("scalar times ndarray throws when passed non-scalar", math => {
      const a = Array2D.new([3, 2], [2, -5, 1, 1, 4, 0]);
      // tslint:disable-next-line:no-any
      const c: any = Array1D.new([1, 2, 3, 4]);

      expect(() => math.scalarTimesArray(c, a)).toThrowError();
    });
  };

  test_util.describeMathCPU("divide", [tests]);
  test_util.describeMathGPU("divide", [tests], [
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 2, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": false}
  ]);
}

// multiply
{
  const tests: MathTests = it => {
    it("elementWiseMul same-shaped ndarrays", math => {
      const a = Array2D.new([2, 2], [1, 2, -3, -4]);
      const b = Array2D.new([2, 2], [5, 3, 4, -7]);
      const expected = [5, 6, -12, 28];
      const result = math.elementWiseMul(a, b);

      expect(result.shape).toEqual([2, 2]);
      test_util.expectArraysClose(result, expected);
    });

    it("elementWiseMul propagates NaNs", math => {
      const a = Array2D.new([2, 2], [1, 3, 4, 0]);
      const b = Array2D.new([2, 2], [NaN, 3, NaN, 3]);

      const result = math.elementWiseMul(a, b);
      test_util.expectArraysClose(result, [NaN, 9, NaN, 0]);
    });

    it("elementWiseMul throws when passed ndarrays of different shapes",
       math => {
         const a = Array2D.new([2, 3], [1, 2, -3, -4, 5, 6]);
         const b = Array2D.new([2, 2], [5, 3, 4, -7]);

         expect(() => math.elementWiseMul(a, b)).toThrowError();
         expect(() => math.elementWiseMul(b, a)).toThrowError();
       });

    it("same-shaped ndarrays", math => {
      const a = Array2D.new([2, 2], [1, 2, -3, -4]);
      const b = Array2D.new([2, 2], [5, 3, 4, -7]);
      const expected = [5, 6, -12, 28];
      const result = math.multiply(a, b);

      expect(result.shape).toEqual([2, 2]);
      test_util.expectArraysClose(result, expected);
    });

    it("broadcasting ndarrays", math => {
      const a = Array2D.new([2, 2], [1, 2, -3, -4]);
      const b = Scalar.new(2);
      const expected = [2, 4, -6, -8];
      const result = math.multiply(a, b);

      expect(result.shape).toEqual([2, 2]);
      test_util.expectArraysClose(result, expected);
    });

    it("broadcasting same rank NDArrays different shape", math => {
      const a = Array2D.new([2, 2], [1, 2, -3, -4]);
      const b = Array2D.new([2, 1], [2, 3]);

      const result = math.multiply(a, b);

      expect(result.shape).toEqual([2, 2]);
      const expected = [2, 4, -9, -12];

      test_util.expectArraysClose(result, expected);
    });

    it("broadcast 2D + 1D", math => {
      const a = Array2D.new([2, 2], [1, 2, -3, -4]);
      const b = Array1D.new([1, 2]);

      const result = math.multiply(a, b);

      expect(result.shape).toEqual([2, 2]);
      const expected = [1, 4, -3, -8];

      test_util.expectArraysClose(result, expected);
    });
  };

  test_util.describeMathCPU("multiply", [tests]);
  test_util.describeMathGPU("multiply", [tests], [
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 2, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": false}
  ]);
}

// pow
{
  const tests: MathTests = it => {
    it("same-shaped ndarrays", math => {
      const a = Array2D.new([2, 3], [1, -2, -3, 0, 7, 1]);
      const b = Array2D.new([2, 3], [5, 3, 4, 5, 2, -3], "int32");
      const expected = [1, -8, 81, 0, 49, 1];
      const result = math.pow(a, b);

      expect(result.shape).toEqual([2, 3]);
      test_util.expectArraysClose(result, expected, 0.01);
    });

    it("int32^int32 returns int32", math => {
      const a = Array1D.new([1, 2, 3], "int32");
      const exp = Scalar.new(2, "int32");

      const result = math.pow(a, exp);

      expect(result.shape).toEqual([3]);
      expect(result.dtype).toBe("int32");
      test_util.expectArraysEqual(result, [1, 4, 9]);
    });

    it("different-shaped ndarrays", math => {
      const a = Array2D.new([2, 3], [1, -2, -3, 0, 7, 1]);
      const b = Scalar.new(2, "int32");
      const expected = [1, 4, 9, 0, 49, 1];
      const result = math.pow(a, b);

      expect(result.shape).toEqual([2, 3]);
      test_util.expectArraysClose(result, expected, 0.05);
    });

    it("propagates NaNs", math => {
      const a = Array2D.new([2, 2], [NaN, 3, NaN, 0]);
      const b = Array2D.new([2, 2], [1, 3, 2, 3], "int32");

      const result = math.pow(a, b);
      test_util.expectArraysClose(result, [NaN, 27, NaN, 0], 0.05);
    });

    it("throws when passed non int32 exponent param", math => {
      const a = Array2D.new([2, 3], [1, 2, -3, -4, 5, 6]);
      const b = Array2D.new([2, 2], [5, 3, 4, -7], "float32");

      // tslint:disable-next-line
      expect(() => math.pow(a, b as any)).toThrowError();
    });

    it("broadcasting same rank NDArrays different shape", math => {
      const a = Array2D.new([2, 2], [1, 2, -3, -4]);
      const b = Array2D.new([2, 1], [2, 1], "int32");

      const result = math.pow(a, b);

      expect(result.shape).toEqual([2, 2]);
      const expected = [1, 4, -3, -4];

      test_util.expectArraysClose(result, expected);
    });

    it("broadcast 2D + 1D", math => {
      const a = Array2D.new([2, 2], [1, 2, -3, -4]);
      const b = Array1D.new([1, 2], "int32");

      const result = math.pow(a, b);

      expect(result.shape).toEqual([2, 2]);
      const expected = [1, 4, -3, 16];

      test_util.expectArraysClose(result, expected);
    });

    it("powStrict same-shaped ndarrays", math => {
      const a = Array2D.new([2, 3], [1, -2, -3, 0, 7, 1]);
      const b = Array2D.new([2, 3], [5, 3, 4, 5, 2, -3], "int32");
      const expected = [1, -8, 81, 0, 49, 1];
      const result = math.powStrict(a, b);

      expect(result.shape).toEqual([2, 3]);
      test_util.expectArraysClose(result, expected, 0.01);
    });

    it("powStrict throws when passed ndarrays of different shapes", math => {
      const a = Array2D.new([2, 3], [1, 2, -3, -4, 5, 6]);
      const b = Array2D.new([2, 2], [5, 3, 4, -7], "int32");

      expect(() => math.powStrict(a, b)).toThrowError();
    });

    it("powStrict throws when passed non int32 exponent param", math => {
      const a = Array2D.new([2, 3], [1, 2, -3, -4, 5, 6]);
      const b = Array2D.new([2, 2], [5, 3, 4, -7], "float32");

      // tslint:disable-next-line
      expect(() => math.powStrict(a, b as any)).toThrowError();
    });
  };

  test_util.describeMathCPU("pow", [tests]);
  test_util.describeMathGPU("pow", [tests], [
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 2, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": false}
  ]);
}

// element-wise add / sub
{
  const tests: MathTests = it => {
    it("c + A", math => {
      const c = Scalar.new(5);
      const a = Array1D.new([1, 2, 3]);

      const result = math.scalarPlusArray(c, a);

      test_util.expectArraysClose(result, [6, 7, 8]);
    });

    it("c + A propagates NaNs", math => {
      const c = Scalar.new(NaN);
      const a = Array1D.new([1, 2, 3]);

      const res = math.scalarPlusArray(c, a);

      test_util.expectArraysEqual(res, [NaN, NaN, NaN]);
    });

    it("c + A throws when passed non scalar", math => {
      // tslint:disable-next-line:no-any
      const c: any = Array1D.new([1, 2, 3]);
      const a = Array1D.new([1, 2, 3]);

      expect(() => math.scalarPlusArray(c, a)).toThrowError();
    });

    it("A + B broadcasting same rank NDArrays different shape", math => {
      const a = Array2D.new([2, 2], [1, 2, -3, -4]);
      const b = Array2D.new([2, 1], [2, 3]);

      const result = math.add(a, b);

      expect(result.shape).toEqual([2, 2]);
      const expected = [3, 4, 0, -1];

      test_util.expectArraysClose(result, expected);
    });

    it("A + B broadcast 2D + 1D", math => {
      const a = Array2D.new([2, 2], [1, 2, -3, -4]);
      const b = Array1D.new([1, 2]);

      const result = math.add(a, b);

      expect(result.shape).toEqual([2, 2]);
      const expected = [2, 4, -2, -2];

      test_util.expectArraysClose(result, expected);
    });

    it("A + B", math => {
      const a = Array1D.new([2, 5, 1]);
      const b = Array1D.new([4, 2, -1]);

      const result = math.add(a, b);

      const expected = [6, 7, 0];
      test_util.expectArraysClose(result, expected);
    });

    it("A + B propagates NaNs", math => {
      const a = Array1D.new([2, 5, NaN]);
      const b = Array1D.new([4, 2, -1]);

      const res = math.add(a, b);
      test_util.expectArraysClose(res, [6, 7, NaN]);
    });

    it("A + B throws when passed ndarrays with different shape", math => {
      const a = Array1D.new([2, 5, 1, 5]);
      const b = Array1D.new([4, 2, -1]);

      expect(() => math.add(a, b)).toThrowError();
      expect(() => math.add(b, a)).toThrowError();
    });

    it("2D+scalar broadcast", math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
      const b = Scalar.new(2);
      const res = math.add(a, b);
      expect(res.shape).toEqual([2, 3]);
      test_util.expectArraysClose(res, [3, 4, 5, 6, 7, 8]);
    });

    it("scalar+1D broadcast", math => {
      const a = Scalar.new(2);
      const b = Array1D.new([1, 2, 3, 4, 5, 6]);
      const res = math.add(a, b);
      expect(res.shape).toEqual([6]);
      test_util.expectArraysClose(res, [3, 4, 5, 6, 7, 8]);
    });

    it("2D+2D broadcast each with 1 dim", math => {
      const a = Array2D.new([1, 3], [1, 2, 5]);
      const b = Array2D.new([2, 1], [7, 3]);
      const res = math.add(a, b);
      expect(res.shape).toEqual([2, 3]);
      test_util.expectArraysClose(res, [8, 9, 12, 4, 5, 8]);
    });

    it("2D+2D broadcast inner dim of b", math => {
      const a = Array2D.new([2, 3], [1, 2, 5, 4, 5, 6]);
      const b = Array2D.new([2, 1], [7, 3]);
      const res = math.add(a, b);
      expect(res.shape).toEqual([2, 3]);
      test_util.expectArraysClose(res, [8, 9, 12, 7, 8, 9]);
    });

    it("3D+scalar", math => {
      const a = Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, 6]);
      const b = Scalar.new(-1);
      const res = math.add(a, b);
      expect(res.shape).toEqual([2, 3, 1]);
      test_util.expectArraysClose(res, [0, 1, 2, 3, 4, 5]);
    });
  };

  test_util.describeMathCPU("add", [tests]);
  test_util.describeMathGPU("add", [tests], [
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 2, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": false}
  ]);
}

// subtract
{
  const tests: MathTests = it => {
    it("c - A", math => {
      const c = Scalar.new(5);
      const a = Array1D.new([7, 2, 3]);

      const result = math.scalarMinusArray(c, a);

      test_util.expectArraysClose(result, [-2, 3, 2]);
    });

    it("c - A throws when passed non scalar", math => {
      // tslint:disable-next-line:no-any
      const c: any = Array1D.new([1, 2, 3]);
      const a = Array1D.new([1, 2, 3]);

      expect(() => math.scalarMinusArray(c, a)).toThrowError();
    });

    it("A - c", math => {
      const a = Array1D.new([1, 2, -3]);
      const c = Scalar.new(5);

      const result = math.arrayMinusScalar(a, c);

      test_util.expectArraysClose(result, [-4, -3, -8]);
    });

    it("A - c propagates NaNs", math => {
      const a = Array1D.new([1, NaN, 3]);
      const c = Scalar.new(5);

      const res = math.arrayMinusScalar(a, c);

      test_util.expectArraysClose(res, [-4, NaN, -2]);
    });

    it("A - c throws when passed non scalar", math => {
      // tslint:disable-next-line:no-any
      const c: any = Array1D.new([1, 2, 3]);
      const a = Array1D.new([1, 2, 3]);

      expect(() => math.arrayMinusScalar(a, c)).toThrowError();
    });

    it("A - B", math => {
      const a = Array1D.new([2, 5, 1]);
      const b = Array1D.new([4, 2, -1]);

      const result = math.subtract(a, b);

      const expected = [-2, 3, 2];
      test_util.expectArraysClose(result, expected);
    });

    it("A - B propagates NaNs", math => {
      const a = Array1D.new([2, 5, 1]);
      const b = Array1D.new([4, NaN, -1]);

      const res = math.subtract(a, b);

      test_util.expectArraysClose(res, [-2, NaN, 2]);
    });

    it("A - B throws when passed ndarrays with different shape", math => {
      const a = Array1D.new([2, 5, 1, 5]);
      const b = Array1D.new([4, 2, -1]);

      expect(() => math.subtract(a, b)).toThrowError();
      expect(() => math.subtract(b, a)).toThrowError();
    });

    it("A - B broadcasting same rank NDArrays different shape", math => {
      const a = Array2D.new([2, 2], [1, 2, -3, -4]);
      const b = Array2D.new([2, 1], [2, 3]);

      const result = math.subtract(a, b);

      expect(result.shape).toEqual([2, 2]);
      const expected = [-1, 0, -6, -7];

      test_util.expectArraysClose(result, expected);
    });

    it("A - B broadcast 2D + 1D", math => {
      const a = Array2D.new([2, 2], [1, 2, -3, -4]);
      const b = Array1D.new([1, 2]);

      const result = math.subtract(a, b);

      expect(result.shape).toEqual([2, 2]);
      const expected = [0, 0, -4, -6];

      test_util.expectArraysClose(result, expected);
    });

    it("2D-scalar broadcast", math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
      const b = Scalar.new(2);
      const res = math.subtract(a, b);
      expect(res.shape).toEqual([2, 3]);
      test_util.expectArraysClose(res, [-1, 0, 1, 2, 3, 4]);
    });

    it("scalar-1D broadcast", math => {
      const a = Scalar.new(2);
      const b = Array1D.new([1, 2, 3, 4, 5, 6]);
      const res = math.subtract(a, b);
      expect(res.shape).toEqual([6]);
      test_util.expectArraysClose(res, [1, 0, -1, -2, -3, -4]);
    });

    it("2D-2D broadcast each with 1 dim", math => {
      const a = Array2D.new([1, 3], [1, 2, 5]);
      const b = Array2D.new([2, 1], [7, 3]);
      const res = math.subtract(a, b);
      expect(res.shape).toEqual([2, 3]);
      test_util.expectArraysClose(res, [-6, -5, -2, -2, -1, 2]);
    });

    it("2D-2D broadcast inner dim of b", math => {
      const a = Array2D.new([2, 3], [1, 2, 5, 4, 5, 6]);
      const b = Array2D.new([2, 1], [7, 3]);
      const res = math.subtract(a, b);
      expect(res.shape).toEqual([2, 3]);
      test_util.expectArraysClose(res, [-6, -5, -2, 1, 2, 3]);
    });

    it("3D-scalar", math => {
      const a = Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, 6]);
      const b = Scalar.new(-1);
      const res = math.subtract(a, b);
      expect(res.shape).toEqual([2, 3, 1]);
      test_util.expectArraysClose(res, [2, 3, 4, 5, 6, 7]);
    });
  };

  test_util.describeMathCPU("subtract", [tests]);
  test_util.describeMathGPU("subtract", [tests], [
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 2, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": false}
  ]);
}

// math.scaledArrayAdd
{
  const tests: MathTests = it => {
    it("Scaled ndarray add", math => {
      const a = Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
      const b = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
      const c1 = Scalar.new(3);
      const c2 = Scalar.new(2);

      const result = math.scaledArrayAdd<Array2D>(c1, a, c2, b);

      expect(result.shape).toEqual([2, 3]);
      test_util.expectArraysClose(result, [8, 16, 24, 32, 40, 48]);

      // Different sizes throws an error.
      const wrongSizeMat = Array2D.new([2, 2], [1, 2, 3, 4]);
      expect(() => math.scaledArrayAdd<Array2D>(c1, wrongSizeMat, c2, b))
          .toThrowError();
    });

    it("throws when passed non-scalars", math => {
      const a = Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
      const b = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
      // tslint:disable-next-line:no-any
      const c1: any = Array1D.randNormal([10]);
      const c2 = Scalar.new(2);

      expect(() => math.scaledArrayAdd(c1 as Scalar, a, c2, b)).toThrowError();
      expect(() => math.scaledArrayAdd(c2, a, c1 as Scalar, b)).toThrowError();
    });

    it("throws when NDArrays are different shape", math => {
      const a = Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
      const b = Array2D.new([2, 4], [1, 2, 3, 4, 5, 6, 7, 8]);
      const c1 = Scalar.new(3);
      const c2 = Scalar.new(2);

      expect(() => math.scaledArrayAdd<Array2D>(c1, a, c2, b)).toThrowError();
    });
  };

  test_util.describeMathCPU("scaledArrayAdd", [tests]);
  test_util.describeMathGPU("scaledArrayAdd", [tests], [
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 2, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": false}
  ]);
}

// element-wise equal
{
  const tests: MathTests = it => {
    it("propagates NaNs", math => {
      const a = Array1D.new([2, 5, NaN]);
      const b = Array1D.new([4, 5, -1]);

      const res = math.equal(a, b);
      expect(res.dtype).toBe("bool");
      test_util.expectArraysEqual(res, [0, 1, util.NAN_BOOL]);
    });

    it("strict version throws when x and y are different shape", math => {
      const a = Array1D.new([2]);
      const b = Array1D.new([4, 2, -1]);

      expect(() => math.equalStrict(a, b)).toThrowError();
      expect(() => math.equalStrict(b, a)).toThrowError();
    });

    it("2D and scalar broadcast", math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 2, 5, 6]);
      const b = Scalar.new(2);
      const res = math.equal(a, b);
      expect(res.dtype).toBe("bool");
      expect(res.shape).toEqual([2, 3]);
      test_util.expectArraysEqual(res, [0, 1, 0, 1, 0, 0]);
    });

    it("scalar and 1D broadcast", math => {
      const a = Scalar.new(2);
      const b = Array1D.new([1, 2, 3, 4, 5, 2]);
      const res = math.equal(a, b);
      expect(res.dtype).toBe("bool");
      expect(res.shape).toEqual([6]);
      test_util.expectArraysEqual(res, [0, 1, 0, 0, 0, 1]);
    });

    it("2D and 2D broadcast each with 1 dim", math => {
      const a = Array2D.new([1, 3], [1, 2, 5]);
      const b = Array2D.new([2, 1], [5, 1]);
      const res = math.equal(a, b);
      expect(res.dtype).toBe("bool");
      expect(res.shape).toEqual([2, 3]);
      test_util.expectArraysEqual(res, [0, 0, 1, 1, 0, 0]);
    });

    it("3D and scalar", math => {
      const a = Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, -1]);
      const b = Scalar.new(-1);
      const res = math.equal(a, b);
      expect(res.dtype).toBe("bool");
      expect(res.shape).toEqual([2, 3, 1]);
      test_util.expectArraysEqual(res, [0, 0, 0, 0, 0, 1]);
    });
  };

  test_util.describeMathCPU("equal", [tests]);
  test_util.describeMathGPU("equal", [tests], [
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 2, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": false}
  ]);
}

// element-wise not equal
{
  const tests: MathTests = it => {
    it("propagates NaNs", math => {
      const a = Array1D.new([2, 5, NaN]);
      const b = Array1D.new([4, 5, -1]);

      const res = math.notEqual(a, b);
      expect(res.dtype).toBe("bool");
      test_util.expectArraysEqual(res, [1, 0, util.NAN_BOOL]);
    });

    it("strict version throws when x and y are different shape", math => {
      const a = Array1D.new([2]);
      const b = Array1D.new([4, 2, -1]);

      expect(() => math.notEqualStrict(a, b)).toThrowError();
      expect(() => math.notEqualStrict(b, a)).toThrowError();
    });

    it("2D and scalar broadcast", math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 2, 5, 6]);
      const b = Scalar.new(2);
      const res = math.notEqual(a, b);
      expect(res.dtype).toBe("bool");
      expect(res.shape).toEqual([2, 3]);
      test_util.expectArraysEqual(res, [1, 0, 1, 0, 1, 1]);
    });

    it("scalar and 1D broadcast", math => {
      const a = Scalar.new(2);
      const b = Array1D.new([1, 2, 3, 4, 5, 2]);
      const res = math.notEqual(a, b);
      expect(res.dtype).toBe("bool");
      expect(res.shape).toEqual([6]);
      test_util.expectArraysEqual(res, [1, 0, 1, 1, 1, 0]);
    });

    it("2D and 2D broadcast each with 1 dim", math => {
      const a = Array2D.new([1, 3], [1, 2, 5]);
      const b = Array2D.new([2, 1], [5, 1]);
      const res = math.notEqual(a, b);
      expect(res.dtype).toBe("bool");
      expect(res.shape).toEqual([2, 3]);
      test_util.expectArraysEqual(res, [1, 1, 0, 0, 1, 1]);
    });

    it("3D and scalar", math => {
      const a = Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, -1]);
      const b = Scalar.new(-1);
      const res = math.notEqual(a, b);
      expect(res.dtype).toBe("bool");
      expect(res.shape).toEqual([2, 3, 1]);
      test_util.expectArraysEqual(res, [1, 1, 1, 1, 1, 0]);
    });
  };

  test_util.describeMathCPU("notEqual", [tests]);
  test_util.describeMathGPU("notEqual", [tests], [
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 2, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": false}
  ]);
}

// element-wise greater
{
  const tests: MathTests = it => {
    it("propagates NaNs", math => {
      const a = Array1D.new([1, 5, 1, 5, NaN]);
      const b = Array1D.new([4, 1, 2, 5, -1]);

      const res = math.greater(a, b);
      expect(res.dtype).toBe("bool");
      expect(res.getValues()).toEqual(
        new Uint8Array([0, 1, 0, 0, util.NAN_BOOL]));

      a.dispose();
      b.dispose();
    });
  };

  test_util.describeMathCPU("greater", [tests]);
  test_util.describeMathGPU("greater", [tests], [
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 2, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": false}
  ]);
}

// element-wise greater equal
{
  const tests: MathTests = it => {
    it("propagates NaNs", math => {
      const a = Array1D.new([1, 5, 1, 5, NaN]);
      const b = Array1D.new([4, 1, 2, 5, -1]);

      const res = math.greaterEqual(a, b);
      expect(res.dtype).toBe("bool");
      expect(res.getValues()).toEqual(
        new Uint8Array([0, 1, 0, 1, util.NAN_BOOL]));

      a.dispose();
      b.dispose();
    });
  };

  test_util.describeMathCPU("greaterEqual", [tests]);
  test_util.describeMathGPU("greaterEqual", [tests], [
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 2, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": false}
  ]);
}

// element-wise less
{
  const tests: MathTests = it => {
    it("propagates NaNs", math => {
      const a = Array1D.new([1, 5, 1, 5, NaN]);
      const b = Array1D.new([4, 1, 2, 5, -1]);

      const res = math.less(a, b);
      expect(res.dtype).toBe("bool");
      expect(res.getValues()).toEqual(
        new Uint8Array([1, 0, 1, 0, util.NAN_BOOL]));

      a.dispose();
      b.dispose();
    });
  };

  test_util.describeMathCPU("less", [tests]);
  test_util.describeMathGPU("less", [tests], [
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 2, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": false}
  ]);
}

// element-wise less equal
{
  const tests: MathTests = it => {
    it("propagates NaNs", math => {
      const a = Array1D.new([1, 5, 1, 5, NaN]);
      const b = Array1D.new([4, 1, 2, 5, -1]);

      const res = math.lessEqual(a, b);
      expect(res.dtype).toBe("bool");
      expect(res.getValues()).toEqual(
        new Uint8Array([1, 0, 1, 1, util.NAN_BOOL]));

      a.dispose();
      b.dispose();
    });
  };

  test_util.describeMathCPU("lessEqual", [tests]);
  test_util.describeMathGPU("lessEqual", [tests], [
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 2, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
    {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": false}
  ]);
}

// select
{
  const tests: MathTests = it => {
    it("propagates NaNs", math => {
      const a = Array1D.new([1, 5, 1, 5, NaN]);
      const b = Array1D.new([4, 1, 2, 5, -1]);
      const cond = Array1D.new([
        false, true, true, false, false], "bool");

      const res = math.select(cond, a, b);
      expect(res.dtype).toBe("float32");
      expect(res.getValues()).toEqual(
        new Float32Array([4, 5, 1, 5, -1]));

      a.dispose();
      b.dispose();
    });
  };

  test_util.describeMathCPU("select", [tests]);
  /*
  test_util.describeMathGPU('select', [tests], [
    {'WEBGL_VERSION': 1, 'WEBGL_FLOAT_TEXTURE_ENABLED': true},
    {'WEBGL_VERSION': 2, 'WEBGL_FLOAT_TEXTURE_ENABLED': true},
    {'WEBGL_VERSION': 1, 'WEBGL_FLOAT_TEXTURE_ENABLED': false}
  ]);
  */
}
