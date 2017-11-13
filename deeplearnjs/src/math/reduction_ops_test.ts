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

import * as test_util from '../test_util';
import {MathTests} from '../test_util';

import {Array1D, Array2D, Scalar} from './ndarray';

// math.min
{
  const tests: MathTests = it => {
    it('Array1D', math => {
      const a = Array1D.new([3, -1, 0, 100, -7, 2]);
      test_util.expectNumbersClose(math.min(a).get(), -7);
      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([3, NaN, 2]);

      expect(math.min(a).get()).toEqual(NaN);

      a.dispose();
    });

    it('2D', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      test_util.expectNumbersClose(math.min(a).get(), -7);

    });

    it('2D axis=[0,1]', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      test_util.expectNumbersClose(math.min(a, [0, 1]).get(), -7);
    });

    it('2D, axis=0', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      const r = math.min(a, 0);

      expect(r.shape).toEqual([3]);
      test_util.expectArraysClose(r.getValues(), new Float32Array([3, -7, 0]));
    });

    it('2D, axis=1 provided as a number', math => {
      const a = Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
      const r = math.min(a, 1);
      test_util.expectArraysClose(r.getValues(), new Float32Array([2, -7]));
    });

    it('2D, axis=[1]', math => {
      const a = Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
      const r = math.min(a, [1]);
      test_util.expectArraysClose(r.getValues(), new Float32Array([2, -7]));
    });
  };

  test_util.describeMathCPU('min', [tests]);
  test_util.describeMathGPU('min', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.max
{
  const tests: MathTests = it => {
    it('with one element dominating', math => {
      const a = Array1D.new([3, -1, 0, 100, -7, 2]);
      const r = math.max(a);
      test_util.expectNumbersClose(r.get(), 100);

      a.dispose();
    });

    it('with all elements being the same', math => {
      const a = Array1D.new([3, 3, 3]);
      const r = math.max(a);
      test_util.expectNumbersClose(r.get(), 3);

      a.dispose();
    });

    it('propagates NaNs', math => {
      expect(math.max(Array1D.new([3, NaN, 2])).get()).toEqual(NaN);
    });

    it('2D', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      test_util.expectNumbersClose(math.max(a).get(), 100);
    });

    it('2D axis=[0,1]', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      test_util.expectNumbersClose(math.max(a, [0, 1]).get(), 100);
    });

    it('2D, axis=0', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      const r = math.max(a, [0]);
      expect(r.shape).toEqual([3]);
      test_util.expectArraysClose(
          r.getValues(), new Float32Array([100, -1, 2]));
    });

    it('2D, axis=1 provided as a number', math => {
      const a = Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
      const r = math.max(a, 1);
      test_util.expectArraysClose(r.getValues(), new Float32Array([5, 100]));
    });

    it('2D, axis=[1]', math => {
      const a = Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
      const r = math.max(a, [1]);
      test_util.expectArraysClose(r.getValues(), new Float32Array([5, 100]));
    });
  };

  test_util.describeMathCPU('max', [tests]);
  test_util.describeMathGPU('max', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.argmax
{
  const tests: MathTests = it => {
    it('Array1D', math => {
      const a = Array1D.new([1, 0, 3, 2]);
      const result = math.argMax(a);
      expect(result.dtype).toBe('int32');
      expect(result.get()).toBe(2);

      a.dispose();
    });

    it('one value', math => {
      const a = Array1D.new([10]);
      const result = math.argMax(a);
      expect(result.dtype).toBe('int32');
      expect(result.get()).toBe(0);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([5, 0, 3, NaN, 3]);
      const res = math.argMax(a);
      expect(res.dtype).toBe('int32');
      test_util.assertIsNan(res.get(), res.dtype);
      a.dispose();
    });

    it('2D, no axis specified', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      expect(math.argMax(a).get()).toBe(3);
    });

    it('2D, axis=0', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      const r = math.argMax(a, 0);

      expect(r.shape).toEqual([3]);
      expect(r.dtype).toBe('int32');
      expect(r.getValues()).toEqual(new Int32Array([1, 0, 1]));
    });

    it('2D, axis=1', math => {
      const a = Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
      const r = math.argMax(a, 1);
      expect(r.dtype).toBe('int32');
      expect(r.getValues()).toEqual(new Int32Array([2, 0]));
    });
  };

  test_util.describeMathCPU('argmax', [tests]);
  test_util.describeMathGPU('argmax', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.argmin
{
  const tests: MathTests = it => {
    it('Array1D', math => {
      const a = Array1D.new([1, 0, 3, 2]);
      const result = math.argMin(a);
      expect(result.get()).toBe(1);

      a.dispose();
    });

    it('one value', math => {
      const a = Array1D.new([10]);
      const result = math.argMin(a);
      expect(result.get()).toBe(0);

      a.dispose();
    });

    it('Arg min propagates NaNs', math => {
      const a = Array1D.new([5, 0, NaN, 7, 3]);
      const res = math.argMin(a);
      test_util.assertIsNan(res.get(), res.dtype);
      a.dispose();
    });

    it('2D, no axis specified', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      expect(math.argMin(a).get()).toBe(4);
    });

    it('2D, axis=0', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      const r = math.argMin(a, 0);

      expect(r.shape).toEqual([3]);
      expect(r.dtype).toBe('int32');
      expect(r.getValues()).toEqual(new Int32Array([0, 1, 0]));
    });

    it('2D, axis=1', math => {
      const a = Array2D.new([2, 3], [3, 2, 5, 100, -7, -8]);
      const r = math.argMin(a, 1);
      expect(r.getValues()).toEqual(new Int32Array([1, 2]));
    });
  };

  test_util.describeMathCPU('argmin', [tests]);
  test_util.describeMathGPU('argmin', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.argMaxEquals
{
  const tests: MathTests = it => {
    it('equals', math => {
      const a = Array1D.new([5, 0, 3, 7, 3]);
      const b = Array1D.new([-100.3, -20.0, -10.0, -5, -100]);
      const result = math.argMaxEquals(a, b);
      expect(result.get()).toBe(1);
    });

    it('not equals', math => {
      const a = Array1D.new([5, 0, 3, 1, 3]);
      const b = Array1D.new([-100.3, -20.0, -10.0, -5, 0]);
      const result = math.argMaxEquals(a, b);
      expect(result.get()).toBe(0);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([0, 3, 1, 3]);
      const b = Array1D.new([NaN, -20.0, -10.0, -5]);
      const result = math.argMaxEquals(a, b);
      test_util.assertIsNan(result.get(), result.dtype);
    });

    it('throws when given arrays of different shape', math => {
      const a = Array1D.new([5, 0, 3, 7, 3, 10]);
      const b = Array1D.new([-100.3, -20.0, -10.0, -5, -100]);
      expect(() => math.argMaxEquals(a, b)).toThrowError();
    });
  };

  test_util.describeMathCPU('argMaxEquals', [tests]);
  test_util.describeMathGPU('argMaxEquals', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.logSumExp
{
  const tests: MathTests = it => {
    it('0', math => {
      const a = Scalar.new(0);
      const result = math.logSumExp(a);

      test_util.expectNumbersClose(result.get(), 0);

      a.dispose();
      result.dispose();
    });

    it('basic', math => {
      const a = Array1D.new([1, 2, -3]);
      const result = math.logSumExp(a);

      test_util.expectNumbersClose(
          result.get(), Math.log(Math.exp(1) + Math.exp(2) + Math.exp(-3)));

      a.dispose();
      result.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([1, 2, NaN]);
      const result = math.logSumExp(a);
      expect(result.get()).toEqual(NaN);
      a.dispose();
    });

    it('axes=0 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const r = math.logSumExp(a, [0]);
      expect(r.shape).toEqual([2]);
      const expected = new Float32Array([
        Math.log(Math.exp(1) + Math.exp(3) + Math.exp(0)),
        Math.log(Math.exp(2) + Math.exp(0) + Math.exp(1))
      ]);
      test_util.expectArraysClose(r.getValues(), expected);
    });

    it('axes=1 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.logSumExp(a, [1]);
      expect(res.shape).toEqual([3]);
      const expected = new Float32Array([
        Math.log(Math.exp(1) + Math.exp(2)),
        Math.log(Math.exp(3) + Math.exp(0)),
        Math.log(Math.exp(0) + Math.exp(1)),
      ]);
      test_util.expectArraysClose(res.getValues(), expected);
    });

    it('2D, axes=1 provided as a single digit', math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
      const res = math.logSumExp(a, 1);
      expect(res.shape).toEqual([2]);
      const expected = new Float32Array([
        Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3)),
        Math.log(Math.exp(0) + Math.exp(0) + Math.exp(1))
      ]);
      test_util.expectArraysClose(res.getValues(), expected);
    });

    it('axes=0,1 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.logSumExp(a, [0, 1]);
      expect(res.shape).toEqual([]);
      const expected = new Float32Array([Math.log(
          Math.exp(1) + Math.exp(2) + Math.exp(3) + Math.exp(0) + Math.exp(0) +
          Math.exp(1))]);
      test_util.expectArraysClose(res.getValues(), expected);
    });
  };

  test_util.describeMathCPU('logSumExp', [tests]);
  test_util.describeMathGPU('logSumExp', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.sum
{
  const tests: MathTests = it => {
    it('basic', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const result = math.sum(a);
      test_util.expectNumbersClose(result.get(), 7);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, NaN, 0, 1]);
      expect(math.sum(a).get()).toEqual(NaN);
      a.dispose();
    });

    it('sum over dtype int32', math => {
      const a = Array1D.new([1, 5, 7, 3], 'int32');
      const sum = math.sum(a);
      expect(sum.get()).toBe(16);
    });

    it('sum over dtype bool', math => {
      const a = Array1D.new([true, false, false, true, true], 'bool');
      const sum = math.sum(a);
      expect(sum.get()).toBe(3);
    });

    it('sums all values in 2D array with keep dim', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.sum(a, null, true /* keepDims */);
      expect(res.shape).toEqual([1, 1]);
      test_util.expectArraysClose(res.getValues(), new Float32Array([7]));
    });

    it('sums across axis=0 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.sum(a, [0]);
      expect(res.shape).toEqual([2]);
      test_util.expectArraysClose(res.getValues(), new Float32Array([4, 3]));
    });

    it('sums across axis=1 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.sum(a, [1]);
      expect(res.shape).toEqual([3]);
      test_util.expectArraysClose(res.getValues(), new Float32Array([3, 3, 1]));
    });

    it('2D, axis=1 provided as number', math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
      const res = math.sum(a, 1);
      expect(res.shape).toEqual([2]);
      test_util.expectArraysClose(res.getValues(), new Float32Array([6, 1]));
    });

    it('sums across axis=0,1 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.sum(a, [0, 1]);
      expect(res.shape).toEqual([]);
      test_util.expectArraysClose(res.getValues(), new Float32Array([7]));
    });
  };

  test_util.describeMathCPU('sum', [tests]);
  test_util.describeMathGPU('sum', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.mean
{
  const tests: MathTests = it => {
    it('basic', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const r = math.mean(a);
      expect(r.dtype).toBe('float32');
      test_util.expectNumbersClose(r.get(), 7 / 6);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, NaN, 0, 1]);
      const r = math.mean(a);
      expect(r.dtype).toBe('float32');
      expect(r.get()).toEqual(NaN);
      a.dispose();
    });

    it('mean(int32) => float32', math => {
      const a = Array1D.new([1, 5, 7, 3], 'int32');
      const r = math.mean(a);
      expect(r.dtype).toBe('float32');
      test_util.expectNumbersClose(r.get(), 4);
    });

    it('mean(bool) => float32', math => {
      const a = Array1D.new([true, false, false, true, true], 'bool');
      const r = math.mean(a);
      expect(r.dtype).toBe('float32');
      test_util.expectNumbersClose(r.get(), 3 / 5);
    });

    it('2D array with keep dim', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.mean(a, null, true /* keepDims */);
      expect(res.shape).toEqual([1, 1]);
      expect(res.dtype).toBe('float32');
      test_util.expectArraysClose(res.getValues(), new Float32Array([7 / 6]));
    });

    it('axis=0 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.mean(a, [0]);

      expect(res.shape).toEqual([2]);
      expect(res.dtype).toBe('float32');
      test_util.expectArraysClose(
          res.getValues(), new Float32Array([4 / 3, 1]));
    });

    it('axis=1 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.mean(a, [1]);
      expect(res.dtype).toBe('float32');
      expect(res.shape).toEqual([3]);
      test_util.expectArraysClose(
          res.getValues(), new Float32Array([1.5, 1.5, 0.5]));
    });

    it('2D, axis=1 provided as number', math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
      const res = math.mean(a, 1);
      expect(res.shape).toEqual([2]);
      expect(res.dtype).toBe('float32');
      test_util.expectArraysClose(
          res.getValues(), new Float32Array([2, 1 / 3]));
    });

    it('axis=0,1 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.mean(a, [0, 1]);
      expect(res.shape).toEqual([]);
      expect(res.dtype).toBe('float32');
      test_util.expectArraysClose(res.getValues(), new Float32Array([7 / 6]));
    });
  };

  test_util.describeMathCPU('mean', [tests]);
  test_util.describeMathGPU('mean', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.moments
{
  const tests: MathTests = it => {
    it('basic', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const {mean, variance} = math.moments(a);

      expect(mean.dtype).toBe('float32');
      expect(variance.dtype).toBe('float32');
      test_util.expectNumbersClose(mean.get(), 7 / 6);
      test_util.expectNumbersClose(variance.get(), 1.1389);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, NaN, 0, 1]);
      const {mean, variance} = math.moments(a);

      expect(mean.dtype).toBe('float32');
      expect(variance.dtype).toBe('float32');
      expect(mean.get()).toEqual(NaN);
      expect(variance.get()).toEqual(NaN);
      a.dispose();
    });

    it('moments(int32) => float32', math => {
      const a = Array1D.new([1, 5, 7, 3], 'int32');
      const {mean, variance} = math.moments(a);

      expect(mean.dtype).toBe('float32');
      expect(variance.dtype).toBe('float32');
      test_util.expectNumbersClose(mean.get(), 4);
      test_util.expectNumbersClose(variance.get(), 5);
    });

    it('moments(bool) => float32', math => {
      const a = Array1D.new([true, false, false, true, true], 'bool');
      const {mean, variance} = math.moments(a);

      expect(mean.dtype).toBe('float32');
      expect(variance.dtype).toBe('float32');
      test_util.expectNumbersClose(mean.get(), 3 / 5);
      test_util.expectNumbersClose(variance.get(), 0.23999998);
    });

    it('2D array with keep dim', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const {mean, variance} = math.moments(a, null, true /* keepDims */);

      expect(mean.shape).toEqual([1, 1]);
      expect(mean.dtype).toBe('float32');
      expect(variance.shape).toEqual([1, 1]);
      expect(variance.dtype).toBe('float32');
      test_util.expectArraysClose(mean.getValues(), new Float32Array([7 / 6]));
      test_util.expectArraysClose(
          variance.getValues(), new Float32Array([1.138889]));
    });

    it('axis=0 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const {mean, variance} = math.moments(a, [0]);

      expect(mean.shape).toEqual([2]);
      expect(mean.dtype).toBe('float32');
      expect(variance.shape).toEqual([2]);
      expect(variance.dtype).toBe('float32');
      test_util.expectArraysClose(
          mean.getValues(), new Float32Array([4 / 3, 1]));
      test_util.expectArraysClose(
          variance.getValues(), new Float32Array([1.556, 2 / 3]));
    });

    it('axis=1 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const {mean, variance} = math.moments(a, [1]);

      expect(mean.dtype).toBe('float32');
      expect(mean.shape).toEqual([3]);
      expect(variance.dtype).toBe('float32');
      expect(variance.shape).toEqual([3]);
      test_util.expectArraysClose(
          mean.getValues(), new Float32Array([1.5, 1.5, 0.5]));
      test_util.expectArraysClose(
          variance.getValues(), new Float32Array([0.25, 2.25, 0.25]));
    });

    it('2D, axis=1 provided as number', math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
      const {mean, variance} = math.moments(a, 1);

      expect(mean.shape).toEqual([2]);
      expect(mean.dtype).toBe('float32');
      expect(variance.shape).toEqual([2]);
      expect(variance.dtype).toBe('float32');
      test_util.expectArraysClose(
          mean.getValues(), new Float32Array([2, 1 / 3]));
      test_util.expectArraysClose(
          variance.getValues(), new Float32Array([2 / 3, 0.222]));
    });

    it('axis=0,1 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const {mean, variance} = math.moments(a, [0, 1]);

      expect(mean.shape).toEqual([]);
      expect(mean.dtype).toBe('float32');
      expect(variance.shape).toEqual([]);
      expect(variance.dtype).toBe('float32');
      test_util.expectArraysClose(mean.getValues(), new Float32Array([7 / 6]));
      test_util.expectArraysClose(
          variance.getValues(), new Float32Array([1.1389]));
    });
  };

  test_util.describeMathCPU('moments', [tests]);
  test_util.describeMathGPU('moments', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
