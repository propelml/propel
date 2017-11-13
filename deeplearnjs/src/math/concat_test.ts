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

import {Array1D, Array2D, Array3D} from './ndarray';

// math.concat1D
{
  const tests: MathTests = it => {
    it('3 + 5', math => {
      const a = Array1D.new([3]);
      const b = Array1D.new([5]);

      const result = math.concat1D(a, b);
      const expected = new Float32Array([3, 5]);
      test_util.expectArraysClose(result.getValues(), expected);

      a.dispose();
      b.dispose();
    });

    it('3 + [5,7]', math => {
      const a = Array1D.new([3]);
      const b = Array1D.new([5, 7]);

      const result = math.concat1D(a, b);
      const expected = new Float32Array([3, 5, 7]);
      test_util.expectArraysClose(result.getValues(), expected);

      a.dispose();
      b.dispose();
    });

    it('[3,5] + 7', math => {
      const a = Array1D.new([3, 5]);
      const b = Array1D.new([7]);

      const result = math.concat1D(a, b);
      const expected = new Float32Array([3, 5, 7]);
      test_util.expectArraysClose(result.getValues(), expected);

      a.dispose();
      b.dispose();
    });
  };

  test_util.describeMathCPU('concat1D', [tests]);
  test_util.describeMathGPU('concat1D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.concat2D
{
  const tests: MathTests = it => {
    it('[[3]] + [[5]], axis=0', math => {
      const axis = 0;
      const a = Array2D.new([1, 1], [3]);
      const b = Array2D.new([1, 1], [5]);

      const result = math.concat2D(a, b, axis);
      const expected = new Float32Array([3, 5]);

      expect(result.shape).toEqual([2, 1]);
      test_util.expectArraysClose(result.getValues(), expected);

      a.dispose();
      b.dispose();
    });

    it('[[3]] + [[5]], axis=1', math => {
      const axis = 1;
      const a = Array2D.new([1, 1], [3]);
      const b = Array2D.new([1, 1], [5]);

      const result = math.concat2D(a, b, axis);
      const expected = new Float32Array([3, 5]);

      expect(result.shape).toEqual([1, 2]);
      test_util.expectArraysClose(result.getValues(), expected);

      a.dispose();
      b.dispose();
    });

    it('[[1, 2], [3, 4]] + [[5, 6]], axis=0', math => {
      const axis = 0;
      const a = Array2D.new([2, 2], [[1, 2], [3, 4]]);
      const b = Array2D.new([1, 2], [[5, 6]]);

      const result = math.concat2D(a, b, axis);
      const expected = new Float32Array([1, 2, 3, 4, 5, 6]);

      expect(result.shape).toEqual([3, 2]);
      test_util.expectArraysClose(result.getValues(), expected);

      a.dispose();
      b.dispose();
    });

    it('[[1, 2], [3, 4]] + [[5, 6]], axis=1 throws error', math => {
      const axis = 1;
      const a = Array2D.new([2, 2], [[1, 2], [3, 4]]);
      const b = Array2D.new([1, 2], [[5, 6]]);

      expect(() => math.concat2D(a, b, axis)).toThrowError();

      a.dispose();
      b.dispose();
    });

    it('[[1, 2], [3, 4]] + [[5, 6], [7, 8]], axis=1', math => {
      const axis = 1;
      const a = Array2D.new([2, 2], [[1, 2], [3, 4]]);
      const b = Array2D.new([2, 2], [[5, 6], [7, 8]]);

      const result = math.concat2D(a, b, axis);
      const expected = new Float32Array([1, 2, 5, 6, 3, 4, 7, 8]);

      expect(result.shape).toEqual([2, 4]);
      test_util.expectArraysClose(result.getValues(), expected);

      a.dispose();
      b.dispose();
    });
  };

  test_util.describeMathCPU('concat2D', [tests]);
  test_util.describeMathGPU('concat2D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.concat3D
{
  const tests: MathTests = it => {
    it('shapes correct concat axis=0', math => {
      const ndarray1 = Array3D.new([1, 1, 3], [1, 2, 3]);
      const ndarray2 = Array3D.new([1, 1, 3], [4, 5, 6]);
      const values = math.concat3D(ndarray1, ndarray2, 0);
      expect(values.shape).toEqual([2, 1, 3]);
      test_util.expectArraysClose(
          values.getValues(), new Float32Array([1, 2, 3, 4, 5, 6]));
    });

    it('concat axis=0', math => {
      const ndarray1 = Array3D.new([1, 2, 3], [1, 11, 111, 2, 22, 222]);
      const ndarray2 = Array3D.new(
          [2, 2, 3], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);
      const values = math.concat3D(ndarray1, ndarray2, 0);
      expect(values.shape).toEqual([3, 2, 3]);
      test_util.expectArraysClose(values.getValues(), new Float32Array([
                                    1, 11, 111, 2, 22, 222, 5, 55, 555, 6, 66,
                                    666, 7, 77, 777, 8, 88, 888
                                  ]));
    });

    it('shapes correct concat axis=1', math => {
      const ndarray1 = Array3D.new([1, 1, 3], [1, 2, 3]);
      const ndarray2 = Array3D.new([1, 1, 3], [4, 5, 6]);
      const values = math.concat3D(ndarray1, ndarray2, 1);
      expect(values.shape).toEqual([1, 2, 3]);
      test_util.expectArraysClose(
          values.getValues(), new Float32Array([1, 2, 3, 4, 5, 6]));
    });

    it('concat axis=1', math => {
      const ndarray1 = Array3D.new([2, 1, 3], [1, 11, 111, 3, 33, 333]);
      const ndarray2 = Array3D.new(
          [2, 2, 3], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);
      const values = math.concat3D(ndarray1, ndarray2, 1);
      expect(values.shape).toEqual([2, 3, 3]);
      test_util.expectArraysClose(values.getValues(), new Float32Array([
                                    1, 11, 111, 5, 55, 555, 6, 66, 666, 3, 33,
                                    333, 7, 77, 777, 8, 88, 888
                                  ]));
    });

    it('shapes correct concat axis=2', math => {
      const ndarray1 = Array3D.new([1, 1, 3], [1, 2, 3]);
      const ndarray2 = Array3D.new([1, 1, 3], [4, 5, 6]);
      const values = math.concat3D(ndarray1, ndarray2, 2);
      expect(values.shape).toEqual([1, 1, 6]);
      test_util.expectArraysClose(
          values.getValues(), new Float32Array([1, 2, 3, 4, 5, 6]));
    });

    it('concat axis=2', math => {
      const ndarray1 = Array3D.new([2, 2, 2], [1, 11, 2, 22, 3, 33, 4, 44]);
      const ndarray2 = Array3D.new(
          [2, 2, 3], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);
      const values = math.concat3D(ndarray1, ndarray2, 2);
      expect(values.shape).toEqual([2, 2, 5]);
      test_util.expectArraysClose(values.getValues(), new Float32Array([
                                    1, 11, 5, 55, 555, 2, 22, 6, 66, 666,
                                    3, 33, 7, 77, 777, 4, 44, 8, 88, 888
                                  ]));
    });

    it('concat throws when invalid non-axis shapes, axis=0', math => {
      const axis = 0;
      const x1 = Array3D.new([1, 1, 3], [1, 11, 111]);
      const x2 = Array3D.new(
          [2, 2, 3], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);
      expect(() => math.concat3D(x1, x2, axis)).toThrowError();
    });

    it('concat throws when invalid non-axis shapes, axis=1', math => {
      const axis = 1;
      const x1 = Array3D.new([1, 1, 3], [1, 11, 111]);
      const x2 = Array3D.new(
          [2, 2, 3], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);
      expect(() => math.concat3D(x1, x2, axis)).toThrowError();
    });

    it('concat throws when invalid non-axis shapes, axis=2', math => {
      const axis = 2;
      const x1 = Array3D.new([1, 2, 2], [1, 11, 2, 22]);
      const x2 = Array3D.new(
          [2, 2, 3], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);
      expect(() => math.concat3D(x1, x2, axis)).toThrowError();
    });
  };

  test_util.describeMathCPU('concat3D', [tests]);
  test_util.describeMathGPU('concat3D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
