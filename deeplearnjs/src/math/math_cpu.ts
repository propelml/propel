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

import * as seedrandom from 'seedrandom';
import * as util from '../util';
import * as axis_util from './axis_util';
import * as broadcast_util from './broadcast_util';
import * as concat_util from './concat_util';
import * as conv_util from './conv_util';
import {ConvInfo} from './conv_util';
import * as copy2D_util from './copy2d_util';
import {MatrixOrientation, NDArrayMath, SumTypes, SumTypesMap} from './math';
// tslint:disable-next-line:max-line-length
import {Array1D, Array2D, Array3D, Array4D, DataTypes, NDArray, Scalar} from './ndarray';

export class NDArrayMathCPU extends NDArrayMath {
  constructor(safeMode = false) {
    super(safeMode);
  }

  protected cloneInternal<T extends NDArray>(ndarray: T): T {
    return NDArray.make(
               ndarray.shape,
               {values: new Float32Array(ndarray.getValues())}) as T;
  }

  protected slice1DInternal(input: Array1D, begin: number, size: number):
      Array1D {
    const newVals = input.getValues().slice(begin, begin + size);
    return Array1D.new(newVals);
  }

  protected slice2DInternal(input: Array2D, begin: [number, number], size: [
    number, number
  ]): Array2D {
    const result = Array2D.zeros(size);
    const [startI, startJ] = begin;

    for (let i = 0; i < size[0]; ++i) {
      for (let j = 0; j < size[1]; ++j) {
        const val = input.get(i + startI, j + startJ);
        result.set(val, i, j);
      }
    }
    return result;
  }

  protected slice3DInternal(
      input: Array3D, begin: [number, number, number],
      size: [number, number, number]): Array3D {
    const result = Array3D.zeros(size);
    const [startI, startJ, startK] = begin;

    for (let i = 0; i < size[0]; ++i) {
      for (let j = 0; j < size[1]; ++j) {
        for (let k = 0; k < size[2]; ++k) {
          const val = input.get(i + startI, j + startJ, k + startK);
          result.set(val, i, j, k);
        }
      }
    }
    return result;
  }
  protected slice4DInternal(
      input: Array4D, begin: [number, number, number, number],
      size: [number, number, number, number]): Array4D {
    const result = Array4D.zeros(size);
    const [startI, startJ, startK, startL] = begin;

    for (let i = 0; i < size[0]; ++i) {
      for (let j = 0; j < size[1]; ++j) {
        for (let k = 0; k < size[2]; ++k) {
          for (let l = 0; l < size[3]; ++l) {
            const val =
                input.get(i + startI, j + startJ, k + startK, l + startL);
            result.set(val, i, j, k, l);
          }
        }
      }
    }
    return result;
  }

  protected copy2DInternal(
      source: Array2D, sourceBeginRowCol: [number, number],
      sourceSizeRowCol: [number, number], dest: Array2D,
      destBeginRowCol: [number, number],
      destSizeRowCol: [number, number]): void {
    copy2D_util.validateShapes(sourceSizeRowCol, destSizeRowCol);
    const srcValues = source.getValues();
    const dstValues = dest.getValues();
    const n = sourceSizeRowCol[0] * sourceSizeRowCol[1];
    for (let i = 0; i < n; ++i) {
      const srcRow = sourceBeginRowCol[0] + Math.floor(i / sourceSizeRowCol[1]);
      const srcCol = sourceBeginRowCol[1] + (i % sourceSizeRowCol[1]);
      const srcOff = srcRow * source.shape[1] + srcCol;
      const dstRow = destBeginRowCol[0] + Math.floor(i / destSizeRowCol[1]);
      const dstCol = destBeginRowCol[1] + (i % destSizeRowCol[1]);
      const dstOff = dstRow * dest.shape[1] + dstCol;
      dstValues[dstOff] = srcValues[srcOff];
    }
  }

  protected concat1DInternal(a: Array1D, b: Array1D): Array1D {
    const outShape = concat_util.computeOutShape(a.shape, b.shape, 0);
    const result = Array1D.zeros(outShape as [number]);

    // Use built-in TypedArray.set() method for speed.
    const aVals = a.getValues();
    const bVals = b.getValues();
    const vals = result.getValues();
    vals.set(aVals, 0);
    vals.set(bVals, a.size);

    return result;
  }

  protected concat2DInternal(a: Array2D, b: Array2D, axis: number): Array2D {
    const outShape = concat_util.computeOutShape(a.shape, b.shape, axis);
    const result = Array2D.zeros(outShape as [number, number]);

    if (axis === 0) {
      // Use built-in TypedArray.set() method for speed.
      const aVals = a.getValues();
      const bVals = b.getValues();
      const vals = result.getValues();
      vals.set(aVals, 0);
      vals.set(bVals, a.size);
      return result;
    }

    for (let i = 0; i < outShape[0]; ++i) {
      for (let j = 0; j < outShape[1]; ++j) {
        const index: [number, number] = [i, j];
        let value: number;
        if (index[axis] < a.shape[axis]) {
          value = a.get(i, j);
        } else {
          index[axis] -= a.shape[axis];
          const [i2, j2] = index;
          value = b.get(i2, j2);
        }

        result.set(value, i, j);
      }
    }
    return result;
  }

  protected concat3DInternal(a: Array3D, b: Array3D, axis: number): Array3D {
    const outShape = concat_util.computeOutShape(a.shape, b.shape, axis);

    const result = Array3D.zeros(outShape as [number, number, number]);

    if (axis === 0) {
      // Use built-in TypedArray.set() method for speed.
      const aVals = a.getValues();
      const bVals = b.getValues();
      const vals = result.getValues();
      vals.set(aVals, 0);
      vals.set(bVals, a.size);
      return result;
    }

    for (let i = 0; i < outShape[0]; ++i) {
      for (let j = 0; j < outShape[1]; ++j) {
        for (let k = 0; k < outShape[2]; ++k) {
          // Shader begins.
          const index: [number, number, number] = [i, j, k];
          let value: number;
          if (index[axis] < a.shape[axis]) {
            value = a.get(i, j, k);
          } else {
            index[axis] -= a.shape[axis];
            const [i2, j2, k2] = index;
            value = b.get(i2, j2, k2);
          }

          result.set(value, i, j, k);
        }
      }
    }

    return result;
  }

  protected concat4DInternal(a: Array4D, b: Array4D, axis: number): Array4D {
    const outShape = concat_util.computeOutShape(a.shape, b.shape, axis);
    const result = Array4D.zeros(outShape as [number, number, number, number]);

    if (axis === 0) {
      // Use built-in TypedArray.set() method for speed.
      const aVals = a.getValues();
      const bVals = b.getValues();
      const vals = result.getValues();
      vals.set(aVals, 0);
      vals.set(bVals, a.size);
      return result;
    }

    for (let i = 0; i < outShape[0]; ++i) {
      for (let j = 0; j < outShape[1]; ++j) {
        for (let k = 0; k < outShape[2]; ++k) {
          for (let l = 0; l < outShape[3]; ++l) {
            // Shader begins.
            const index: [number, number, number, number] = [i, j, k, l];
            let value: number;
            if (index[axis] < a.shape[axis]) {
              value = a.get(i, j, k, l);
            } else {
              index[axis] -= a.shape[axis];
              const [i2, j2, k2, l2] = index;
              value = b.get(i2, j2, k2, l2);
            }

            result.set(value, i, j, k, l);
          }
        }
      }
    }

    return result;
  }

  protected scaledArrayAddInternal<T extends NDArray>(
      c1: Scalar, a: T, c2: Scalar, b: T): T {
    const c1Val = c1.get();
    const c2Val = c2.get();
    return this.broadcastedBinaryOp(a, b, 'float32', (aVal, bVal) => {
      return c1Val * aVal + c2Val * bVal;
    }) as T;
  }

  protected negInternal<T extends NDArray>(a: T): T {
    return this.scalarTimesArray(Scalar.NEG_ONE, a);
  }

  protected addInternal<T extends NDArray>(a: T, b: T): T {
    return this.scaledArrayAddInternal<T>(Scalar.ONE, a, Scalar.ONE, b);
  }

  protected subtractInternal<T extends NDArray>(a: T, b: T): T {
    return this.scaledArrayAddInternal<T>(Scalar.ONE, a, Scalar.NEG_ONE, b);
  }

  protected matMulInternal(
      a: Array2D, b: Array2D, aOrientation = MatrixOrientation.REGULAR,
      bOrientation = MatrixOrientation.REGULAR): Array2D {
    const sharedDim =
        (aOrientation === MatrixOrientation.REGULAR) ? a.shape[1] : a.shape[0];

    const leftDim =
        (aOrientation === MatrixOrientation.REGULAR) ? a.shape[0] : a.shape[1];
    const rightDim =
        (bOrientation === MatrixOrientation.REGULAR) ? b.shape[1] : b.shape[0];

    const normalGetter = (matrix: Array2D, i: number, j: number) =>
        matrix.get(i, j);
    const transposedGetter = (matrix: Array2D, i: number, j: number) =>
        matrix.get(j, i);

    const aGetter = (aOrientation === MatrixOrientation.REGULAR) ?
        normalGetter :
        transposedGetter;
    const bGetter = (bOrientation === MatrixOrientation.REGULAR) ?
        normalGetter :
        transposedGetter;
    const values = new Float32Array(leftDim * rightDim);
    let index = 0;

    for (let i = 0; i < leftDim; ++i) {
      for (let j = 0; j < rightDim; ++j) {
        let sum = 0;
        for (let k = 0; k < sharedDim; ++k) {
          // TODO: optimize CPU matmul.
          sum += aGetter(a, i, k) * bGetter(b, k, j);
        }
        values[index++] = sum;
      }
    }
    return Array2D.new([leftDim, rightDim], values);
  }

  protected multiplyInternal<T extends NDArray>(a: T, b: T): T {
    const newShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const newValues = new Float32Array(util.sizeFromShape(newShape));

    const aValues = a.getValues();
    const bValues = b.getValues();
    for (let i = 0; i < newValues.length; ++i) {
      newValues[i] = aValues[i % a.size] * bValues[i % b.size];
    }
    return NDArray.make(newShape, {values: newValues}) as T;
  }

  protected divideInternal(a: NDArray, b: NDArray): NDArray<'float32'> {
    const newShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const newValues = new Float32Array(util.sizeFromShape(newShape));

    const aValues = a.getValues();
    const bValues = b.getValues();

    for (let i = 0; i < newValues.length; ++i) {
      newValues[i] = aValues[i % a.size] / bValues[i % b.size];
    }
    return NDArray.make(newShape, {values: newValues}, 'float32');
  }

  protected sumInternal<T extends keyof DataTypes>(
      input: NDArray<T>, axes: number[]): NDArray<SumTypes[T]> {
    axis_util.assertAxesAreInnerMostDims('sum', axes, input.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(input.shape, axes);
    const resultDtype = SumTypesMap[input.dtype] as keyof SumTypes;
    const result = NDArray.zeros(outShape, resultDtype);
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = result.getValues();

    const aVals = input.getValues();
    for (let i = 0; i < vals.length; ++i) {
      const offset = i * reduceSize;
      let sum = 0;
      for (let j = 0; j < reduceSize; ++j) {
        sum += aVals[offset + j];
      }
      vals[i] = sum;
    }
    return result as NDArray<SumTypes[T]>;
  }

  protected argMinInternal(input: NDArray, axes: number[]): NDArray<'int32'> {
    axis_util.assertAxesAreInnerMostDims('argMax', axes, input.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(input.shape, axes);
    const result = NDArray.zeros(outShape, 'int32');
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = result.getValues();

    const aVals = input.getValues();
    for (let i = 0; i < vals.length; ++i) {
      const offset = i * reduceSize;
      let min = aVals[offset];
      let minIndex = 0;
      for (let j = 0; j < reduceSize; ++j) {
        const value = aVals[offset + j];
        if (isNaN(value)) {
          minIndex = util.NAN_INT32;
          break;
        }
        if (value < min) {
          min = value;
          minIndex = j;
        }
      }
      vals[i] = minIndex;
    }
    return result;
  }

  protected argMaxInternal(input: NDArray, axes: number[]): NDArray<'int32'> {
    axis_util.assertAxesAreInnerMostDims('argMax', axes, input.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(input.shape, axes);
    const result = NDArray.zeros(outShape, 'int32');
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = result.getValues();

    const aVals = input.getValues();
    for (let i = 0; i < vals.length; ++i) {
      const offset = i * reduceSize;
      let max = aVals[offset];
      let maxIndex = 0;
      for (let j = 0; j < reduceSize; ++j) {
        const value = aVals[offset + j];
        if (isNaN(value)) {
          maxIndex = util.NAN_INT32;
          break;
        }
        if (value > max) {
          max = value;
          maxIndex = j;
        }
      }
      vals[i] = maxIndex;
    }
    return result;
  }

  protected equalInternal(a: NDArray, b: NDArray): NDArray<'bool'> {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      if (util.isValNaN(aVal, a.dtype) || util.isValNaN(bVal, b.dtype)) {
        return util.getNaN('bool');
      } else {
        return (aVal === bVal) ? 1 : 0;
      }
    });
  }

  protected topKInternal(ndarray: NDArray, k: number):
      {values: Array1D, indices: Array1D} {
    const values = ndarray.getValues();
    const valuesAndIndices: Array<{value: number, index: number}> = [];
    for (let i = 0; i < values.length; i++) {
      valuesAndIndices.push({value: values[i], index: i});
    }
    valuesAndIndices.sort((a, b) => {
      return b.value - a.value;
    });
    const topkValues = new Float32Array(k);
    const topkIndices = new Float32Array(k);
    for (let i = 0; i < k; i++) {
      topkValues[i] = valuesAndIndices[i].value;
      topkIndices[i] = valuesAndIndices[i].index;
    }
    return {values: Array1D.new(topkValues), indices: Array1D.new(topkIndices)};
  }

  protected minInternal<G extends keyof DataTypes>(
      input: NDArray<G>, axes: number[]): NDArray<G> {
    axis_util.assertAxesAreInnerMostDims('min', axes, input.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(input.shape, axes);
    const result = NDArray.zeros(outShape, input.dtype);
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = result.getValues();

    const aVals = input.getValues();
    for (let i = 0; i < vals.length; ++i) {
      const offset = i * reduceSize;
      let min = aVals[0];
      for (let j = 0; j < reduceSize; ++j) {
        const value = aVals[offset + j];
        if (isNaN(value)) {
          min = Number.NaN;
          break;
        }
        if (value < min) {
          min = value;
        }
      }
      vals[i] = min;
    }
    return result;
  }

  protected maxInternal<G extends keyof DataTypes>(
      input: NDArray<G>, axes: number[]): NDArray<G> {
    axis_util.assertAxesAreInnerMostDims('max', axes, input.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(input.shape, axes);
    const result = NDArray.zeros(outShape, input.dtype);
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = result.getValues();

    const aVals = input.getValues();
    for (let i = 0; i < vals.length; ++i) {
      const offset = i * reduceSize;
      let max = aVals[offset];
      for (let j = 0; j < reduceSize; ++j) {
        const value = aVals[offset + j];
        if (isNaN(value)) {
          max = Number.NaN;
          break;
        }
        if (value > max) {
          max = value;
        }
      }
      vals[i] = max;
    }
    return result;
  }

  protected ceilInternal<T extends NDArray>(ndarray: T): T {
    const values = ndarray.getValues();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = Math.ceil(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: newValues}) as T;
  }

  protected floorInternal<T extends NDArray>(ndarray: T): T {
    const values = ndarray.getValues();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = Math.floor(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: newValues}) as T;
  }

  protected expInternal<T extends NDArray>(ndarray: T): T {
    const values = ndarray.getValues();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = Math.exp(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: newValues}) as T;
  }

  protected logInternal<T extends NDArray>(ndarray: T): T {
    const values = ndarray.getValues();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = Math.log(value);
    }
    return NDArray.make(ndarray.shape, {values: newValues}) as T;
  }

  protected sqrtInternal<T extends NDArray>(ndarray: T): T {
    const values = ndarray.getValues();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = Math.sqrt(value);
    }
    return NDArray.make(ndarray.shape, {values: newValues}) as T;
  }

  protected squareInternal<T extends NDArray>(x: T): T {
    const values = x.getValues();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = value * value;
    }
    return NDArray.make(x.shape, {values: newValues}) as T;
  }

  protected logSumExpInternal(input: NDArray, axes: number[]): NDArray {
    axis_util.assertAxesAreInnerMostDims('logSumExp', axes, input.rank);
    const xMax = this.max(input, axes, true /* keepDims */);
    const a = this.subtract(input, xMax);
    const b = this.exp(a);
    const c = this.sum(b, axes);
    const d = this.log(c);
    const result = this.add(xMax.reshape(d.shape), d);
    return result;
  }

  protected reluInternal<T extends NDArray>(input: T): T {
    const res = NDArray.zeros(input.shape, input.dtype);
    const resVals = res.getValues();
    const inVals = input.getValues();
    for (let i = 0; i < inVals.length; ++i) {
      const val = inVals[i];
      if (util.isValNaN(val, input.dtype)) {
        resVals[i] = util.getNaN(res.dtype);
      } else {
        resVals[i] = Math.max(0, inVals[i]);
      }
    }
    return res as T;
  }

  protected eluInternal<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.dataSync();
    for (let i = 0; i < values.length; ++i) {
      const v = values[i];
      if (v >= 0) {
        resultValues[i] = v;
      } else {
        resultValues[i] = (Math.exp(v) - 1);
      }
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  protected leakyReluInternal<T extends NDArray>(ndarray: T, alpha: number) {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.dataSync();
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v >= 0) {
        resultValues[i] = v;
      } else {
        resultValues[i] = alpha * v;
      }
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  protected clipInternal<T extends NDArray>(
      ndarray: T, min: number, max: number): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.min(max, Math.max(min, values[i]));
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  protected absInternal<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.abs(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  protected sigmoidInternal<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = 1 / (1 + Math.exp(-values[i]));
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  protected sinInternal<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.sin(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  protected cosInternal<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.cos(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  protected tanInternal<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.tan(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  protected asinInternal<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.asin(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  protected acosInternal<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.acos(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  protected atanInternal<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.atan(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  protected sinhInternal<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.sinh(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  protected coshInternal<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.cosh(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  protected tanhInternal<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = util.tanh(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  protected stepInternal<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      resultValues[i] = value > 0 ? 1 : (value < 0 ? 0 : value);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  protected conv2dInternal(
      x: Array3D, filter: Array4D, bias: Array1D|null,
      convInfo: ConvInfo): Array3D {
    const [xRows, xCols, inputDepth] = x.shape;
    const filterHeight = filter.shape[0];
    const filterWidth = filter.shape[1];
    const outDepth = filter.shape[3];
    const padLeft = convInfo.padInfo.left;
    const padTop = convInfo.padInfo.top;

    const y = Array3D.zeros(convInfo.outShape);
    for (let d2 = 0; d2 < outDepth; ++d2) {
      for (let yR = 0; yR < y.shape[0]; ++yR) {
        const xRCorner = yR * convInfo.strideHeight - padLeft;
        const xRMin = Math.max(0, xRCorner);
        const xRMax = Math.min(xRows, filterHeight + xRCorner);
        for (let yC = 0; yC < y.shape[1]; ++yC) {
          const xCCorner = yC * convInfo.strideWidth - padTop;
          const xCMin = Math.max(0, xCCorner);
          const xCMax = Math.min(xCols, filterWidth + xCCorner);
          let dotProd = 0;
          for (let xR = xRMin; xR < xRMax; ++xR) {
            const wR = xR - xRCorner;
            for (let xC = xCMin; xC < xCMax; ++xC) {
              const wC = xC - xCCorner;
              for (let d1 = 0; d1 < inputDepth; ++d1) {
                const pixel = x.get(xR, xC, d1);
                const weight = filter.get(wR, wC, d1, d2);
                dotProd += pixel * weight;
              }
            }
          }
          const biasVal = (bias != null) ? bias.get(d2) : 0;
          y.set(dotProd + biasVal, yR, yC, d2);
        }
      }
    }
    return y;
  }

  protected conv2dDerInputInternal(
      dy: Array3D, filter: Array4D, convInfo: ConvInfo): Array3D {
    const inDepth = filter.shape[2];
    const outDepth = filter.shape[3];
    const yRows = dy.shape[0];
    const yCols = dy.shape[1];
    const filterHeight = filter.shape[0];
    const filterWidth = filter.shape[1];
    const topPad = filterHeight - 1 - convInfo.padInfo.top;
    const leftPad = filterWidth - 1 - convInfo.padInfo.left;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;

    const dx = Array3D.zeros(convInfo.inShape);
    for (let d1 = 0; d1 < inDepth; ++d1) {
      for (let xR = 0; xR < dx.shape[0]; ++xR) {
        const xRCorner = xR - leftPad;
        const xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
        const yRMax = Math.min(yRows, (filterHeight + xRCorner) / strideHeight);

        for (let xC = 0; xC < dx.shape[1]; ++xC) {
          const xCCorner = xC - topPad;
          const xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
          const yCMax = Math.min(yCols, (filterWidth + xCCorner) / strideWidth);

          let dotProd = 0;
          for (let yR = xRMin; yR < yRMax; ++yR) {
            const wR = yR * strideHeight - xRCorner;

            for (let yC = xCMin; yC < yCMax; ++yC) {
              const wC = yC * strideWidth - xCCorner;

              for (let d2 = 0; d2 < outDepth; ++d2) {
                const pixel = dy.get(yR, yC, d2);
                const weight = filter.get(
                    filterHeight - 1 - wR, filterWidth - 1 - wC, d1, d2);
                dotProd += pixel * weight;
              }
            }
          }
          dx.set(dotProd, xR, xC, d1);
        }
      }
    }
    return dx;
  }

  protected conv2dDerFilterInternal(
      x: Array3D, dY: Array3D, convInfo: ConvInfo): Array4D {
    const inputDepth = x.shape[2];
    const outputDepth = dY.shape[2];
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const weightsShape = conv_util.computeWeightsShape4D(
        inputDepth, outputDepth, filterHeight, filterWidth);
    const dW = Array4D.zeros(weightsShape);

    const yNumRows = dY.shape[0];
    const yNumCols = dY.shape[1];
    const xNumRows = x.shape[0];
    const xNumCols = x.shape[1];

    const leftPad = convInfo.padInfo.left;
    const topPad = convInfo.padInfo.top;

    for (let wR = 0; wR < filterHeight; ++wR) {
      const yRMin = Math.max(0, Math.ceil((topPad - wR) / strideHeight));
      const yRMax = Math.min(yNumRows, (xNumRows + topPad - wR) / strideHeight);

      for (let wC = 0; wC < filterWidth; ++wC) {
        const yCMin = Math.max(0, Math.ceil((leftPad - wC) / strideWidth));
        const yCMax =
            Math.min(yNumCols, (xNumCols + leftPad - wC) / strideWidth);

        for (let d1 = 0; d1 < inputDepth; ++d1) {
          for (let d2 = 0; d2 < outputDepth; ++d2) {
            // Need to convolve.
            let dotProd = 0;
            for (let yR = yRMin; yR < yRMax; ++yR) {
              const xR = wR + yR * strideHeight - topPad;
              for (let yC = yCMin; yC < yCMax; ++yC) {
                const xC = wC + yC * strideWidth - leftPad;
                dotProd += x.get(xR, xC, d1) * dY.get(yR, yC, d2);
              }
            }
            dW.set(dotProd, wR, wC, d1, d2);
          }
        }
      }
    }
    return dW;
  }

  protected conv2dDerBiasInternal(dY: Array3D): Array1D {
    const outputDepth = dY.shape[2];
    const numRows = dY.shape[0];
    const numCols = dY.shape[1];
    const values = new Float32Array(outputDepth);
    for (let d2 = 0; d2 < outputDepth; ++d2) {
      let sum = 0;
      for (let r = 0; r < numRows; ++r) {
        for (let c = 0; c < numCols; ++c) {
          sum += dY.get(r, c, d2);
        }
      }
      values[d2] = sum;
    }
    return Array1D.new(values);
  }

  protected transposeInternal<D extends keyof DataTypes, T extends NDArray<D>>(
      a: T, perm: number[]): T {
    const newShape: number[] = new Array(a.rank);
    for (let i = 0; i < newShape.length; i++) {
      newShape[i] = a.shape[perm[i]];
    }
    const resultValues = new Float32Array(a.size);
    const values = a.getValues();
    const result = NDArray.make(newShape, {values: resultValues}) as T;
    for (let i = 0; i < a.size; ++i) {
      const loc = a.indexToLoc(i);

      // Permute location.
      const newLoc: number[] = new Array(loc.length);
      for (let i = 0; i < newLoc.length; i++) {
        newLoc[i] = loc[perm[i]];
      }

      const newIndex = result.locToIndex(newLoc);
      resultValues[newIndex] = values[i];
    }
    return result;
  }

  private pool(x: Array3D, convInfo: ConvInfo, poolType: 'max'|'min'|'avg') {
    const [xRows, xCols, depth] = x.shape;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const y = Array3D.zeros(convInfo.outShape);
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    for (let d = 0; d < depth; ++d) {
      for (let yR = 0; yR < y.shape[0]; ++yR) {
        const xRCorner = yR * strideHeight - padTop;
        const xRMin = Math.max(0, xRCorner);
        const xRMax = Math.min(xRows, filterHeight + xRCorner);
        for (let yC = 0; yC < y.shape[1]; ++yC) {
          const xCCorner = yC * strideWidth - padLeft;
          const xCMin = Math.max(0, xCCorner);
          const xCMax = Math.min(xCols, filterWidth + xCCorner);

          let minMaxValue =
              (poolType === 'max' ? Number.NEGATIVE_INFINITY :
                                    Number.POSITIVE_INFINITY);
          let avgValue = 0;

          for (let xR = xRMin; xR < xRMax; ++xR) {
            for (let xC = xCMin; xC < xCMax; ++xC) {
              const pixel = x.get(xR, xC, d);
              if (isNaN(pixel)) {
                minMaxValue = NaN;
                avgValue = NaN;
                break;
              }
              if ((poolType === 'max' && pixel > minMaxValue) ||
                  (poolType === 'min' && pixel < minMaxValue)) {
                minMaxValue = pixel;
              } else if (poolType === 'avg') {
                avgValue += pixel / (filterHeight * filterWidth);
              }
            }
            if (isNaN(minMaxValue)) {
              break;
            }
          }
          y.set(poolType === 'avg' ? avgValue : minMaxValue, yR, yC, d);
        }
      }
    }
    return y;
  }

  protected maxPoolInternal(x: Array3D, convInfo: ConvInfo): Array3D {
    return this.pool(x, convInfo, 'max');
  }

  maxPoolPositions(x: Array3D, convInfo: ConvInfo) {
    const [xRows, xCols, depth] = x.shape;
    const outputShape = convInfo.outShape;
    const maxPositions = Array3D.zeros(outputShape);
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;

    for (let d = 0; d < depth; ++d) {
      for (let yR = 0; yR < outputShape[0]; ++yR) {
        const xRCorner = yR * strideHeight - padTop;
        const xRMin = Math.max(0, xRCorner);
        const xRMax = Math.min(xRows, filterHeight + xRCorner);
        for (let yC = 0; yC < outputShape[1]; ++yC) {
          const xCCorner = yC * strideWidth - padLeft;
          const xCMin = Math.max(0, xCCorner);
          const xCMax = Math.min(xCols, filterWidth + xCCorner);
          let maxValue = Number.NEGATIVE_INFINITY;
          let maxPosition = -1;
          for (let xR = xRMin; xR < xRMax; ++xR) {
            const wR = xR - xRCorner;
            for (let xC = xCMin; xC < xCMax; ++xC) {
              const wC = xC - xCCorner;
              const pixel = x.get(xR, xC, d);
              if (pixel > maxValue) {
                maxValue = pixel;
                maxPosition = wR * filterWidth + wC;
              }
            }
          }
          maxPositions.set(maxPosition, yR, yC, d);
        }
      }
    }
    return maxPositions;
  }

  protected maxPoolBackpropInternal(
      dy: Array3D, x: Array3D, convInfo: ConvInfo): Array3D {
    const maxPositions = this.maxPoolPositions(x, convInfo);
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const padLeft = filterWidth - 1 - convInfo.padInfo.left;
    const padTop = filterHeight - 1 - convInfo.padInfo.top;
    const [dyRows, dyCols, depth] = dy.shape;
    const dx = Array3D.zeros(x.shape);

    for (let d = 0; d < depth; ++d) {
      for (let dxR = 0; dxR < dx.shape[0]; ++dxR) {
        for (let dxC = 0; dxC < dx.shape[1]; ++dxC) {
          // Shader code begins.
          const dyRCorner = dxR - padTop;
          const dyCCorner = dxC - padLeft;
          let dotProd = 0;
          for (let wR = 0; wR < filterHeight; ++wR) {
            const dyR = (dyRCorner + wR) / strideHeight;
            if (dyR < 0 || dyR >= dyRows || Math.floor(dyR) !== dyR) {
              continue;
            }
            for (let wC = 0; wC < filterWidth; ++wC) {
              const dyC = (dyCCorner + wC) / strideWidth;
              if (dyC < 0 || dyC >= dyCols || Math.floor(dyC) !== dyC) {
                continue;
              }
              const maxPos = filterHeight * filterWidth - 1 -
                  maxPositions.get(dyR, dyC, d);
              const curPos = wR * filterWidth + wC;

              const mask = maxPos === curPos ? 1 : 0;
              if (mask === 0) {
                continue;
              }

              const pixel = dy.get(dyR, dyC, d);
              dotProd += pixel * mask;
            }
          }
          dx.set(dotProd, dxR, dxC, d);
        }
      }
    }
    return dx;
  }

  protected minPoolInternal(x: Array3D, convInfo: ConvInfo): Array3D {
    return this.pool(x, convInfo, 'min');
  }

  protected avgPoolInternal(x: Array3D, convInfo: ConvInfo): Array3D {
    return this.pool(x, convInfo, 'avg');
  }

  protected resizeBilinear3DInternal(
      x: Array3D, newShape2D: [number, number],
      alignCorners: boolean): Array3D {
    const output = Array3D.zeros([newShape2D[0], newShape2D[1], x.shape[2]]);

    const effectiveInputSize =
        alignCorners ? [x.shape[0] - 1, x.shape[1] - 1, x.shape[2]] : x.shape;
    const effectiveOutputSize = alignCorners ?
        [output.shape[0] - 1, output.shape[1] - 1, output.shape[2]] :
        output.shape;
    for (let r = 0; r < output.shape[0]; r++) {
      for (let c = 0; c < output.shape[1]; c++) {
        for (let d = 0; d < output.shape[2]; d++) {
          // Begin shader.

          // Compute the fractional index of the source.
          const sourceFracRow =
              (effectiveInputSize[0]) * r / (effectiveOutputSize[0]);
          const sourceFracCol =
              (effectiveInputSize[1]) * c / (effectiveOutputSize[1]);

          const sourceRowFloor = Math.floor(sourceFracRow);
          const sourceRowCeil =
              Math.min(x.shape[0] - 1, Math.ceil(sourceFracRow));
          const sourceColFloor = Math.floor(sourceFracCol);
          const sourceColCeil =
              Math.min(x.shape[1] - 1, Math.ceil(sourceFracCol));

          const topLeft = x.get(sourceRowFloor, sourceColFloor, d);
          const bottomLeft = x.get(sourceRowCeil, sourceColFloor, d);
          const topRight = x.get(sourceRowFloor, sourceColCeil, d);
          const bottomRight = x.get(sourceRowCeil, sourceColCeil, d);

          const rowFrac = sourceFracRow - sourceRowFloor;
          const colFrac = sourceFracCol - sourceColFloor;

          const top = topLeft + (topRight - topLeft) * colFrac;
          const bottom = bottomLeft + (bottomRight - bottomLeft) * colFrac;
          const newValue = top + (bottom - top) * rowFrac;

          output.set(newValue, r, c, d);
        }
      }
    }

    return output;
  }

  protected batchNormalization3DInternal(
      x: Array3D, mean: Array3D|Array1D, variance: Array3D|Array1D,
      varianceEpsilon = .001, scale?: Array3D|Array1D,
      offset?: Array3D|Array1D): Array3D {
    const xValues = x.getValues();
    const meanValues = mean.getValues();
    const varianceValues = variance.getValues();
    const scaleValues = scale ? scale.getValues() : new Float32Array([1]);
    const offsetValues = offset ? offset.getValues() : new Float32Array([0]);
    const outValues = new Float32Array(xValues.length);

    for (let i = 0; i < xValues.length; i++) {
      outValues[i] = offsetValues[i % offsetValues.length] +
          (xValues[i] - meanValues[i % meanValues.length]) *
              scaleValues[i % scaleValues.length] /
              Math.sqrt(
                  varianceValues[i % varianceValues.length] + varianceEpsilon);
    }
    return Array3D.new(x.shape, outValues);
  }

  protected multinomialInternal(
      probabilities: Array2D, numSamples: number,
      seed: number): Array2D<'int32'> {
    const batchSize = probabilities.shape[0];
    const numEvents = probabilities.shape[1];
    const res = Array2D.zeros([batchSize, numSamples], 'int32');
    const resVals = res.getValues();
    const probVals = probabilities.getValues();

    for (let b = 0; b < batchSize; ++b) {
      const offset = b * numEvents;
      // The cdf won't include the last event. It will be implicit if no other
      // event happened.
      const cdf = new Float32Array(numEvents - 1);
      cdf[0] = probVals[offset];
      for (let event = 1; event < cdf.length; ++event) {
        cdf[event] = cdf[event - 1] + probVals[offset + event];
      }

      const random = seedrandom.alea(seed.toString());
      const outOffset = b * numSamples;
      for (let sampleId = 0; sampleId < numSamples; ++sampleId) {
        const r = random();

        // Assume last event happened by default.
        resVals[outOffset + sampleId] = cdf.length;

        for (let event = 0; event < cdf.length; event++) {
          if (r < cdf[event]) {
            resVals[outOffset + sampleId] = event;
            break;
          }
        }
      }
    }
    return res;
  }

  protected oneHotInternal(
      indices: Array1D, depth: number, onValue: number,
      offValue: number): Array2D {
    const res = new Float32Array(indices.size * depth);
    res.fill(offValue);

    for (let event = 0; event < indices.size; ++event) {
      res[event * depth + indices.get(event)] = onValue;
    }
    return Array2D.new([indices.size, depth], res);
  }

  private broadcastedBinaryOp<D extends keyof DataTypes>(
      a: NDArray, b: NDArray, dtype: D,
      op: (a: number, b: number) => number): NDArray<D> {
    const newShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const result = NDArray.zeros(newShape, dtype);
    const newValues = result.getValues();
    const aValues = a.getValues();
    const bValues = b.getValues();

    const aBroadcastDims = broadcast_util.getBroadcastDims(a.shape, newShape);
    const bBroadcastDims = broadcast_util.getBroadcastDims(b.shape, newShape);

    for (let i = 0; i < newValues.length; ++i) {
      const loc = result.indexToLoc(i);

      const aLoc = loc.slice(-a.rank);
      aBroadcastDims.forEach(d => aLoc[d] = 0);
      const aIndex = a.locToIndex(aLoc);

      const bLoc = loc.slice(-b.rank);
      bBroadcastDims.forEach(d => bLoc[d] = 0);
      const bIndex = b.locToIndex(bLoc);

      newValues[i] = op(aValues[aIndex], bValues[bIndex]);
    }
    return result;
  }
}
