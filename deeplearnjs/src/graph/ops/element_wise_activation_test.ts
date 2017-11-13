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
import * as test_util from '../../test_util';
import {NDArrayMathCPU} from '../../math/math_cpu';
import {Array1D, Array2D} from '../../math/ndarray';
import {Tensor} from '../graph';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';

import {ReLU, Sigmoid, Square, TanH} from './element_wise_activation';

describe('Element wise activation', () => {
  let math: NDArrayMathCPU;
  let xTensor: Tensor;
  let yTensor: Tensor;
  let activations: TensorArrayMap;
  let gradients: SummedTensorArrayMap;

  beforeEach(() => {
    math = new NDArrayMathCPU();
    activations = new TensorArrayMap();
    gradients = new SummedTensorArrayMap(math);
  });

  afterEach(() => {
    activations.disposeArray(xTensor);
    activations.disposeArray(yTensor);
    gradients.disposeArray(xTensor);
    gradients.disposeArray(yTensor);
  });

  it('ReLU', () => {
    const x = Array2D.new([2, 3], [3, 0, -1, 2, 9, -5]);

    xTensor = new Tensor(x.shape);
    yTensor = new Tensor(x.shape);
    activations.set(xTensor, x);

    const op = new ReLU(xTensor, yTensor);
    op.feedForward(math, activations);

    const y = activations.get(yTensor);
    expect(y.getValues()).toEqual(new Float32Array([3, 0, 0, 2, 9, 0]));

    // Backprop.
    const dy = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    gradients.add(yTensor, dy);

    op.backProp(math, activations, gradients);

    const dx = gradients.get(xTensor);

    expect(dx.getValues()).toEqual(new Float32Array([1, 0, 0, 4, 5, 0]));
  });

  it('TanH', () => {
    const x = Array1D.new([3, 0, -3]);

    xTensor = new Tensor(x.shape);
    yTensor = new Tensor(x.shape);
    activations.set(xTensor, x);

    const op = new TanH(xTensor, yTensor);
    op.feedForward(math, activations);

    const y = activations.get(yTensor);

    test_util.expectNumbersClose(y.get(0), 0.99505475);
    test_util.expectNumbersClose(y.get(1), 0);
    test_util.expectNumbersClose(y.get(2), -0.99505475);

    // Backprop.
    const dy = Array1D.new([2, 4, 3]);
    gradients.add(yTensor, dy);

    op.backProp(math, activations, gradients);

    const dx = gradients.get(xTensor);
    test_util.expectNumbersClose(dx.get(0), 2 * (1 - 0.99505475 * 0.99505475));
    test_util.expectNumbersClose(dx.get(1), 4);
    test_util.expectNumbersClose(dx.get(2), 3 * (1 - 0.99505475 * 0.99505475));
  });

  it('Sigmoid', () => {
    const x = Array1D.new([3, 0, -3]);

    xTensor = new Tensor(x.shape);
    yTensor = new Tensor(x.shape);
    activations.set(xTensor, x);

    const op = new Sigmoid(xTensor, yTensor);
    op.feedForward(math, activations);

    const y = activations.get(yTensor);
    test_util.expectNumbersClose(y.get(0), 0.9525741268);
    test_util.expectNumbersClose(y.get(1), 0.5);
    test_util.expectNumbersClose(y.get(2), 0.0474258731);

    // Backprop.
    const dy = Array1D.new([2, 4, 3]);
    gradients.add(yTensor, dy);

    op.backProp(math, activations, gradients);

    const dx = gradients.get(xTensor);
    test_util.expectNumbersClose(
      dx.get(0),
      2 * 0.9525741268 * (1 - 0.9525741268)
    );
    test_util.expectNumbersClose(dx.get(1), 4 * 0.5 * 0.5);
    test_util.expectNumbersClose(
      dx.get(2),
      3 * 0.0474258731 * (1 - 0.0474258731)
    );
  });

  it('Square', () => {
    const x = Array1D.new([2, 0, -3]);

    xTensor = new Tensor(x.shape);
    yTensor = new Tensor(x.shape);
    activations.set(xTensor, x);

    const op = new Square(xTensor, yTensor);
    op.feedForward(math, activations);

    const y = activations.get(yTensor);
    expect(y.getValues()).toEqual(new Float32Array([4, 0, 9]));

    // Backprop.
    const dy = Array1D.new([1, 2, 3]);
    gradients.add(yTensor, dy);

    op.backProp(math, activations, gradients);

    const dx = gradients.get(xTensor);
    expect(dx.get(0)).toBe(2 * x.get(0) * dy.get(0));
    expect(dx.get(1)).toBe(2 * x.get(1) * dy.get(1));
    expect(dx.get(2)).toBe(2 * x.get(2) * dy.get(2));
  });
});
