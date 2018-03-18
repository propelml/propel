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

import { ENV } from "../../environment";
import * as util from "../../util";
import { TypedArray } from "../../util";
import * as axis_util from "../axis_util";
import { Conv2DInfo } from "../conv_util";
import { Array1D, Array2D, Array3D, Array4D, DataId, DataType, DataTypeMap,
  IntDType, NDArray, Rank } from "../ndarray";
import * as reduce_util from "../reduce_util";
import * as types from "../types";
import { SumTypes, SumTypesMap } from "../types";
import { MathBackend } from "./backend";
import { ArgMinMaxProgram } from "./webgl/argminmax_gpu";
import * as binaryop_gpu from "./webgl/binaryop_gpu";
import { BinaryOpProgram } from "./webgl/binaryop_gpu";
import { ClipProgram } from "./webgl/clip_gpu";
import { ConcatProgram } from "./webgl/concat_gpu";
// tslint:disable-next-line:max-line-length
import { Conv2DDerBiasProgram, Conv2DDerFilterProgram, Conv2DDerInputProgram } from "./webgl/conv_backprop_gpu";
import { Conv2DProgram } from "./webgl/conv_gpu";
import { DepthwiseConv2DProgram } from "./webgl/conv_gpu_depthwise";
import { Copy2DProgram } from "./webgl/copy_gpu";
import { GatherProgram } from "./webgl/gather_gpu";
import { GPGPUContext } from "./webgl/gpgpu_context";
import { ArrayData, GPGPUBinary, GPGPUProgram } from "./webgl/gpgpu_math";
import * as gpgpu_math from "./webgl/gpgpu_math";
import * as gpgpu_util from "./webgl/gpgpu_util";
import { MaxPool2DBackpropProgram } from "./webgl/max_pool_backprop_gpu";
import { MatMulProgram } from "./webgl/mulmat_gpu";
import { MultinomialProgram } from "./webgl/multinomial_gpu";
import { OneHotProgram } from "./webgl/onehot_gpu";
import { PadProgram } from "./webgl/pad_gpu";
import { Pool2DProgram } from "./webgl/pool_gpu";
import { ReduceProgram } from "./webgl/reduce_gpu";
import { ResizeBilinear3DProgram } from "./webgl/resize_bilinear_gpu";
import { SliceProgram } from "./webgl/slice_gpu";
import { TextureData, TextureType } from "./webgl/tex_util";
import { TextureManager } from "./webgl/texture_manager";
import { TileProgram } from "./webgl/tile_gpu";
import { TransposeProgram } from "./webgl/transpose_gpu";
import * as unary_op from "./webgl/unaryop_gpu";
import { UnaryOpProgram } from "./webgl/unaryop_gpu";
import * as webgl_util from "./webgl/webgl_util";

export class MathBackendWebGL implements MathBackend {
  private texData = new WeakMap<DataId, TextureData>();
  private canvas: HTMLCanvasElement;

  register(dataId: DataId, shape: number[], dtype: DataType): void {
    if (this.texData.has(dataId)) {
      throw new Error(`data id ${dataId} already registered`);
    }
    this.texData.set(dataId, {
      shape,
      dtype,
      values: null,
      texture: null,
      textureShape: null,
      textureType: null
    });
  }
  writePixels(
      dataId: DataId,
      pixels: ImageData | HTMLImageElement | HTMLCanvasElement |
              HTMLVideoElement,
      numChannels: number): void {
    if (pixels == null) {
      throw new Error("MathBackendWebGL.writePixels(): pixels can not be null");
    }
    this.throwIfNoData(dataId);
    const textureShape: [number, number] = [pixels.height, pixels.width];
    const texture = this.texData.get(dataId).texture ||
        this.textureManager.acquireTexture(textureShape);
    const {shape} = this.texData.get(dataId);

    this.texData.set(dataId, {
      shape,
      dtype: "int32",
      values: null,
      texture,
      textureShape,
      textureType: TextureType.RGBA_COLOR,
      numChannels
    });
    if (pixels instanceof HTMLVideoElement) {
      if (this.canvas == null) {
        throw new Error(
            "Can't read pixels from HTMLImageElement outside " +
            "the browser.");
      }
      this.canvas.width = pixels.width;
      this.canvas.height = pixels.height;
      this.canvas.getContext("2d").drawImage(
          pixels, 0, 0, pixels.width, pixels.height);
      pixels = this.canvas;
    }
    // Pixel data is immediate storage since it already lives on gpu.
    this.gpgpu.uploadPixelDataToTexture(texture, pixels);
  }
  write<D extends DataType>(dataId: DataId, values: DataTypeMap[D]): void {
    if (values == null) {
      throw new Error("MathBackendWebGL.write(): values can not be null");
    }
    this.throwIfNoData(dataId);

    const {texture, textureShape} = this.texData.get(dataId);
    if (texture != null) {
      // Release texture, because it is now out of sync. A new texture will be
      // created when a GPU program needs it.
      this.textureManager.releaseTexture(texture, textureShape);
      Object.assign(this.texData.get(dataId), {
        texture: null,
        textureShape: null,
        textureType: null
      });
    }
    this.texData.get(dataId).values = values;
  }

  readSync<D extends DataType>(dataId: DataId): DataTypeMap[D] {
    this.throwIfNoData(dataId);
    const {texture, values, textureType, textureShape, numChannels} =
        this.texData.get(dataId);
    if (values != null) {
      this.cacheOnCPU(dataId);
      return values;
    }
    let float32Values: Float32Array;
    if (textureType === TextureType.DEFAULT) {
      float32Values = this.gpgpu.downloadMatrixFromTexture(
          texture, textureShape[0], textureShape[1]);
    } else {
      float32Values = this.gpgpu.downloadMatrixFromRGBAColorTexture(
          texture, textureShape[0], textureShape[1], numChannels);
    }
    this.cacheOnCPU(dataId, float32Values);
    return this.texData.get(dataId).values;
  }
  async read<D extends DataType>(dataId: DataId): Promise<DataTypeMap[D]> {
    this.throwIfNoData(dataId);
    const {texture, values, textureType, textureShape} =
        this.texData.get(dataId);
    if (values != null) {
      this.cacheOnCPU(dataId);
      return values;
    }
    if (ENV.get("WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED") &&
        textureType === TextureType.DEFAULT) {
      const float32Values = await this.gpgpu.downloadMatrixFromTextureAsync(
          texture, textureShape[0], textureShape[1]);
      this.cacheOnCPU(dataId, float32Values);
      return this.texData.get(dataId).values;
    }

    if (!ENV.get("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED")) {
      return this.readSync(dataId);
    }

    // Construct an empty query. We're just interested in getting a callback
    // when the GPU command queue has executed until this point in time.
    await this.gpgpu.runQuery(() => {});
    return this.readSync(dataId);
  }
  async time(query: () => NDArray): Promise<number> {
    if (!ENV.get("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED")) {
      const start = performance.now();
      const a = query();
      await a.data();
      return performance.now() - start;
    }
    return this.gpgpu.runQuery(query);
  }
  disposeData(dataId: DataId): void {
    if (this.texData.has(dataId)) {
      const {texture, textureShape} = this.texData.get(dataId);
      if (texture != null) {
        this.textureManager.releaseTexture(texture, textureShape);
      }
      this.texData.delete(dataId);
    }
  }

  getTexture(dataId: DataId): WebGLTexture {
    this.uploadToGPU(dataId);
    return this.texData.get(dataId).texture;
  }

  getTextureData(dataId: DataId): TextureData {
    this.uploadToGPU(dataId);
    return this.texData.get(dataId);
  }

  private textureManager: TextureManager;
  private binaryCache: {[key: string]: GPGPUBinary} = {};
  private gpgpuCreatedLocally: boolean;

  constructor(private gpgpu?: GPGPUContext) {
    if (ENV.get("WEBGL_VERSION") < 1) {
      throw new Error("WebGL is not supported on this device");
    }
    if (gpgpu == null) {
      const gl = gpgpu_util.createWebGLContext();
      this.gpgpu = new GPGPUContext(gl);
      this.gpgpuCreatedLocally = true;
    } else {
      this.gpgpuCreatedLocally = false;
    }
    if (typeof document !== "undefined") {
      this.canvas = document.createElement("canvas");
    }
    this.textureManager = new TextureManager(this.gpgpu);
  }

  getGPGPUContext(): GPGPUContext {
    return this.gpgpu;
  }

  clone<D extends DataType, T extends NDArray<D>>(x: T): T {
    this.throwIfNoData(x.dataId);
    this.uploadToGPU(x.dataId);
    const {textureShape} = this.texData.get(x.dataId);
    // Pretend the source was in logical shape that matches the texture shape.
    const source = x.as2D(textureShape[0], textureShape[1]);
    // Do the same for output.
    const output = this.makeOutputArray<D, Array2D<D>>(textureShape, x.dtype);
    this.copy2D(source, [0, 0], textureShape, output, [0, 0], textureShape);
    // Get back to the original logical shape.
    return output.reshape(x.shape) as T;
  }

  gather(x: NDArray, indices: Array1D<"int32">, axis: number): NDArray {
    const program = new GatherProgram(x.shape, indices.shape[0], axis);
    return this.compileAndRun(program, [x, indices]);
  }

  pad(x: NDArray, paddings: Array<[number, number]>,
      padValue: number): NDArray {
    const program = new PadProgram(x.shape, paddings, padValue);
    return this.compileAndRun(program, [x]);
  }

  slice1D(x: Array1D, begin: number, size: number): Array1D {
    const program = new SliceProgram([size]);
    const customSetup = program.getCustomSetupFunc([begin]);
    return this.compileAndRun(program, [x], null, customSetup);
  }

  slice2D(x: Array2D, begin: [number, number], size: [number, number]):
      Array2D {
    const program = new SliceProgram(size);
    const customSetup = program.getCustomSetupFunc(begin);
    return this.compileAndRun(program, [x], null, customSetup);
  }

  slice3D(x: Array3D, begin: [number, number, number], size: [
    number, number, number
  ]): Array3D {
    const program = new SliceProgram(size);
    const customSetup = program.getCustomSetupFunc(begin);
    return this.compileAndRun(program, [x], null, customSetup);
  }

  slice4D(x: Array4D, begin: [number, number, number, number], size: [
    number, number, number, number
  ]): Array4D {
    const program = new SliceProgram(size);
    const customSetup = program.getCustomSetupFunc(begin);
    return this.compileAndRun(program, [x], null, customSetup);
  }

  private copy2D(
      source: Array2D, sourceBeginRowCol: [number, number],
      sourceSizeRowCol: [number, number], dest: Array2D,
      destBeginRowCol: [number, number],
      destSizeRowCol: [number, number]): void {
    const program = new Copy2DProgram(sourceSizeRowCol[1], destSizeRowCol[1]);
    const customSetup = program.getCustomSetupFunc(
        sourceBeginRowCol, destBeginRowCol, destSizeRowCol);
    this.compileAndRun(program, [source], dest, customSetup);
  }

  concat1D(a: Array1D, b: Array1D): Array1D {
    const program = new ConcatProgram(a.shape, b.shape, 0);
    return this.compileAndRun(program, [a, b]);
  }

  concat2D(a: Array2D, b: Array2D, axis: number): Array2D {
    const program = new ConcatProgram(a.shape, b.shape, axis);
    return this.compileAndRun(program, [a, b]);
  }

  concat3D(a: Array3D, b: Array3D, axis: number): Array3D {
    const program = new ConcatProgram(a.shape, b.shape, axis);
    return this.compileAndRun(program, [a, b]);
  }

  concat4D(a: Array4D, b: Array4D, axis: number): Array4D {
    const program = new ConcatProgram(a.shape, b.shape, axis);
    return this.compileAndRun(program, [a, b]);
  }

  neg<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.NEG);
    return this.compileAndRun(program, [x]) as T;
  }

  matMul(
      a: Array2D, b: Array2D, aOrientation: types.MatrixOrientation,
      bOrientation: types.MatrixOrientation): Array2D {
    const program =
        new MatMulProgram(a.shape, b.shape, aOrientation, bOrientation);
    return this.compileAndRun<Array2D, Array2D>(program, [a, b]);
  }

  multiply<D extends DataType>(a: NDArray<D>, b: NDArray<D>): NDArray<D> {
    const program = new BinaryOpProgram(binaryop_gpu.MUL, a.shape, b.shape);
    const output = this.makeOutputArray(
                       program.outputShape,
                       types.upcastType(a.dtype, b.dtype)) as NDArray<D>;
    return this.compileAndRun(program, [a, b], output) as NDArray<D>;
  }

  tile<D extends DataType, T extends NDArray<D>>(x: T, reps: number[]): T {
    const program = new TileProgram(x.shape, reps);
    return this.compileAndRun(program, [x]);
  }

  transpose<D extends DataType, T extends NDArray<D>>(x: T, perm: number[]): T {
    const program = new TransposeProgram(x.shape, perm);
    return this.compileAndRun(program, [x]);
  }

  private reduce<D extends DataType>(
      x: Array2D, reduceType: "max" | "min" | "sum", dtype: D): Array2D<D> {
    const batchSize = x.shape[0];
    const inSize = x.shape[1];
    const windowSize = reduce_util.computeOptimalWindowSize(inSize);
    const reduceInfo = {windowSize, inSize, batchSize};
    const program = new ReduceProgram(reduceInfo, reduceType);
    const [rows, cols] = program.outputShape;
    const output =
        this.makeOutputArray(program.outputShape, dtype).as2D(rows, cols);
    this.compileAndRun(program, [x], output);
    // No need to run another GPGPU program.
    if (output.shape[1] === 1) {
      return output;
    }
    return this.reduce(output, reduceType, dtype);
  }

  private argReduce(
      x: Array2D, reduceType: "max" | "min",
      bestIndicesA: Array2D = null): Array2D<"int32"> {
    let batchSize = x.shape[0];
    let inSize = x.shape[1];
    if (bestIndicesA != null) {
      batchSize = bestIndicesA.shape[0];
      inSize = bestIndicesA.shape[1];
    }
    const windowSize = reduce_util.computeOptimalWindowSize(inSize);
    const reduceInfo = {windowSize, inSize, batchSize};
    const program =
        new ArgMinMaxProgram(reduceInfo, reduceType, bestIndicesA == null);
    const [rows, cols] = program.outputShape;
    const output =
        this.makeOutputArray(program.outputShape, "int32").as2D(rows, cols);
    const inputs = [x];
    if (bestIndicesA != null) {
      inputs.push(bestIndicesA);
    }
    this.compileAndRun(program, inputs, output);
    // No need to run another GPGPU program.
    if (output.shape[1] === 1) {
      return output;
    }
    return this.argReduce(x, reduceType, output);
  }

  sum<D extends DataType>(x: NDArray<D>, axes: number[]): NDArray<SumTypes[D]> {
    axis_util.assertAxesAreInnerMostDims("sum", axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    const outputDType = SumTypesMap[x.dtype];
    return this.reduce(a2D, "sum", outputDType).reshape(outShape);
  }

  argMin(x: NDArray, axes: number[]): NDArray<"int32"> {
    axis_util.assertAxesAreInnerMostDims("argMin", axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.argReduce(a2D, "min").reshape(outShape);
  }

  argMax(x: NDArray, axes: number[]): NDArray<"int32"> {
    axis_util.assertAxesAreInnerMostDims("argMax", axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.argReduce(a2D, "max").reshape(outShape);
  }

  equal(a: NDArray, b: NDArray): NDArray<"bool"> {
    const program = new BinaryOpProgram(binaryop_gpu.EQUAL, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, "bool");
    return this.compileAndRun(program, [a, b], output);
  }

  notEqual(a: NDArray, b: NDArray): NDArray<"bool"> {
    const program =
        new BinaryOpProgram(binaryop_gpu.NOT_EQUAL, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, "bool");
    return this.compileAndRun(program, [a, b], output);
  }

  greater(a: NDArray, b: NDArray): NDArray<"bool"> {
    const program = new BinaryOpProgram(binaryop_gpu.GREATER, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, "bool");
    return this.compileAndRun(program, [a, b], output);
  }

  greaterEqual(a: NDArray, b: NDArray): NDArray<"bool"> {
    const program =
        new BinaryOpProgram(binaryop_gpu.GREATER_EQUAL, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, "bool");
    return this.compileAndRun(program, [a, b], output);
  }

  less(a: NDArray, b: NDArray): NDArray<"bool"> {
    const program = new BinaryOpProgram(binaryop_gpu.LESS, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, "bool");
    return this.compileAndRun(program, [a, b], output);
  }

  lessEqual(a: NDArray, b: NDArray): NDArray<"bool"> {
    const program =
        new BinaryOpProgram(binaryop_gpu.LESS_EQUAL, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, "bool");
    return this.compileAndRun(program, [a, b], output);
  }

  select(cond: NDArray<"bool">, a: NDArray, b: NDArray): NDArray {
    throw new Error("select GPU not yet implemented!");
  }

  topKValues<D extends DataType, T extends NDArray<D>>(x: T, k: number):
      Array1D<D> {
    throw new Error("topKValues GPU not yet implemented!");
  }

  topKIndices(x: NDArray, k: number): Array1D<"int32"> {
    throw new Error("topKIndices GPU not yet implemented!");
  }

  min<D extends DataType>(x: NDArray<D>, axes: number[]): NDArray<D> {
    axis_util.assertAxesAreInnerMostDims("min", axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.reduce(a2D, "min", a2D.dtype).reshape(outShape);
  }

  minimum<D extends DataType>(a: NDArray<D>, b: NDArray<D>): NDArray<D> {
    const program = new BinaryOpProgram(binaryop_gpu.MIN, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  max<D extends DataType>(x: NDArray<D>, axes: number[]): NDArray<D> {
    axis_util.assertAxesAreInnerMostDims("max", axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = x.as2D(-1, inSize);
    return this.reduce(a2D, "max", a2D.dtype).reshape(outShape);
  }

  maximum<D extends DataType>(a: NDArray<D>, b: NDArray<D>): NDArray<D> {
    const program = new BinaryOpProgram(binaryop_gpu.MAX, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]);
  }

  divide(a: NDArray, b: NDArray): NDArray<"float32"> {
    const program = new BinaryOpProgram(binaryop_gpu.DIV, a.shape, b.shape);
    const output = this.makeOutputArray(program.outputShape, "float32");
    return this.compileAndRun<NDArray, NDArray<"float32">>(
        program, [a, b], output);
  }

  add<D extends DataType>(a: NDArray<D>, b: NDArray<D>): NDArray<D> {
    const program = new BinaryOpProgram(binaryop_gpu.ADD, a.shape, b.shape);
    const output = this.makeOutputArray(
                       program.outputShape,
                       types.upcastType(a.dtype, b.dtype)) as NDArray<D>;
    return this.compileAndRun<NDArray, NDArray<D>>(program, [a, b], output);
  }

  subtract<D extends DataType>(a: NDArray<D>, b: NDArray<D>): NDArray<D> {
    const program = new BinaryOpProgram(binaryop_gpu.SUB, a.shape, b.shape);
    const output = this.makeOutputArray(
                       program.outputShape,
                       types.upcastType(a.dtype, b.dtype)) as NDArray<D>;
    return this.compileAndRun<NDArray, NDArray<D>>(program, [a, b], output);
  }

  pow<T extends NDArray>(a: T, b: NDArray): T {
    const program = new BinaryOpProgram(binaryop_gpu.POW, a.shape, b.shape);
    const output =
        this.makeOutputArray(
            program.outputShape, types.upcastType(a.dtype, b.dtype)) as T;
    return this.compileAndRun<NDArray, T>(program, [a, b], output);
  }

  ceil<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.CEIL);
    return this.compileAndRun(program, [x]) as T;
  }

  floor<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.FLOOR);
    return this.compileAndRun(program, [x]) as T;
  }

  exp<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.EXP);
    return this.compileAndRun(program, [x]) as T;
  }

  log<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.LOG);
    return this.compileAndRun(program, [x]) as T;
  }

  sqrt<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SQRT);
    return this.compileAndRun(program, [x]) as T;
  }

  square<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SQUARE);
    return this.compileAndRun(program, [x]) as T;
  }

  relu<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.RELU);
    return this.compileAndRun(program, [x]) as T;
  }

  elu<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ELU);
    return this.compileAndRun(program, [x]) as T;
  }

  eluDer<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ELU_DER);
    return this.compileAndRun(program, [x]) as T;
  }

  selu<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SELU);
    return this.compileAndRun(program, [x]) as T;
  }

  leakyRelu<T extends NDArray>(x: T, alpha: number): T {
    const program = new UnaryOpProgram(x.shape, unary_op.LEAKY_RELU(alpha));
    return this.compileAndRun(program, [x]) as T;
  }

  prelu<T extends NDArray>(a: T, b: T): T {
    const program = new BinaryOpProgram(binaryop_gpu.PRELU, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]) as T;
  }

  preluDer<T extends NDArray>(a: T, b: T): T {
    const program =
        new BinaryOpProgram(binaryop_gpu.PRELU_DER, a.shape, b.shape);
    return this.compileAndRun(program, [a, b]) as T;
  }

  int<D extends IntDType, R extends Rank>(
      x: NDArray<DataType, R>, dtype: D): NDArray<D, R> {
    const program = new UnaryOpProgram(x.shape, unary_op.TO_INT);
    const output = this.makeOutputArray(program.outputShape, dtype);
    return this.compileAndRun(program, [x], output) as NDArray<D, R>;
  }

  clip<T extends NDArray>(x: T, min: number, max: number): T {
    const program = new ClipProgram(x.shape, min, max);
    return this.compileAndRun(program, [x]) as T;
  }

  abs<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ABS);
    return this.compileAndRun(program, [x]) as T;
  }

  sigmoid<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SIGMOID);
    return this.compileAndRun(program, [x]) as T;
  }

  sin<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SIN);
    return this.compileAndRun(program, [x]) as T;
  }

  cos<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.COS);
    return this.compileAndRun(program, [x]) as T;
  }

  tan<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.TAN);
    return this.compileAndRun(program, [x]) as T;
  }

  asin<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ASIN);
    return this.compileAndRun(program, [x]) as T;
  }

  acos<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ACOS);
    return this.compileAndRun(program, [x]) as T;
  }

  atan<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.ATAN);
    return this.compileAndRun(program, [x]) as T;
  }

  sinh<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.SINH);
    return this.compileAndRun(program, [x]) as T;
  }

  cosh<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.COSH);
    return this.compileAndRun(program, [x]) as T;
  }

  tanh<T extends NDArray>(x: T): T {
    const program = new UnaryOpProgram(x.shape, unary_op.TANH);
    return this.compileAndRun(program, [x]) as T;
  }

  step<T extends NDArray>(x: T, alpha: number): T {
    const program = new UnaryOpProgram(x.shape, unary_op.STEP(alpha));
    return this.compileAndRun(program, [x]) as T;
  }

  conv2d(x: Array4D, filter: Array4D, bias: Array1D | null,
         convInfo: Conv2DInfo): Array4D {
    const program = new Conv2DProgram(convInfo, bias != null);
    const inputs = bias != null ? [x, filter, bias] : [x, filter];
    return this.compileAndRun(program, inputs);
  }

  conv2dDerInput(dy: Array4D, filter: Array4D, convInfo: Conv2DInfo): Array4D {
    const program = new Conv2DDerInputProgram(convInfo);
    return this.compileAndRun(program, [dy, filter]);
  }

  conv2dDerFilter(x: Array4D, dy: Array4D, convInfo: Conv2DInfo): Array4D {
    const program = new Conv2DDerFilterProgram(convInfo);
    return this.compileAndRun(program, [x, dy]);
  }

  conv2dDerBias(dy: Array4D): Array1D {
    const program = new Conv2DDerBiasProgram(dy.shape);
    return this.compileAndRun(program, [dy]);
  }

  depthwiseConv2D(x: Array4D, filter: Array4D, convInfo: Conv2DInfo): Array4D {
    const program = new DepthwiseConv2DProgram(convInfo);
    return this.compileAndRun(program, [x, filter]);
  }

  maxPool(x: Array4D, convInfo: Conv2DInfo): Array4D {
    const program = new Pool2DProgram(convInfo, "max", false);
    return this.compileAndRun(program, [x]);
  }

  minPool(x: Array4D, convInfo: Conv2DInfo): Array4D {
    const program = new Pool2DProgram(convInfo, "min", false);
    return this.compileAndRun(program, [x]);
  }

  avgPool(x: Array4D, convInfo: Conv2DInfo): Array4D {
    const program = new Pool2DProgram(convInfo, "avg", false);
    return this.compileAndRun(program, [x]);
  }

  maxPoolBackprop(dy: Array4D, x: Array4D, convInfo: Conv2DInfo): Array4D {
    const getPositions = true;
    const maxPoolPositionsProgram =
        new Pool2DProgram(convInfo, "max", getPositions);
    const maxPoolPositions: Array4D =
        this.compileAndRun(maxPoolPositionsProgram, [x]);

    const maxPoolBackPropProgram = new MaxPool2DBackpropProgram(convInfo);

    const result =
        this.compileAndRun(maxPoolBackPropProgram, [dy, maxPoolPositions]);
    maxPoolPositions.dispose();
    return result as Array4D;
  }

  resizeBilinear3D(
      x: Array3D, newShape2D: [number, number],
      alignCorners: boolean): Array3D {
    const program =
        new ResizeBilinear3DProgram(x.shape, newShape2D, alignCorners);
    return this.compileAndRun(program, [x]);
  }

  multinomial(probs: Array2D, numSamples: number, seed: number):
      Array2D<"int32"> {
    const batchSize = probs.shape[0];
    const numOutcomes = probs.shape[1];
    const program = new MultinomialProgram(batchSize, numOutcomes, numSamples);
    const output =
        this.makeOutputArray(program.outputShape, "int32") as Array2D<"int32">;
    const customSetup = program.getCustomSetupFunc(seed);
    return this.compileAndRun(program, [probs], output, customSetup);
  }

  oneHot(indices: Array1D, depth: number, onValue: number, offValue: number):
      Array2D {
    const program = new OneHotProgram(indices.size, depth, onValue, offValue);
    return this.compileAndRun(program, [indices]);
  }

  setDiag(input: Array2D, diag: Array1D): Array2D {
    throw new Error("setDiag GPU not yet implemented!");
  }

  private makeOutputArray<D extends DataType, T extends NDArray<D>>(
      shape: number[], dtype: D): T {
    return NDArray.make(shape, {}, dtype) as T;
  }

  private compileAndRun<T extends NDArray, K extends NDArray>(
      program: GPGPUProgram, inputs: T[], output?: K,
      customSetup?: (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => void):
      K {
    if (output == null) {
      output = this.makeOutputArray(program.outputShape, inputs[0].dtype);
    }
    const inputsData: Array<ArrayData<T>> = inputs.map(input => {
      this.uploadToGPU(input.dataId);
      return {array: input, texData: this.texData.get(input.dataId)};
    });
    this.uploadToGPU(output.dataId);
    const outputData =
        {array: output, texData: this.texData.get(output.dataId)};
    const key = gpgpu_math.makeShaderKey(program, inputsData, outputData);
    const binary = this.getAndSaveBinary(key, () => {
      return gpgpu_math.compileProgram(
          this.gpgpu, program, inputsData, outputData);
    });
    gpgpu_math.runProgram(binary, inputsData, outputData, customSetup);
    return output;
  }

  private getAndSaveBinary(key: string, getBinary: () => GPGPUBinary):
      GPGPUBinary {
    if (!(key in this.binaryCache)) {
      this.binaryCache[key] = getBinary();
    }
    return this.binaryCache[key];
  }

  getTextureManager(): TextureManager {
    return this.textureManager;
  }

  dispose() {
    for (const key in this.binaryCache) {
      if (!this.binaryCache.hasOwnProperty(key)) continue;
      this.gpgpu.deleteProgram(this.binaryCache[key].webGLProgram);
    }
    this.textureManager.dispose();

    if (this.gpgpuCreatedLocally) {
      this.gpgpu.dispose();
    }
  }

  private throwIfNoData(dataId: DataId) {
    if (!(this.texData.has(dataId))) {
      throw new Error(
          `No data found for NDArray with data id ${dataId}. ` +
          `Use dl.ENV.math instead of constructing your own NDArrayMath. ` +
          `If you need to construct your own math, make sure this array is ` +
          `allocated after the math construction`);
    }
  }

  private uploadToGPU(dataId: DataId): void {
    this.throwIfNoData(dataId);
    const {shape, values, texture, dtype} = this.texData.get(dataId);
    if (texture != null) {
      // Array is already on GPU. No-op.
      return;
    }
    const textureShape =
        webgl_util.getTextureShapeFromLogicalShape(this.gpgpu.gl, shape);
    this.texData.get(dataId).textureType = TextureType.DEFAULT;
    this.texData.get(dataId).textureShape = textureShape;
    const newTexture = this.textureManager.acquireTexture(textureShape);
    this.texData.get(dataId).texture = newTexture;
    if (values != null) {
      this.gpgpu.uploadMatrixToTexture(
          newTexture, textureShape[0],
          // TODO(smilkov): Propagate the original typed array to gpgpu.
          textureShape[1], typedArrayToFloat32(values, dtype));
    }
  }

  private cacheOnCPU(dataId: DataId, float32Values?: Float32Array) {
    // When the user reads data, don't keep a copy on the gpu, to minimize
    // likelihood of memory leak. We re-upload to gpu the next time a gpgpu
    // program needs the texture.
    const {texture, textureShape, dtype} = this.texData.get(dataId);
    if (texture != null) {
      this.textureManager.releaseTexture(texture, textureShape);
      Object.assign(this.texData.get(dataId), {
        texture: null,
        textureShape: null,
        textureType: null,
      });
    }
    if (float32Values != null) {
      this.texData.get(dataId).values =
          float32ToTypedArray(float32Values, dtype);
    }
  }
}

ENV.registerBackend("webgl", () => new MathBackendWebGL());

function float32ToTypedArray<D extends DataType>(
    a: Float32Array, dtype: D): DataTypeMap[D] {
  if (dtype === "float32") {
    return a;
  } else if (dtype === "int32" || dtype === "uint8" || dtype === "bool") {
    const result = (dtype === "int32") ? new Int32Array(a.length) :
                                         new Uint8Array(a.length);
    for (let i = 0; i < result.length; ++i) {
      let val = a[i];
      val = isNaN(val) ? util.getNaN(dtype) : Math.round(val);
      result[i] = val;
    }
    return result;
  } else {
    throw new Error(`Unknown dtype ${dtype}`);
  }
}

function typedArrayToFloat32(a: TypedArray, dtype: DataType): Float32Array {
  if (a instanceof Float32Array) {
    return a;
  } else {
    const res = new Float32Array(a.length);
    for (let i = 0; i < res.length; i++) {
      const val = a[i];
      res[i] = util.isValNaN(val, dtype) ? NaN : val;
    }
    return res;
  }
}
