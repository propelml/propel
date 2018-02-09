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
// This module is just to implement tensor.toString.
import * as types from "./types";

interface FormatOptions {
  precision: number;
  dtype: types.DType;
  maxBefore: number;
  maxAfter: number;
}

function split(s, precision): [string, string] {
  return s.toFixed(precision).replace(/0+$/, "").split(".", 2);
}

function preprocess(shape: types.Shape, data: types.TypedArray,
                    precision: number): FormatOptions {
  const dtype = types.getDType(data);
  let maxBefore = 0;
  let maxAfter = 0;
  for (let i = 0; i < data.length; i++) {
    const [before, after] = data[i].toFixed(precision)
                                   .replace(/0+$/, "")
                                   .replace(/^-/, "")
                                   .split(".", 2);
    if (maxBefore < before.length) maxBefore = before.length;
    if (maxAfter < after.length) maxAfter = after.length;
  }
  // We removed the negative sign. We always want to leave room for it, even if
  // there are no negative values. So add one more here.
  maxBefore += 1;
  return { precision, dtype, maxBefore, maxAfter };
}

function formatNumber(value: number, opts: FormatOptions): string {
  switch (opts.dtype) {
      case "int32":
      case "uint8":
        return "" + value;
      case "float32":
        const [before, after] = split(value, opts.precision);
        const b = opts.maxBefore - before.length;
        const a = opts.maxAfter - after.length;
        return " ".repeat(b) + before + "." + after + " ".repeat(a);
      default:
        throw new Error("Bad dtype.");
  }
}

export function toString(shape: types.Shape, data: types.TypedArray): string {
  const PRECISION = 3;
  let s;
  const opts = preprocess(shape, data, PRECISION);
  switch (shape.length) {
    case 0:
      return formatNumber(data[0], opts);

    case 1:
      s = "[";
      for (let i = 0; i < shape[0]; i++) {
        s += i === 0 ? "" : ", ";
        s += formatNumber(data[i], opts);
      }
      s += "]";
      return s;

    case 2:
      let w = 1;
      for (let y = 0; y < shape[0]; y++) {
        for (let x = 0; x < shape[1]; x++) {
          const off = y * shape[1] + x;
          const val = formatNumber(data[off], opts);
          if (val.length > w) {
            w = val.length;
          }
        }
      }

      s = "[";
      for (let y = 0; y < shape[0]; y++) {
        s += y === 0 ? "[" : "\n [";
        for (let x = 0; x < shape[1]; x++) {
          const off = y * shape[1] + x;
          const val = formatNumber(data[off], opts);
          s += x === 0 ? "" : ", ";
          s += (val as any).padStart(w);
        }
        s += "]";
        // Add trailing comma to row.
        if (y !== shape[0] - 1) {
          s += ",";
        }
      }
      s += "]";
      return s;

    default:
      return "Tensor([" + shape + "])";
  }
}
