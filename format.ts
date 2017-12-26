// This module is just to implement tensor.toString.
import * as types from "./types";

function formatNumber(value: number, precision: number,
                      targetWidth = 0): string {
  const s1 = value.toString();
  const s2 = value.toPrecision(precision);

  if (s1.length <= s2.length && s2.length > targetWidth) {
    return s1;
  } else {
    return s2;
  }
}

export function toString(shape: types.Shape, data: types.TypedArray): string {
  const PRECISION = 3;
  let s;

  switch (shape.length) {
    case 0:
      return "Tensor() " + data[0];

    case 1:
      s = "[";
      for (let i = 0; i < shape[0]; i++) {
        s += i === 0 ? " " : ", ";
        s += formatNumber(data[i], PRECISION);
      }
      s += " ]";
      return s;

    case 2:
      let w = 1;
      for (let y = 0; y < shape[0]; y++) {
        for (let x = 0; x < shape[1]; x++) {
          const off = y * shape[1] + x;
          const val = formatNumber(data[off], PRECISION);
          if (val.length > w) {
            w = val.length;
          }
        }
      }

      s = "[";
      for (let y = 0; y < shape[0]; y++) {
        s += y === 0 ? " [" : "\n  [";
        for (let x = 0; x < shape[1]; x++) {
          const off = y * shape[1] + x;
          const val = formatNumber(data[off], PRECISION, w);
          s += x === 0 ? " " : ", ";
          s += (val as any).padStart(w);
        }
        s += " ]";
        // Add trailing comma to row.
        if (y !== shape[0] - 1) {
          s += ",";
        }
      }
      s += " ]";
      return s;

    default:
      return "Tensor([" + shape + "])";
  }
}
