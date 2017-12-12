#!/usr/bin/env ts-node

import * as repl from "repl";
import { inspect } from "util";
import * as propel from "./api";

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

// This is currently node.js specific.
// TODO: move to api.js once it can be shared with the browser.
// TODO: make formatting prettier.
propel.Tensor.prototype[inspect.custom] = function(depth, opts) {
  const PRECISION = 3;
  const shape = this.shape;
  const data = this.getData();
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
      }
      s += " ]";
      return s;

    default:
      return "Tensor([" + shape + "])";
  }
};

// Wait for 1ms to allow node and tensorflow to print their junk.
setTimeout(() => {
  const context = repl.start("> ").context;
  context.$ = propel.$;
  context.propel = propel;
}, 1);
