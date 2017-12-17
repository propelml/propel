// TODO This file is Node specific. Adapt for web.
import { existsSync, readFileSync } from "fs";
import { basename, resolve } from "path";
import { $, Tensor } from "./api";
import { assert, assertEqual, log } from "./util";

interface Elements {
  images: Tensor;
  labels: Tensor;
}

// If compiled to JS, this might be in a different directory.
export const dirname = basename(__dirname) === "dist" ?
  resolve(__dirname, "../deps/mnist") :
  resolve(__dirname, "deps/mnist");

export function filenames(split: string): [string, string] {
  if (split === "train") {
    return [
      resolve(dirname, "train-labels-idx1-ubyte"),
      resolve(dirname, "train-images-idx3-ubyte"),
    ];
  } else if (split === "test") {
    return [
      resolve(dirname, "t10k-labels-idx1-ubyte"),
      resolve(dirname, "t10k-images-idx3-ubyte"),
    ];
  } else {
    assert(false, "Bad split: " + split);
  }
}

function littleEndianToBig(val) {
  return ((val & 0x00FF) << 24) |
         ((val & 0xFF00) << 8) |
         ((val >> 8) & 0xFF00) |
         ((val >> 24) & 0x00FF);
}

function bufferToTypedArray(b: any): [Int32Array, Uint8Array] {
  // TODO ensure zero copy if possible.
  // var ab = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  const i32 = new Int32Array(b.buffer, b.byteOffset,
                           b.byteLength / Int32Array.BYTES_PER_ELEMENT);
  const ui8 = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  return [i32, ui8];
}

// TODO Remove once pretty printing lands.
function inspectImg(t, idx) {
  const img = t.slice([idx, 0, 0], [1, -1, -1]);
  console.log("img");
  const imgData = img.getData();
  let s = "";
  for (let j = 0; j < 28 * 28; j++) {
    s += imgData[j].toString() + " ";
    if (j % 28 === 27) s += "\n";
  }
  console.log(s);
}

function loadFile(fn, split, images: boolean) {
  const fileBuffer = readFileSync(fn, null);
  const [i32, ui8] = bufferToTypedArray(fileBuffer);
  const magicValue = images ? 2051 : 2049;
  const numExamples = split === "train" ? 60000 : 10000;
  let i = 0;
  assertEqual(littleEndianToBig(i32[i++]), magicValue);
  assertEqual(littleEndianToBig(i32[i++]), numExamples);
  if (images) {
    assertEqual(littleEndianToBig(i32[i++]), 28);
    assertEqual(littleEndianToBig(i32[i++]), 28);
  }
  const tensorData = ui8.slice(4 * i) as Uint8Array;
  let t = $(tensorData, "uint8");
  if (images) {
    t = t.reshape([numExamples, 28, 28]);
  } else {
    t = t.reshape([numExamples]);
  }
  return t;
}

export function load(split: string, batchSize: number) {
  assert(existsSync(dirname));
  const [labelFn, imageFn] = filenames(split);
  const images = loadFile(imageFn, split, true);
  const labels = loadFile(labelFn, split, false);
  // inspectImg(images, 7);
  const ds = {
    idx: 0,
    next: (): Promise<Elements> => {
      return new Promise((resolve, reject) => {
        if (ds.idx + batchSize >= images.shape[0]) {
          // Wrap around.
          ds.idx = 0;
        }
        const imagesBatch = images.slice([ds.idx, 0, 0],
                                         [batchSize, -1, -1]);
        const labelsBatch = labels.slice([ds.idx], [batchSize]);
        ds.idx += batchSize;
        resolve({
          images: imagesBatch,
          labels: labelsBatch,
        });
      });
    }
  };
  return ds;
}
