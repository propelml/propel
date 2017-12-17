import { $, Tensor } from "./api";
import { assert, assertEqual, IS_WEB, log } from "./util";

interface Elements {
  images: Tensor;
  labels: Tensor;
}

export function makeHref(fn) {
  if (IS_WEB) {
    return "http://localhost:8000/mnist/" + fn;
  } else {
    // If compiled to JS, this might be in a different directory.
    const path = require("path");
    const dirname = path.basename(__dirname) === "dist" ?
      path.resolve(__dirname, "../deps/mnist") :
      path.resolve(__dirname, "deps/mnist");
    return path.resolve(dirname, fn);
  }
}

export function filenames(split: string): [string, string] {
  if (split === "train") {
    return [
      makeHref("train-labels-idx1-ubyte"),
      makeHref("train-images-idx3-ubyte"),
    ];
  } else if (split === "test") {
    return [
      makeHref("t10k-labels-idx1-ubyte"),
      makeHref("t10k-images-idx3-ubyte"),
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

// TODO Remove once pretty printing lands.
export function inspectImg(t, idx) {
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

async function fetch2(href): Promise<ArrayBuffer> {
  if (IS_WEB) {
    const res = await fetch(href, { mode: "no-cors" });
    return res.arrayBuffer();
  } else {
    const b = require("fs").readFileSync(href, null);
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  }
}

async function loadFile(href, split: string, images: boolean) {
  const ab = await fetch2(href);
  const i32 = new Int32Array(ab);
  const ui8 = new Uint8Array(ab);

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
  const t = $(tensorData, "uint8");
  const shape = images ? [numExamples, 28, 28] : [numExamples];
  return t.reshape(shape);
}

export function load(split: string, batchSize: number) {
  const [labelFn, imageFn] = filenames(split);
  const imagesPromise = loadFile(imageFn, split, true);
  const labelsPromise = loadFile(labelFn, split, false);

  const ds = {
    idx: 0,
    images: null,
    labels: null,
    loadPromise: Promise.all([imagesPromise, labelsPromise]),
    next: (): Promise<Elements> => {
      return new Promise((resolve, reject) => {
        ds.loadPromise.then((_) => {
          if (ds.idx + batchSize >= ds.images.shape[0]) {
            // Wrap around.
            ds.idx = 0;
          }
          const imagesBatch = ds.images.slice([ds.idx, 0, 0],
                                              [batchSize, -1, -1]);
          const labelsBatch = ds.labels.slice([ds.idx], [batchSize]);
          ds.idx += batchSize;
          resolve({
            images: imagesBatch,
            labels: labelsBatch,
          });
        });
      });
    }
  };

  ds.loadPromise.then(([images, labels]) => {
    ds.images = images;
    ds.labels = labels;
  });
  return ds;
}
