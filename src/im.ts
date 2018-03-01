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

// This module allows Propel to read PNG and JPG images.

import { Tensor } from "./api";
import { convert } from "./tensor";
import { Mode } from "./types";
import { IS_NODE, nodeRequire } from "./util";

function toTensor(data: Uint8Array, height: number, width: number,
                  mode: Mode): Tensor {
  // Convert a potentially Uint8Array to int32 tensor.
  // TODO Propel does not yet support uinti8 (#306). Ideally we'd return
  // a uint8 tensor here.
  const i32 = new Int32Array(data);
  let tensor = convert(i32, { dtype: "int32" })
    .reshape([height, width, 4])
    .transpose([2, 0, 1]);
  if (mode === "RGBA") {
    return tensor;
  }
  tensor = tensor.slice(0, 3);
  if (mode === "RGB") {
    return tensor;
  }
  if (mode === "L") {
    return tensor.reduceMean([0], true);
  }
  throw new Error("Unsupported convertion mode.");
}

function webImageDecoder(filename: string, mode: Mode)
    : Promise<Tensor> {
  return new Promise((resolve) => {
    try {
      const img = new Image();
      img.onload = function() {
        const canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const image: Tensor = toTensor(
          Uint8Array.from(pixels.data),
          canvas.height,
          canvas.width,
          mode
        );
        resolve(image);
      };
      img.crossOrigin = "Anonymous";
      img.src = filename;
    } catch (e) {
      resolve();
    }
  });
}

const imageSignatures: Array<[number[], string]> = [
  [[0xFF, 0xD8], "image/jpeg"],
  [[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A], "image/png"]
];
const sigMaxLength: number = Math.max(...imageSignatures.map(x => x[0].length));

function readMIME(filename: string): string {
  const fs = nodeRequire("fs");
  // read file header
  const header = new Buffer(sigMaxLength);
  const fd = fs.openSync(filename, "r");
  fs.readSync(fd, header, 0, sigMaxLength);
  fs.closeSync(fd);
  // find matched MIME type
  for (let i = 0; i < imageSignatures.length; ++i) {
    const [sig, type] = imageSignatures[i];
    let ret = true;
    for (let j = 0; j < sig.length; ++j) {
      if (header[j] !== sig[j]) {
        ret = false;
        break;
      }
    }
    if (ret) {
      return type;
    }
  }
  return null;
}

function pngReadHandler(filename: string, mode: Mode): Promise<Tensor> {
  const PNG = nodeRequire("pngjs").PNG;
  const fs = nodeRequire("fs");
  return new Promise(resolve => {
    fs.createReadStream(filename)
      .pipe(new PNG())
      .on("parsed", function(this: any) {
        const data = new Uint8Array(this.data);
        resolve(toTensor(data, this.height, this.width, mode));
      })
      .on("error", function() {
        resolve();
      });
  });
}

function jpegReadHandler(filename: string, mode: Mode): Tensor {
  console.warn("The JPEG decoder is just working and it may not return" +
               " the exact data from image.\nUse it at your own risk.");
  const JPEG = require("jpeg-js");
  const fs = nodeRequire("fs");
  const jpegData = fs.readFileSync(filename);
  const img = JPEG.decode(jpegData, true);
  return toTensor(img.data, img.height, img.width, mode);
}

async function nodeImageDecoder(filename: string, mode: Mode)
    : Promise<Tensor> {
  const MIME = readMIME(filename);
  switch (MIME){
    case "image/png":
      return await pngReadHandler(filename, mode);
    case "image/jpeg":
      return jpegReadHandler(filename, mode);
    default:
      throw new Error("This file is not a valid PNG/JPEG image.");
  }
}

/** Read an image from given file to a 3D tensor in the following
 * The tensor will have the shape [channel, height, width].
 * The second argument specifies if you want "RGBA", "RGB", or "L" (for
 * luminosity/greyscale).
 *
 *    import { imread } from "propel"
 *    img = await imread("/src/testdata/sample.png")
 *    // Average RGBA values
 *    img.reduceMean([1, 2])
 */
export async function imread(filename: string, mode: Mode = "RGBA")
    : Promise<Tensor> {
  if (IS_NODE) {
    return await nodeImageDecoder(filename, mode);
  }
  return await webImageDecoder(filename, mode);
}
