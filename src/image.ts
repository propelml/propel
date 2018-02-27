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
import { IS_NODE, nodeRequire } from "./util";

function toTensor(data: Int32Array, height: number, width: number,
mode: "RGB" | "RGBA"): Tensor {
  const tensor = convert(data)
    .reshape([height, width, 4])
    .transpose([2, 0, 1]);
  if (mode === "RGB") {
    return tensor.slice([0, 0, 0], [3, height, width]);
  }
  return tensor;
}

function WebImageDecoder(filename: string, mode: "RGB" | "RGBA")
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
          new Int32Array(pixels.data.buffer),
          canvas.height,
          canvas.width,
          mode
        );
        resolve(image);
      };
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
    const [sign, type] = imageSignatures[i];
    let ret = true;
    for (let j = 0; j < sign.length; ++j) {
      if (header[j] !== sign[j]) {
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

function PNGHandler(filename: string, mode: "RGB" | "RGBA"): Promise<Tensor> {
  const PNG = nodeRequire("pngjs").PNG;
  const fs = nodeRequire("fs");
  return new Promise(resolve => {
    fs.createReadStream(filename)
      .pipe(new PNG())
      .on("parsed", function(this: any) {
        resolve(toTensor(this.data, this.height, this.width, mode));
      })
      .on("error", function() {
        resolve();
      });
  });
}

function JPEGHandler(filename: string, mode: "RGB" | "RGBA"): Tensor {
  const JPEG = require("jpeg-js");
  const fs = nodeRequire("fs");
  const jpegData = fs.readFileSync(filename);
  const img = JPEG.decode(jpegData);
  return toTensor(img.data, img.height, img.width, mode);
}

async function NodeImageDecoder(filename: string, mode: "RGB" | "RGBA")
: Promise<Tensor> {
  const MIME = readMIME(filename);
  switch (MIME){
    case "image/png":
      return await PNGHandler(filename, mode);
    case "image/jpeg":
      return JPEGHandler(filename, mode);
    default:
      throw new Error("This file is not a valid PNG/JPEG image.");
  }
}

/** Opens an image from given path, returns a new Tensor containg image data in
 * the following format: img[channel][height][width]
 *
 *    import { linspace, plot } from "propel";
 *    const filename = https://avatars2.githubusercontent.com/u/80?s=40&v=4;
 *    const tensor = await imread(filename, "RGB");
 *    tensor.shape();
 */
export async function imread(filename: string, mode: "RGB" | "RGBA" = "RGBA")
: Promise<Tensor> {
  if (IS_NODE) {
    return await NodeImageDecoder(filename, mode);
  }
  return await WebImageDecoder(filename, mode);
}
