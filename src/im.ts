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

import { concat, fill, Tensor, tensor } from "./api";
import { fetchArrayBuffer } from "./fetch";
import { convert } from "./tensor";
import { Mode } from "./types";
import { createResolvable, IS_NODE, nodeRequire } from "./util";

// TODO These modules are only used in Node and it is excessive to include them
// in the browser bundle. However at the moment we only distribute a single
// bundle. So, for now, just include them normally.
import * as JPEG from "jpeg-js";
import * as pngjs from "pngjs";

export interface Image {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

function toTensor(data: Uint8Array, height: number, width: number,
                  mode: Mode): Tensor {
  let image = convert(data).reshape([height, width, 4]);
  if (mode === "RGBA") {
    return image;
  }
  image = image.slice([0, 0, 0], [-1, -1, 3]);
  if (mode === "RGB") {
    return image;
  }
  if (mode === "L") {
    return image.reduceMean([2], true);
  }
  throw new Error("Unsupported convertion mode.");
}

export function toUint8Image(image: Tensor): Image {
  const dtype = image.dtype;
  const height = image.shape[0] as number;
  const width = image.shape[1] as number;
  let data;
  if (image.shape.length === 2) {
    // convert to a 3D tensor
    image = image.reshape([height, width, 1]);
  }
  if (image.shape.length === 3) {
    const channels = image.shape[2];
    if (channels > 4 || channels === 2) {
      throw new Error(`${image.shape} does not match any valid image mode.`);
    }
    if (channels === 1) {
      // Grayscale to RGB
      image = concat([image, image, image], 2);
    }
    if (channels <= 3) {
      // RGB to RGBA
      const alpha = fill(tensor(255, {dtype}), [height, width, 1]);
      image = concat([image, alpha], 2);
    }
    // Convert to 1D array
    data = image
      .reshape([height * width * 4])
      .dataSync();
    data = Uint8ClampedArray.from(data);
    return { width, height, data };
  }
  throw new Error(`Unsupported image rank.`);
}

async function webImageDecoder(filename: string, mode: Mode)
    : Promise<Tensor> {
  const buffer = await fetchArrayBuffer(filename);
  const blob = new Blob( [ buffer ], { type: "image/jpeg" } );
  const dataURI = window.URL.createObjectURL(blob);

  const img = new Image();
  const onLoad = createResolvable();
  img.onload = onLoad.resolve;
  img.src = dataURI;
  await onLoad;

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
  window.URL.revokeObjectURL(dataURI);
  return image;
}

export function createCanvas(image: Image) {
  const { height, width, data } = image;
  const canvas = document.createElement("canvas");
  canvas.height = height;
  canvas.width = width;
  const ctx = canvas.getContext("2d");
  const imageData = new ImageData(data, width, height);
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

const imageSignatures: Array<[number[], string]> = [
  [[0xFF, 0xD8], "image/jpeg"],
  [[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A], "image/png"]
];

function readMIME(data: Uint8Array): string {
  // find matched MIME type
  for (let i = 0; i < imageSignatures.length; ++i) {
    const [sig, type] = imageSignatures[i];
    let ret = true;
    for (let j = 0; j < sig.length; ++j) {
      if (data[j] !== sig[j]) {
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

async function pngReadHandler(data: ArrayBuffer, mode: Mode): Promise<Tensor> {
  const promise = createResolvable<Tensor>();
  new pngjs.PNG().parse(data).on("parsed", function(this: any) {
    const data = new Uint8Array(this.data);
    promise.resolve(toTensor(data, this.height, this.width, mode));
  });
  return await promise;
}

function pngSaveHandler(filename: string, image): Promise<void> {
  const fs = nodeRequire("fs");
  const file = new pngjs.PNG({
    height: image.height,
    width: image.width
  });
  file.data = image.data;
  return new Promise(resolve => {
    file.pack()
      .pipe(fs.createWriteStream(filename))
      .on("finish", resolve);
  });
}

function jpegReadHandler(data: ArrayBuffer, mode: Mode): Tensor {
  const img = JPEG.decode(data, true);
  return toTensor(img.data, img.height, img.width, mode);
}

function jpegSaveHandler(filename: string, image): void {
  const fs = nodeRequire("fs");
  const {data} = JPEG.encode(image);
  fs.writeFileSync(filename, data);
  return;
}

async function nodeImageDecoder(filename: string, mode: Mode)
    : Promise<Tensor> {
  const ab = await fetchArrayBuffer(filename);
  const ui8 = new Uint8Array(ab);
  const MIME = readMIME(ui8);
  switch (MIME) {
    case "image/png":
      return await pngReadHandler(ab, mode);
    case "image/jpeg":
      return jpegReadHandler(ab, mode);
    default:
      throw new Error("This file is not a valid PNG/JPEG image.");
  }
}

/** Read an image from given file to a 3D tensor in the following
 * The tensor will have the shape [height, width, channel].
 * The second argument specifies if you want "RGBA", "RGB", or "L" (for
 * luminosity/greyscale).
 *
 *    import { imread, imshow } from "propel"
 *    img = await imread("/src/testdata/sample.png")
 *    imshow(img.transpose([1, 0, 2]))
 */
export async function imread(filename: string, mode: Mode = "RGBA")
    : Promise<Tensor> {
  if (IS_NODE) {
    return await nodeImageDecoder(filename, mode);
  }
  // Neither image decoder works in JSDOM while building the website.
  // TODO: fix this - https://github.com/propelml/propel/issues/309
  if (/jsdom/.test(window.navigator.userAgent)) {
    return null;
  }
  return await webImageDecoder(filename, mode);
}

/** Save a 3D tensor to disk as an image
 */
export async function imsave(tensor: Tensor,
                             filename: string,
                             handler?: "PNG" | "JPEG"): Promise<void> {
  if (IS_NODE) {
    const image = toUint8Image(tensor);
    if (!handler) {
      const path = nodeRequire("path");
      const ext = path.extname(filename)
        .toLowerCase()
        .replace(".jpg", ".jpeg");
      if (ext === ".png" || ext === ".jpeg") {
        handler = ext.substr(1).toUpperCase();
      }
    }
    handler = handler || "PNG";
    switch (handler) {
      case "PNG":
        return await pngSaveHandler(filename, image);
      case "JPEG":
        return jpegSaveHandler(filename, image);
    }
    throw new Error(`Unsupported image format "${handler}"`);
  }
  const image = toUint8Image(tensor);
  const canvas = createCanvas(image);
  const base64 = canvas.toDataURL("image/png");
  const link = document.createElement("a");
  link.setAttribute("href", base64);
  link.setAttribute("download", filename);
  link.click();
}
