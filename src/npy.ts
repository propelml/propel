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

// This module saves and loads from the numpy format.
// Propel uses this for storing checkpoints.
// https://docs.scipy.org/doc/numpy/neps/npy-format.html

import { TextDecoder } from "text-encoding";
import { bo } from "./backend";
import { Tensor } from "./tensor";
import * as types from "./types";
import * as util from "./util";

/** Serializes a tensor into a npy file contents. */
export async function serialize(tensor: Tensor): Promise<ArrayBuffer> {
  const descr = {
    "float32": "<f4",
    "int32": "<i4",
  }[tensor.dtype];

  // First figure out how long the file is going to be so we can create the
  // output ArrayBuffer.
  const magicStr = "NUMPY";
  const versionStr = "\x01\x00";
  const [ d, fo, s ] = [ descr, "False", tensor.shape.join(",") + "," ];
  let header = `{'descr': '${d}', 'fortran_order': ${fo}, 'shape': (${s}), }`;
  const unpaddedLength = 1 + magicStr.length + versionStr.length +
                         2 + header.length;
  // Spaces to 16-bit align.
  const padding = " ".repeat((16 - unpaddedLength % 16) % 16);
  header += padding;
  util.assert((unpaddedLength + padding.length) % 16 === 0);
  // Either int32 or float32 for now Both 4 bytes per element.
  // TODO support uint8 and bool.
  const bytesPerElement = 4;
  const dataLen = bytesPerElement * numEls(tensor.shape);
  const totalSize = unpaddedLength + padding.length + dataLen;

  const ab = new ArrayBuffer(totalSize);
  const view = new DataView(ab);
  let pos = 0;

  // Write magic string and version.
  view.setUint8(pos++, 0x93);
  pos = writeStrToDataView(view, magicStr + versionStr, pos);

  // Write header length and header.
  view.setUint16(pos, header.length, true);
  pos += 2;
  pos = writeStrToDataView(view, header, pos);

  // Write data
  const data = await tensor.data();
  util.assert(data.length === numEls(tensor.shape));
  for (let i = 0; i < data.length; i++) {
    switch (tensor.dtype) {
      case "float32":
        view.setFloat32(pos, data[i], true);
        pos += 4;
        break;

      case "int32":
        view.setInt32(pos, data[i], true);
        pos += 4;
        break;
    }
  }
  return ab;
}

/** Parses an ArrayBuffer containing a npy file. Returns a tensor. */
export function parse(ab: ArrayBuffer): Tensor {
  util.assert(ab.byteLength > 5);
  const view = new DataView(ab);
  const decoder = new TextDecoder("ascii");
  let pos = 0;

  // First parse the magic string.
  const byte0 = view.getUint8(pos++);
  const magicStr = decoder.decode(new DataView(ab, pos, 5));
  pos += 5;
  if (byte0 !== 0x93 || magicStr !== "NUMPY") throw Error("Not a numpy file.");

  // Parse the version
  const version = [view.getUint8(pos++), view.getUint8(pos++)].join(".");
  if (version !== "1.0") throw Error("Unsupported version.");

  // Parse the header length.
  const headerLen = view.getUint16(pos, true);
  pos += 2;

  // Parse the header.
  // header is almost json, so we just manipulated it until it is.
  //  {'descr': '<f8', 'fortran_order': False, 'shape': (1, 2), }
  const headerPy = decoder.decode(new DataView(ab, pos, headerLen));
  pos += headerLen;
  const bytesLeft = view.byteLength - pos;
  const headerJson = headerPy.replace("True", "true")
                             .replace("False", "false")
                             .replace(/'/g, `"`)
                             .replace(/,\s*}/, " }")
                             .replace(/,?\)/, "]")
                             .replace("(", "[");
  const header = JSON.parse(headerJson);
  if (header.fortran_order) {
    throw Error("NPY parse error. Implement me.");
  }

  // Finally parse the actual data.
  const size = numEls(header.shape);
  if (header["descr"] === "<f8") {
    // 8 byte float. float64.
    util.assert(bytesLeft === size * 8);
    const s = ab.slice(pos, pos + size * 8);
    const ta = new Float32Array(new Float64Array(s));
    return fromTypedArrayAndShape(ta, header.shape);

  } else if (header["descr"] === "<f4") {
    // 4 byte float. float32.
    util.assert(bytesLeft === size * 4);
    const s = ab.slice(pos, pos + size * 4);
    const ta = new Float32Array(s);
    return fromTypedArrayAndShape(ta, header.shape);

  } else if (header["descr"] === "<i8") {
    // 8 byte int. int64.
    util.assert(bytesLeft === size * 8);
    const s = ab.slice(pos, pos + size * 8);
    const ta = new Int32Array(s).filter((val, i) => i % 2 === 0);
    return fromTypedArrayAndShape(ta, header.shape);

  } else if (header["descr"] === "|u1") {
    // uint8.
    util.assert(bytesLeft === size);
    const s = ab.slice(pos, pos + size);
    const ta = new Uint8Array(s);
    return fromTypedArrayAndShape(ta, header.shape);

  } else {
    throw Error(`Unknown dtype "${header["descr"]}". Implement me.`);
  }
}

/** Loads and parses a npy file. */
export async function load(filename: string): Promise<Tensor> {
  const ab = await util.fetchArrayBuffer(filename);
  return parse(ab);
}

// TODO move to backend.ts.
function fromTypedArrayAndShape(ta: types.TypedArray,
                                shape: types.Shape): Tensor {
  const storage = bo.fromTypedArray(ta, shape);
  return new Tensor(storage);
}

function numEls(shape: types.Shape): number {
  if (shape.length === 0) {
    return 1;
  } else {
    return shape.reduce((a, b) => a * b);
  }
}

function writeStrToDataView(view: DataView, str: string, pos: number) {
  for (let i = 0; i < str.length; i++) {
    view.setInt8(pos + i, str.charCodeAt(i));
  }
  return pos + str.length;
}
