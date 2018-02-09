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
import * as types from "./types";

declare class Context {
  constructor();
}

export type DTypeCode = number;
export type AttrType = number;

declare class Handle {
  constructor(ta: types.TypedArray, shape: types.Shape, dtype: DTypeCode);
}

// TODO this could be improved:
export type AttrDef = Array<string | number | boolean>;

interface DeviceDesc {
  name: string;
  deviceType: types.DeviceType;
  memoryBytes: number;
}

export interface BindingInterface {
  Handle: typeof Handle;
  Context: typeof Context;

  asArrayBuffer(h: Handle): ArrayBuffer;
  getDType(h: Handle): DTypeCode;
  getShape(h: Handle): types.Shape;
  getDevice(h: Handle): string;
  listDevices(ctx: Context): DeviceDesc[];
  createSmallHandle(ctx: Context, dtype: DTypeCode, device: string,
                    data: number | number[]): Handle;
  copyToDevice(ctx: Context, h: Handle, device: string): Handle;
  execute(ctx: Context, op: string, attrs: AttrDef[],
          inputs: Handle[]): Handle[];
  dispose(h: Handle): void;

  TF_FLOAT: DTypeCode;
  TF_DOUBLE: DTypeCode;
  TF_INT32: DTypeCode;
  TF_UINT8: DTypeCode;
  TF_INT16: DTypeCode;
  TF_INT8: DTypeCode;
  TF_STRING: DTypeCode;
  TF_COMPLEX64: DTypeCode;
  TF_COMPLEX: DTypeCode;
  TF_INT64: DTypeCode;
  TF_BOOL: DTypeCode;
  TF_QINT8: DTypeCode;
  TF_QUINT8: DTypeCode;
  TF_QINT32: DTypeCode;
  TF_BFLOAT16: DTypeCode;
  TF_QINT16: DTypeCode;
  TF_QUINT16: DTypeCode;
  TF_UINT16: DTypeCode;
  TF_COMPLEX128: DTypeCode;
  TF_HALF: DTypeCode;
  TF_RESOURCE: DTypeCode;
  TF_VARIANT: DTypeCode;
  TF_UINT32: DTypeCode;
  TF_UINT64: DTypeCode;

  ATTR_STRING: AttrType;
  ATTR_INT: AttrType;
  ATTR_FLOAT: AttrType;
  ATTR_BOOL: AttrType;
  ATTR_TYPE: AttrType;
  ATTR_SHAPE: AttrType;
  ATTR_FUNCTION: AttrType;
  ATTR_STRING_LIST: AttrType;
  ATTR_INT_LIST: AttrType;
  ATTR_FLOAT_LIST: AttrType;
  ATTR_BOOL_LIST: AttrType;
  ATTR_TYPE_LIST: AttrType;
  ATTR_SHAPE_LIST: AttrType;
}
