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

import { toString as formatTensor } from "../src/format";
import { Tensor } from "../src/tensor";
import { isNumericalKey } from "../src/util";

export interface AtomDescriptor {
  type: "null" | "undefined" | "proto" | "getter" | "gettersetter" | "setter";
}
export interface PrimitiveDescriptor {
  type: "boolean" | "date" | "number" | "regexp" | "string" | "symbol";
  value: string;
}
export interface BaseObjectDescriptor {
  ctor: string | null;            // Name of the object's constructor.
  props: PropertyDescriptor[];    // Descriptors for object properties.
}
export interface ArrayDescriptor extends BaseObjectDescriptor {
  type: "array";
  length: number;
}
export interface BoxDescriptor extends BaseObjectDescriptor {
  type: "box";
  primitive: PrimitiveDescriptor; // Primitive value boxed in this object.
}
export interface FunctionDescriptor extends BaseObjectDescriptor {
  type: "function";
  name: string;
  async: boolean;
  class: boolean;
  generator: boolean;
}
export interface ObjectDescriptor extends BaseObjectDescriptor {
  type: "object";
}
export interface TensorDescriptor extends BaseObjectDescriptor{
  type: "tensor";
  dtype: string;
  shape: number[];
  formatted: string;
}

export interface PropertyDescriptor {
  key: DescriptorRef;
  value: DescriptorRef;
  hidden: boolean; // Property is private or non-enumerable.
}

export type ValueDescriptor =
  // TODO: add Maps and Sets.
  AtomDescriptor | PrimitiveDescriptor | ArrayDescriptor | BoxDescriptor |
  ObjectDescriptor | FunctionDescriptor | TensorDescriptor;
export type DescriptorRef = number;
export interface DescriptorSet { [id: number]: ValueDescriptor; }

export interface InspectorData {
  roots: DescriptorRef[];
  descriptors: DescriptorSet;
}

class Placeholder {
  static getter = new Placeholder("getter");
  static gettersetter = new Placeholder("getter");
  static proto = new Placeholder("proto");
  static setter = new Placeholder("setter");
  private constructor(readonly type: AtomDescriptor["type"]) {}
}

// tslint:disable:variable-name
const AsyncFunction = (async function() {}).constructor;
const GeneratorFunction = (function*() {}).constructor;
const TypedArray = Object.getPrototypeOf(Int32Array);
// tslint:enable:variable-name

// This function calls fn(), and returns true if the succeeds. If fn() throws
// an exception, it'll catch the exception and return false.
function succeeds(fn: () => void): boolean {
  try {
    fn();
    return true;
  } catch (e) {
    return false;
  }
}

// This function "describes" any javascript object for the purpose of
// inspection/pretty printing. The returned descriptor object can be
// serialized (e.g. using JSON) and sent to a different javascript context.
class DescriptionBuilder {
  private descriptorCount = 0;
  private descriptors = new Array<ValueDescriptor>();
  private valueMap = new Map<any, DescriptorRef>();

  getDescriptors(): ValueDescriptor[] {
    return this.descriptors;
  }

  describe(value: any): DescriptorRef {
    let id = this.valueMap.get(value);
    // If no descriptor for this value is found, create a new one.
    if (id === undefined) {
      // Assign an id.
      id = this.descriptorCount++;
      this.valueMap.set(value, id);
      // Create a new descriptor.
      this.descriptors[id] = this.createDescriptor(value);
    }
    return id;
  }

  private createDescriptor(value: any): ValueDescriptor {
    // Handle primitive value types.
    const type = typeof value;
    switch (type) {
      case "boolean":
      case "number":
      case "string":
      case "symbol":
        return { type, value: String(value) };
      case "undefined":
        return { type };
      case "object":
        if (value === null) {
          return { type: "null" };
        }
    }
    // Handle non-value placeholders (e.g. Placeholder.getter).
    if (value instanceof Placeholder) {
      return value;
    }

    // If we get here, the value must be non-primitive (an object or function).
    // The `d` variable will hold the value's description.
    let d: any;

    // Detect what kind of non-primitive we're dealing with.
    if (type === "function") {
      // Function or class.
      d = {
        type: "function",
        name: value.name,
        class: /^class\s/.test(value),
        async: value instanceof AsyncFunction,
        generator: value instanceof GeneratorFunction
      };
    } else if ((value instanceof Array || value instanceof TypedArray) &&
              succeeds(() => value.length)) {
      // Array-like object (with a magic length property).
      d = {
        type: "array",
        length: value.length
      };
    } else if (value instanceof Tensor) {
      // Tensor shapes [] and [1] are equivalent, but the latter looks better.
      const tensor = value.shape.length > 0 ? value : value.reshape([1]);
      d = {
        type: "tensor",
        dtype: tensor.dtype,
        shape: [...tensor.shape],
        formatted: formatTensor(tensor)
      };
    } else if ((value instanceof Boolean || value instanceof Number ||
              value instanceof String) && succeeds(() => value.valueOf())) {
      // This is a box object as they are created by e.g. `new Number(3)`.
      d = { type: "box", primitive: this.createDescriptor(value.valueOf()) };
    } else if (value instanceof Date && succeeds(() => value.toISOString())) {
      // For practical purposes we pretend that there exist "regexp" and "date"
      // primitives, and that Date/RegExp objects are boxed versions of them.
      const primitive = { type: "date", value: value.toISOString() };
      d = { type: "box", primitive };
    } else if (value instanceof RegExp && succeeds(() => String(value))) {
      d = { type: "box", primitive: { type: "regexp", value: String(value) }};
    } else {
      // Regular or other type of object.
      d = { type: "object" };
    }

    // Capture the name of the constructor.
    const proto = Object.getPrototypeOf(value);
    if (proto === null) {
      d.ctor = null;  // Object has no prototype.
    } else if (typeof proto.constructor === "function") {
      d.ctor = proto.constructor.name;
    } else {
      d.ctor = "[unknown]";
    }

    // Some helper variables that we need later to decide which keys to skip.
    const valueIsBoxedString = value instanceof String;
    const valueIsTensor = d.type === "tensor";

    // List named properties and symbols.
    d.props = [];
    const keys = [
      ...Object.getOwnPropertyNames(value),
      ...Object.getOwnPropertySymbols(value)
    ];
    for (const key of keys) {
      const descriptor = Object.getOwnPropertyDescriptor(value, key);
      // For strings, skip numeric keys that point at a character in the string.
      if (valueIsBoxedString && typeof key === "string" &&
          isNumericalKey(key) && Number(key) < value.length) {
        continue;
      }
      // Set the 'hidden' flag for non-enumerable properties, as well as a few
      // known-private fields.
      const hidden = !descriptor.enumerable ||
                    (valueIsTensor && key === "_id") ||
                    (valueIsTensor && key === "storage");
      // If the property has a getter/setter, list it as such. We make no
      // attempt to read the property value, since this may have side effects.
      if (descriptor.get || descriptor.set) {
        // `type` becomes either "getter", "setter", or "gettersetter".
        const type = (descriptor.get ? "getter" : "") +
                     (descriptor.set ? "setter" : "");
        d.props.push({
          key: this.describe(key),
          value: this.describe(Placeholder[type]),
          hidden
        });
        continue;
      }
      // Describe the property value.
      d.props.push({
        key: this.describe(key),
        value: this.describe(descriptor.value),
        hidden
      });
    }
    // Add the object's prototype to the properties list.
    // TODO: disabled for performance reasons.
    // if (proto) {
    //   d.props.push({
    //     key: this.describe(Placeholder.proto),
    //     value: this.describe(proto),
    //     hidden: true
    //   });
    // }

    return d;
  }
}

export function describe(values: any[]): InspectorData {
  const builder = new DescriptionBuilder();
  const roots = values.map(v => builder.describe(v));
  return {
    descriptors: builder.getDescriptors(),
    roots
  };
}
