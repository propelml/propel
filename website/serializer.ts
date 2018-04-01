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

import { Tensor } from "../src/api";
import * as types from "../src/types";

// This module generates a non-circular object from any of js data types
// so they can be sent or received thru RPC.
// TODO optimize generated data size.

export interface NodeTypeMap {
  number: number;
  string: string;
  symbol: string;
  function: {
    isNative: boolean;
    name: string;
  };
  undefined: undefined;
  null: undefined;
  object: undefined;
  date: string;
  regexp: string;
  boolean: 1 | 0;
  promise: undefined;
  array: undefined;
  tensor: {
    shape: number[];
    dtype: types.DType;
    data: string;
  };
}

export type NodeTypes = keyof NodeTypeMap;

export interface GNode<T extends NodeTypes>{
  type: T;
  data?: NodeTypeMap[T];
  cons?: string;
}

function isNative(fn) {
  return (/\{\s*\[native code\]\s*\}/).test("" + fn);
}

function Nodify(data): GNode<keyof NodeTypeMap> {
  switch (typeof data){
    case "number":
    case "string":
    case "boolean":
      return {
        type: typeof data,
        data
      };
    case "undefined":
      return {
        type: "undefined"
      };
    case "function":
      return {
        type: "function",
        data: {
          isNative: isNative(data),
          name: data.name
        }
      };
    case "object":
      if (data === null) {
        return {
          type: "null"
        };
      }
      if (data instanceof Date) {
        return {
          type: "date",
          data: data.toString()
        };
      }
      if (data instanceof RegExp) {
        return {
          type: "regexp",
          data: data.toString()
        };
      }
      if (data instanceof Promise) {
        return {
          type: "promise"
        };
      }
      if (data instanceof Array) {
        return {
          type: "array"
        };
      }
      if (data instanceof Tensor) {
        return {
          type: "tensor",
          data: {
            data: data.toString(),
            dtype: data.dtype,
            shape: data.shape
          }
        };
      }
      return {
        type: "object",
        cons: data.constructor ? data.constructor.name : "Object"
      };
    case "symbol":
      return {
        type: "symbol",
        data: data.toString()
      };
  }
  return undefined;
}

// object, property name, value
export type Edge = [number, string, number];

export interface SerializedObject {
  data: { [key: number] : GNode<any> };
  edges: Edge[];
}

export type ExtendableData = { [key: string]: UnserializedObject };
export interface ExtendableUnserializedObject {
  data: ExtendableData;
  type: "object" | "array" | "tensor";
  cons?: string;
  isCircular: boolean;
}
export interface UnserializedNode extends GNode<any> {
  isCircular: boolean;
}
export type UnserializedObject = ExtendableUnserializedObject |
                                 UnserializedNode;

// This class provides a graph data structure.
// Which means it split objects into data and edges.
// Data is an array (actually an object) of nodes. (look at GNode data-type)
// We keep all references to children in one array called `edges`.
class Graph {
  id = 0;
  map: WeakMap<object, number>;

  constructor(private data = {}, private edges: Edge[] = []) {
    this.map = new WeakMap<object, number>();
  }

  stringify(data, parent? : number, name?: string) {
    const node = Nodify(data);
    let id;
    if (node.type === "object" || node.type === "array") {
      if (this.map.has(data)) {
        return this.map.get(data);
      }
      id = this.push(node, parent, name);
      this.map.set(data, id);
      // insert childs
      let keys: Array<string | symbol> = Object.getOwnPropertyNames(data);
      keys = keys.concat(Object.getOwnPropertySymbols(data));
      for (const propertyName of keys) {
        const propertyStr = String(propertyName);
        const child = data[propertyName];
        if (typeof child === "object" && child !== null) {
          const childId = this.stringify(data[propertyName], id, propertyStr);
          this.edges.push([id, propertyStr, childId]);
          continue;
        }
        this.push(Nodify(child), id, propertyStr);
      }
    } else {
      id = this.push(node, parent, name);
    }
    return id;
  }

  push(node: GNode<any>, parent?: number, name?: string): number {
    const nodeId = this.id++;
    if (parent !== undefined) {
      this.edges.push([parent, name.toString(), nodeId]);
    }
    this.data[nodeId] = node;
    return nodeId;
  }

  serialized(): SerializedObject {
    return {data: this.data, edges: this.edges};
  }

  object(): UnserializedObject {
    const space = {};
    for (const key in this.data) {
      if (this.data.hasOwnProperty(key)) {
        const node = this.data[key];
        space[key] = {type: node.type};
        if (node.data !== undefined) {
          space[key].data = node.data;
        } else if (node.type === "object" || node.type === "array") {
          space[key].data = {};
          if (node.type === "object") {
            space[key].cons = node.cons;
          }
        }
      }
    }
    for (let i = 0; i < this.edges.length; ++i) {
      const [from, name, to] = this.edges[i];
      space[from].data[name] = {...space[to], isCircular: from >= to};
    }
    return space[0];
  }
}

export function serialize(data) {
  const graph = new Graph();
  graph.stringify(data);
  return graph.serialized();
}

export function unserialize(data: SerializedObject) {
  const graph = new Graph(data.data, data.edges);
  return graph.object();
}
