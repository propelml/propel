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

// This module is inspired by TensorFlow's tf.data.Dataset.
// https://www.tensorflow.org/api_docs/python/tf/data/Dataset
import { TextDecoder } from "text-encoding";
import { isUndefined } from "util";
import { stack, tensor, Tensor } from "./api";
import * as cache from "./cache";
import * as mnist from "./mnist";
import { NamedTensors } from "./tensor";
import { assert, delay } from "./util";

export function datasetFromSlices(tensors: NamedTensors): Dataset {
  return new SliceDataset(tensors);
}

const datasets = {
  "breast_cancer": () => new SliceDataset(loadBreastCancer()),
  "iris": () => new SliceDataset(loadIris()),
  "mnist/test": () => new SliceDataset(mnist.loadSplit("test")),
  "mnist/train": () => new SliceDataset(mnist.loadSplit("train")),
  "wine": () => new SliceDataset(loadWine()),
};

/** Loads a dataset. The available datasets are:
 * breast_cancer, iris, mnist/test, mnist/train, wine.
 * Example:
 *
 *    import * as pr from "propel";
 *    let ds = pr.dataset("iris").batch(10);
 *    let { features } = await ds.next();
 *    features;
 */
export function dataset(name: string): Dataset {
  const loader = datasets[name];
  if (!loader) {
    throw Error("Bad dataset " + name);
  }
  return loader();
}

abstract class Dataset {
  constructor(protected parent?: Dataset) { }

  abstract async next(): Promise<NamedTensors>;

  // TODO really this should be asyncIterator, but it's not well supported yet.
  [Symbol.iterator]() {
    return {
      next: () => {
        if (this.done) {
          return { value: null, done: true };
        } else {
          return { value: this.next(), done: false };
        }
      }
    };
  }

  // Most of the datasets just pass their reset calls upward.
  reset(): void {
    if (this.parent) this.parent.reset();
  }

  abstract get done(): boolean;

  batch(batchSize: number): Dataset {
    return new BatchDataset(this, batchSize);
  }

  repeat(count?: number): Dataset {
    return new RepeatDataset(this, count);
  }

  shuffle(bufSize?: number): Dataset {
    return new ShuffleDataset(this, bufSize);
  }
}

class SliceDataset extends Dataset {
  pos = 0;
  batchDim?: number;
  promise?: Promise<NamedTensors>;
  tensors?: NamedTensors;

  constructor(tensorsOrPromise?: NamedTensors | Promise<NamedTensors>) {
    super(null);
    if ((tensorsOrPromise as any).then) {
      this.promise = tensorsOrPromise as Promise<NamedTensors>;
      this.promise.then((tensors) => {
        this.setup(tensors);
        this.promise = null;
      });
    } else {
      this.setup(tensorsOrPromise as NamedTensors);
    }
  }

  private setup(tensors: NamedTensors) {
    this.tensors = tensors;
    // Check that the tensors all have the same batch dim.
    for (const name of Object.keys(this.tensors)) {
      const b = this.tensors[name].shape[0];
      if (isUndefined(this.batchDim)) this.batchDim = b;
      if (b !== this.batchDim) throw Error("Incompatible tensor shape.");
    }
  }

  reset() {
    this.pos = 0;
    this._done = false;
  }

  async next(): Promise<NamedTensors> {
    return this.nextBatch(1);
  }

  async nextBatch(batchSize: number): Promise<NamedTensors> {
    if (this.promise) await this.promise;

    // Make this truly async to avoid busy loops.
    await delay(0);

    const out: NamedTensors = {};
    // End condition.
    if (this.pos >= this.batchDim) {
      return null;
    }
    const sliceSize = Math.min(batchSize, this.batchDim - this.pos);
    for (const name of Object.keys(this.tensors)) {
      out[name] = this.tensors[name].slice(this.pos, sliceSize);
    }
    this.pos += sliceSize;

    if (this.pos >= this.batchDim) {
      this._done = true;
    }

    return out;
  }

  private _done = false;
  get done() {
    return this._done;
  }
}

class BatchDataset extends Dataset {
  constructor(parent: Dataset, readonly batchSize: number) {
    super(parent);
  }

  get done(): boolean {
    return this.parent.done;
  }

  async next(): Promise<NamedTensors> {
    // If parent is SliceDataset, then do an optimization where
    // we just slice off batches, to avoid creating many small
    // tensors for each row.
    if (this.parent instanceof SliceDataset) {
      return this.parent.nextBatch(this.batchSize);

    } else {
      // normal
      const batchComponents = [];
      for (let i = 0; i < this.batchSize; i++) {
        const tensors = await this.parent.next();
        if (tensors === null) {
          break;
        }
        batchComponents.push(tensors);
      }
      if (batchComponents.length === 0) {
        return null;
      }

      const out: NamedTensors = {};
      for (const name of Object.keys(batchComponents[0])) {
        const batch = batchComponents.map(tensors => tensors[name]);
        const first = batch.shift();
        out[name] = stack([first, ...batch], 0);
      }
      return out;
    }
  }
}

class RepeatDataset extends Dataset {
  epoch = 0;

  constructor(parent: Dataset, readonly count?: number) {
    super(parent);
    if (count != null && count <= 0) {
      throw Error("Bad value for repeat count.");
    }
  }

  reset() {
    this.epoch = 0;
    this.parent.reset();
  }

  async next(): Promise<NamedTensors> {
    while (true) {
      const r = await this.parent.next();
      if (r != null) {
        return r;
      } else {
        if (this.done) {
          return null;
        } else {
          this.epoch++;
          this.parent.reset();
        }
      }
    }
  }

  get done(): boolean {
    if (this.count == null || this.epoch < this.count - 1) {
      return false;
    } else {
      return this.parent.done;
    }
  }
}

class ShuffleDataset extends Dataset {
  private buffer: NamedTensors[] = [];

  constructor(parent: Dataset, readonly bufSize: number = 1) {
    super(parent);
  }

  get done(): boolean {
    return this.buffer.length === 0 && this.parent.done;
  }

  async next(): Promise<NamedTensors> {
    // Re-populate the buffer
    while (!this.parent.done && this.buffer.length < this.bufSize) {
      const tensors = await this.parent.next();
      if (tensors !== null) {
        this.buffer.push(tensors);
      }
    }

    // Still nothing to read
    if (this.buffer.length === 0) {
      return null;
    }

    const index = Math.floor(Math.random() * this.buffer.length);
    const out = this.buffer[index];
    this.buffer.splice(index, 1);
    return out;
  }
}

async function loadData(fn: string):
    Promise<{ features: Tensor, labels: Tensor }> {
  const ab = await cache.fetchWithCache(fn);
  const dec = new TextDecoder("ascii");
  const csv = dec.decode(new Uint8Array(ab));

  const lines = csv.trim().split("\n").map(line => line.split(","));
  const header = lines.shift();
  const nSamples = Number(header.shift());
  const nFeatures = Number(header.shift());
  // let labelNames = header;
  assert(lines.length === nSamples);
  const features: number[][] = [];
  const labels: number[] = [];
  for (const line of lines) {
    const row = line.map(Number);
    features.push(row.slice(0, nFeatures));
    labels.push(row[row.length - 1]);
  }

  const tensors = {
    features: tensor(features, { dtype: "float32" }),
    labels: tensor(labels, { dtype: "int32" }),
  };
  return tensors;
}

/**
 * Features are
 * 0 sepal length (cm)
 * 1 sepal width (cm)
 * 2 petal length (cm)
 * 3 petal width (cm)
 */
export async function loadIris():
    Promise<{ features: Tensor, labels: Tensor }> {
  return loadData("http://propelml.org/data/iris.csv");
}

export async function loadBreastCancer():
    Promise<{ features: Tensor, labels: Tensor }> {
  return loadData("http://propelml.org/data/breast_cancer.csv");
}

export async function loadWine():
    Promise<{ features: Tensor, labels: Tensor }> {
  return loadData("http://propelml.org/data/wine_data.csv");
}

// TODO
// boston_house_prices.csv
// diabetes_data.csv.gz diabetes_target.csv.gz
// digits.csv.gz
// iris.csv
// linnerud_exercise.csv linnerud_physiological.csv
