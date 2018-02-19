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
import { isUndefined } from "util";
import * as mnist from "./mnist";
import { Tensor } from "./tensor";

type NamedTensors = { [name: string]: Tensor };

export function datasetFromSlices(tensors: NamedTensors): Dataset {
  return new SliceDataset(tensors);
}

export function dataset(name: string, split = "train"): Dataset {
  if (name !== "mnist") throw Error("Bad dataset " + name);

  return new SliceDataset(mnist.loadSplit(split));
}

abstract class Dataset {
  constructor(protected parent?: Dataset) { }

  abstract async next(): Promise<NamedTensors>;

  // Most of the datasets just pass their reset calls upward.
  reset(): void {
    if (this.parent) this.parent.reset();
  }

  batch(batchSize: number) {
    return new BatchDataset(this, batchSize);
  }

  repeat(count?: number) {
    return new RepeatDataset(this, count);
  }

  shuffle(bufSize?: number) {
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
  }

  async next(): Promise<NamedTensors> {
    return this.nextBatch(1);
  }

  async nextBatch(batchSize: number): Promise<NamedTensors> {
    if (this.promise) await this.promise;
    const out: NamedTensors = {};
    // End condition.
    if (this.pos >= this.batchDim) return null;
    const sliceSize = Math.min(batchSize, this.batchDim - this.pos);
    for (const name of Object.keys(this.tensors)) {
      const t = this.tensors[name];
      // TODO This is ugly. slice() should take truncated begin and size
      // arguments.
      const begin = [this.pos, ...Array(t.rank - 1).fill(0)];
      const size = [sliceSize, ...Array(t.rank - 1).fill(-1)];
      out[name] = t.slice(begin, size);
    }
    this.pos += batchSize;
    return out;
  }
}

// TODO
function stack(tensors: Tensor[], axis = 0): Tensor {
  throw Error("not implemented");
}

class BatchDataset extends Dataset {
  constructor(parent: Dataset, readonly batchSize: number) {
    super(parent);
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
        batchComponents.push(tensors);
      }
      const out: NamedTensors = {};
      for (const name of Object.keys(batchComponents[0])) {
        const batch = batchComponents.map(tensors => tensors[name]);
        out[name] = stack(batch, 0);
      }
      return out;
    }
  }
}

class RepeatDataset extends Dataset {
  epoch = 0;

  constructor(parent: Dataset, readonly count?: number) {
    super(parent);
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
        this.epoch++;
        if (this.count && this.epoch >= this.count) {
          return null;
        }
        this.parent.reset();
      }
    }
  }
}

class ShuffleDataset extends Dataset {
  bufferSize: number;

  constructor(parent: Dataset, readonly bufSize?: number) {
    super(parent);
  }

  async next(): Promise<NamedTensors> {
    throw Error("not implemented.");
  }
}
