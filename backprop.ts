// This implementation closely follows gradients_function from TF Eager:
// tslint:disable-next-line:max-line-length
// https://github.com/tensorflow/tensorflow/blob/16b0bb095296fcfa17182aeae656a35faf70f36e/tensorflow/python/eager/backprop.py#L442

import { fill } from "./api";
import { BWFunc, getBackwardFuncs } from "./ops";
import { convert, Tensor } from "./tensor";
import * as types from "./types";
import { assert, assertEqual, CounterMap, log } from "./util";

// The global tape stack. The tape stack is used to support higher order
// gradients.
const tapeStack: Tape[] = [];

// Hacky. How does one define methods on interfaces?
function tapeEntryToString(e: types.TapeEntry): string {
  const i = e.inputIds.toString();
  const o = e.outputIds.toString();
  return `${e.name}(${e.oid}) in ${i} out ${o}`;
}

// Represents a gradient propagation trace.
export class Tape {
  // Maps from Tensor ids to their operation ids.
  tensorToOp = new Map<number, number>();

  // Maps from operation id to TapeEntry.
  oidLookup = new Map<number, types.TapeEntry>();

  // Returns true if any tensor should be recorded.
  shouldRecord(tids: number[]): boolean {
    for (const tid of tids) {
      if (this.tensorToOp.has(tid)) {
        return true;
      }
    }
    return false;
  }

  // Adds a tensor to the tape.
  watch(tensor: Tensor): void {
    const id = tensor.id;
    if (!this.tensorToOp.has(id)) {
      this.tensorToOp.set(id, -1);
    }
    log("- watch tensor id", id);
  }

  recordOp(tapeEntry: types.TapeEntry): void {
    if (!this.shouldRecord(tapeEntry.inputIds)) {
      return;
    }
    log("recordOp %s", tapeEntryToString(tapeEntry));
    for (const tid of tapeEntry.outputIds) {
      this.tensorToOp.set(tid, tapeEntry.oid);
    }
    this.oidLookup.set(tapeEntry.oid, tapeEntry);
  }
}

// Returns a function which differentiates f with respect to the given
// argnum indexes.
export function multigrad(f, argnums: number[]) {
  const g = multigradAndVal(f, argnums);
  return function(...args: types.TensorLike[]): Tensor[] {
    // Ignore the forward pass result.
    return g(...args)[0];
  };
}

export function multigradAndVal(f, argnums: number[]) {
  return function(...args: types.TensorLike[]):
      [Tensor[], Tensor] {
    pushNewTape();
    const targs: Tensor[] = args.map((tl) => convert(tl));
    // Watch the specified argnums.
    for (const i of argnums) {
      watch(targs[i]);
    }
    let result = f.apply(null, targs); // Do the forward pass.
    result = convert(result);
    return [imperativeGrad(result, targs), result];
  };
}

// Returns the gradient with respect to a single input.
export function grad(f, argnum = 0) {
  const g = multigradAndVal(f, [argnum]);
  return function(...args: types.TensorLike[]): Tensor {
    return g(...args)[0][0];
  };
}

export function gradAndVal(f, argnum = 0) {
  const g = multigradAndVal(f, [argnum]);
  return function(...args: types.TensorLike[]):
      [Tensor, Tensor] {
    const [grad, val] = g(...args);
    return [grad[0], val];
  };
}

function imperativeGrad(target: Tensor,
                        sources: Tensor[]): Tensor[] {
  const tape = popTape();
  const readyOps: number[] = [];
  const sourceIds = new Set(sources.map((t) => t.id));

  // We discard tape.oidLookup and instead use the oidLookup returned from
  // prepareBackprop, this potentially allows the VM to release all memory used
  // to keep traces that are irrelevant to the gradient computation we're
  // doing.
  const [usageCounts, opMissingTensor, oidLookup] = prepareBackprop(target,
    tape, sourceIds);
  log("usageCounts", usageCounts);
  log("opMissingTensor", opMissingTensor);

  const targetOid = tape.tensorToOp.get(target.id);
  if (targetOid) {
    readyOps.push(targetOid);
  }

  const gradients = new GradientCollector();
  gradients.append(target.id, target.onesLike());

  // Execute backwards passes.
  while (readyOps.length > 0) {
    const oid = readyOps.pop();
    const op = oidLookup.get(oid);

    // TODO(scalar) Currently assuming ops have single output.
    assertEqual(op.outputIds.length, 1);
    const outGrad = gradients.aggregate(op.outputIds[0]);

    log("backprop", tapeEntryToString(op));
    log("- outGrad %s", outGrad.shape, outGrad.getData());

    const inGrads = getBackwardFuncs(op.name).map(
      (bwFunc: null | BWFunc, i): Tensor => {
        if (bwFunc) {
          return bwFunc(outGrad, ...op.savedForBackward);
        } else {
          // Null backwards function, return a zero tensor of the same shape and
          // dtype as the input.
          const shapeDType = op.inputShapeDTypes[i];
          const zero = convert(0, shapeDType[1]);
          return fill(zero, shapeDType[0]);
        }
      });

    log("- inGrad shapes", inGrads.map((g) => g ? g.shape : null));

    for (let i = 0; i < op.inputIds.length; i++) {
      const tid: null | number = op.inputIds[i];
      if (tid === null || inGrads[i] == null) {
        log("- null inGrad[%d] ", i, inGrads[i], "tid", tid);
        continue;
      }

      assert(inGrads[i] != null, `inGrads[${i}] = null but tid = ${tid}`);
      gradients.append(tid, inGrads[i]);

      if (usageCounts.get(tid) > 0) {
        usageCounts.dec(tid);
        if (tape.tensorToOp.has(tid) &&
            usageCounts.get(tid) === 0 &&
            !sourceIds.has(tid)) {
          const inOp = tape.tensorToOp.get(tid);
          if (inOp > 0) {
            if (opMissingTensor.get(inOp) > 0) {
              opMissingTensor.dec(inOp);
              if (opMissingTensor.get(inOp) === 0) {
                readyOps.push(inOp);
              }
            }
          }
        }
      }
    }
  }

  // Collect the gradients that we want.
  const result: Tensor[] = [];
  for (const t of sources) {
    const r = gradients.aggregate(t.id);
    log("- result", t.id, r.shape);
    result.push(r);
  }

  return result;
}

// The purpose of this function is to pre-traverse the computation graph,
// without performing the backwards pass operations, in order to gather
// information about how many edges tensors have. This information is later
// used imperativeGrad() to know when a tensor that is being used as input to
// multiple operations has had all of its gradients calculated, and thus that
// the algorithm can move on to the tensor's origin.
function prepareBackprop(target, tape, sourceIds): [CounterMap, CounterMap,
  Map<number, types.TapeEntry>] {
  const tensorStack = [target.id];
  const oidLookup = new Map<number, types.TapeEntry>();
  const usageCounts = new CounterMap(); // tensor id -> count

  while (tensorStack.length > 0) {
    const t = tensorStack.pop();
    const oid = tape.tensorToOp.get(t);

    // oid is -1 if tensor is a source, continue
    // Or if we've already processed this op, continue.
    if (oid === undefined || oid < 0 || oidLookup.has(oid)) { continue; }

    const op = tape.oidLookup.get(oid);
    oidLookup.set(oid, op);

    // Iterate thru the op's input tensors.
    for (const inputId of op.inputIds) {
      usageCounts.inc(inputId);
      // Conditionally add inputId to the stack:
      // - if this is the first usage of this tensor, and
      // - if the input tensor has a registered op, and
      // - it's not one of the source tensors,
      if (usageCounts.get(inputId) === 1 && tape.tensorToOp.has(inputId) &&
        !sourceIds.has(inputId)) {
        tensorStack.push(inputId);
      }
    }
  }

  const opMissingTensor = new CounterMap(); // op id -> count
  for (const t of usageCounts.keys()) {
    if (tape.tensorToOp.has(t) && tape.tensorToOp.get(t) > 0) {
      const oid = tape.tensorToOp.get(t);
      opMissingTensor.inc(oid);
    }
  }

  return [usageCounts, opMissingTensor, oidLookup];
}

// Pushes a new tape onto the tape stack.
export function pushNewTape(): void {
  tapeStack.push(new Tape());
}

export function popTape(): Tape {
  return tapeStack.pop();
}

// Marks this tensor to be watched by all tapes in the stack.
function watch(t: Tensor) {
  for (const tape of tapeStack) {
    tape.watch(t);
  }
}

export function recordOp(tapeEntry: types.TapeEntry) {
  for (const tape of tapeStack) {
    tape.recordOp(tapeEntry);
  }
}

export class GradientCollector {
  // Maps tensor id -> gradient tensor array
  private map = new Map<number, Tensor[]>();

  append(tid: number, grad: Tensor): void {
    log("- GradientCollector append", tid, grad.shape);
    if (this.map.has(tid)) {
      this.map.get(tid).push(grad);
    } else {
      this.map.set(tid, [grad]);
    }
  }

  // Sum up the gradients for a given tensor id.
  aggregate(tid: number): Tensor {
    if (!this.map.has(tid) || this.map.get(tid).length === 0) {
      // TODO(scalar) Handle non-scalar shapes.
      return convert(0);
    }
    const grads = this.map.get(tid);
    // log('aggregate tid %d ngrads %d', tid, grads.length);
    let sum = grads[0];
    for (let i = 1; i < grads.length; i++) {
      sum = sum.add(grads[i]);
    }
    return sum;
  }
}
