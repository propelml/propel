// This implementation closely follows gradients_function from TF Eager:
// https://github.com/tensorflow/tensorflow/blob/16b0bb095296fcfa17182aeae656a35faf70f36e/tensorflow/python/eager/backprop.py#L442

import { Tensor, TensorLike } from "./tensor";
import { NDArray } from './deeplearnjs/src/math/ndarray';
import { log, GradientCollector, CounterMap } from './util';
import { NDArrayMath } from './deeplearnjs/src/math/math';
import { assertEqual } from './util';
import { getBackwardFuncs } from './ops';

// The global tape stack. The tape stack is used to support higher order
// gradients.
let tapeStack: Tape[] = [];

interface TapeEntry {
  name: string;
  oid: number;
  inputIds: number[];
  outputIds: number[];

  // TODO These should not be set for, ops like exp or neg. Keeping references
  // to all Tensors is very inefficient. 
  inputs: Tensor[];
  ans: Tensor;
};

// Hacky. How does one define methods on interfaces?
function tapeEntryToString(e: TapeEntry): string {
  let i = e.inputIds.toString();
  let o = e.outputIds.toString();
  return `${e.name}(${e.oid}) in ${i} out ${o}`;
}

// Represents a gradient propagation trace.
export class Tape {
  // Maps from Tensor ids to their operation ids.
  tensorToOp = new Map<number, number>();

  oidLookup = new Map<number, TapeEntry>(); // Maps from operation id to TapeEntry.

  // Returns true if any tensor should be recorded.
  shouldRecord(tids: number[]): boolean {
    for (let tid of tids) {
      if (this.tensorToOp.has(tid)) {
        return true;
      }
    }
    return false;
  }

  // Adds a tensor to the tape.
  watch(tensor: Tensor): void {
    let id = tensor.id;
    if (!this.tensorToOp.has(id)) {
      this.tensorToOp.set(id, -1);
    }
  }

  recordOp(tapeEntry: TapeEntry): void {
    if (!this.shouldRecord(tapeEntry.inputIds)) {
      return;
    }
    log("recordOp %s", tapeEntryToString(tapeEntry));
    for (let tid of tapeEntry.outputIds) {
      this.tensorToOp.set(tid, tapeEntry.oid);
    }
    this.oidLookup.set(tapeEntry.oid, tapeEntry);
  }
}

// Returns a function which differentiates f with respect to the given
// argnum indexes.
//
// Assumputions to be removed later:
// - User doesn't need the forward pass result of f. (Need gradAndVal func
//   which returns both.)
// - Tensors are scalars. This is only assumed in a few places and they're
//   marked with TODO(scalar). Need to propigate shape_and_dtype (See
//   backprop.py).
export function multigrad(f, argnums: number[]) {
  return function(...args: TensorLike[]): Tensor[] {
    pushNewTape();
    let targs: Tensor[] = [];
    // Convert args to Tensors.
    for (let i = 0; i < args.length; ++i) {
      targs.push(Tensor.convert(args[i]));
    }
    // Watch the specified argnums.
    for (let i of argnums) {
      watch(targs[i]);
    }
    let result = f.apply(null, targs); // Do the forward pass.
    result = Tensor.convert(result);
    return imperativeGrad(result, targs);
  };
}

// Returns the gradient with respect to a single input.
export function grad(f, argnum = 0) {
  //return multigrad(f, [argnum])[0];
  let g = multigrad(f, [argnum]);
  return function(...args: TensorLike[]): Tensor {
    return g(...args)[0];
  };
}

function imperativeGrad(target: Tensor, sources: Tensor[]): Tensor[] {
  let tape = popTape();
  let readyOps: number[] = [];
  let sourceIds = new Set(sources.map(t => t.id));

  // We discard tape.oidLookup and instead use the oidLookup returned from
  // prepareBackprop, this potentially allows the VM to release all memory used
  // to keep traces that are irrelevant to the gradient computation we're
  // doing.
  let [usageCounts, opMissingTensor, oidLookup] = prepareBackprop(target, tape,
    sourceIds);
  log("usageCounts", usageCounts);
  log("opMissingTensor", opMissingTensor);

  let targetOid = tape.tensorToOp.get(target.id);
  if (targetOid) {
    readyOps.push(targetOid);
  }

  let gradients = new GradientCollector();
  gradients.append(target.id, target.onesLike());

  // Execute backwards passes.
  while (readyOps.length > 0) {
    let oid = readyOps.pop();
    let op = oidLookup.get(oid);

    // TODO(scalar) Currently assuming ops have single output.
    assertEqual(op.outputIds.length, 1)
    let outGrad = gradients.aggregate(op.outputIds[0]);

    log("backprop %s", op);
    log("- outGrad %s", outGrad);

    let inGrads = getBackwardFuncs(op.name).map(bwFunc => {
      return bwFunc(outGrad, op.ans, ...op.inputs);
    });

    log("- inGrad %s", inGrads);

    for (let i = 0; i < op.inputIds.length; i++) {
      let t = op.inputIds[i];

      gradients.append(t, inGrads[i]);

      if (usageCounts.get(t) > 0) {
        usageCounts.dec(t);
        if (tape.tensorToOp.has(t) && usageCounts.get(t) == 0 &&
          !sourceIds.has(t)) {
          let inOp = tape.tensorToOp.get(t);
          if (inOp > 0) {
            if (opMissingTensor.get(inOp) > 0) {
              opMissingTensor.dec(inOp);
              if (opMissingTensor.get(inOp) == 0) {
                readyOps.push(inOp);
              }
            }
          }
        }
      }
    }
  }

  // Collect the gradients that we want.
  let result: Tensor[] = [];
  for (let t of sources) {
    let r = gradients.aggregate(t.id);
    log('- result %s', r);
    result.push(r);
  }

  return result;
}

// The purpose of this function is to pre-traverse the computation graph, without
// performing the backwards pass operations, in order to gather information
// about how many edges tensors have. This information is later used
// imperativeGrad() to know when a tensor that is being used as input to
// multiple operations has had all of its gradients calculated, and thus
// that the algorithm can move on to the tensor's origin.
function prepareBackprop(target, tape, sourceIds): [CounterMap, CounterMap, Map<number, TapeEntry>] {
  let tensorStack = [target.id];
  let oidLookup = new Map<number, TapeEntry>();
  let usageCounts = new CounterMap(); // tensor id -> count

  while (tensorStack.length > 0) {
    let t = tensorStack.pop();
    let oid = tape.tensorToOp.get(t);

    // oid is -1 if tensor is a source, continue
    // Or if we've already processed this op, continue.
    if (oid === undefined || oid < 0 || oidLookup.has(oid)) continue;

    let op = tape.oidLookup.get(oid);
    oidLookup.set(oid, op);

    // Iterate thru the op's input tensors.
    for (let inputId of op.inputIds) {
      usageCounts.inc(inputId);
      // Conditionally add inputId to the stack:
      // - if this is the first usage of this tensor, and
      // - if the input tensor has a registered op, and 
      // - it's not one of the source tensors, 
      if (usageCounts.get(inputId) == 1 && tape.tensorToOp.has(inputId) &&
        !sourceIds.has(inputId)) {
        tensorStack.push(inputId);
      }
    }
  }

  let opMissingTensor = new CounterMap(); // op id -> count
  for (let t of usageCounts.keys()) {
    if (tape.tensorToOp.has(t) && tape.tensorToOp.get(t) > 0) {
      let oid = tape.tensorToOp.get(t);
      opMissingTensor.inc(oid)
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
  for (let tape of tapeStack) {
    tape.watch(t);
  }
}

export function recordOp(tapeEntry: TapeEntry) {
  for (let tape of tapeStack) {
    tape.recordOp(tapeEntry);
  }
}
