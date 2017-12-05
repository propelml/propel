// This implementation closely follows gradients_function from TF Eager:
// tslint:disable-next-line:max-line-length
// https://github.com/tensorflow/tensorflow/blob/16b0bb095296fcfa17182aeae656a35faf70f36e/tensorflow/python/eager/backprop.py#L442

import { TensorLike } from "./types";
import { ChainableTensor, convertChainable } from "./chainable_tensor";
import { getBackwardFuncs } from "./ops";
import { CounterMap, log, assertEqual } from "./util";

// The global tape stack. The tape stack is used to support higher order
// gradients.
const tapeStack: Tape[] = [];

interface TapeEntry {
  name: string;
  oid: number;
  inputIds: number[];
  outputIds: number[];

  // TODO These should not be set for, ops like exp or neg. Keeping references
  // to all Tensors is very inefficient.
  inputs: TensorLike[];
  ans: ChainableTensor;
}

// Hacky. How does one define methods on interfaces?
function tapeEntryToString(e: TapeEntry): string {
  const i = e.inputIds.toString();
  const o = e.outputIds.toString();
  return `${e.name}(${e.oid}) in ${i} out ${o}`;
}

// Represents a gradient propagation trace.
export class Tape {
  // Maps from ChainableTensor ids to their operation ids.
  tensorToOp = new Map<number, number>();

  // Maps from operation id to TapeEntry.
  oidLookup = new Map<number, TapeEntry>();

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
  watch(tensor: ChainableTensor): void {
    const id = tensor.id;
    if (!this.tensorToOp.has(id)) {
      this.tensorToOp.set(id, -1);
    }
  }

  recordOp(tapeEntry: TapeEntry): void {
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
//
// Assumputions to be removed later:
// - User doesn't need the forward pass result of f. (Need gradAndVal func
//   which returns both.)
// - Tensors are scalars. This is only assumed in a few places and they're
//   marked with TODO(scalar). Need to propigate shape_and_dtype (See
//   backprop.py).
export function multigrad(f, argnums: number[]) {
  return function(...args: TensorLike[]): ChainableTensor[] {
    pushNewTape();
    const targs: ChainableTensor[] = [];
    // Convert args to Tensors.
    for (let i = 0; i < args.length; ++i) {
      targs.push(convertChainable(args[i]));
    }
    // Watch the specified argnums.
    for (const i of argnums) {
      watch(targs[i]);
    }
    let result = f.apply(null, targs); // Do the forward pass.
    result = convertChainable(result);
    return imperativeGrad(result, targs);
  };
}

// Returns the gradient with respect to a single input.
export function grad(f, argnum = 0) {
  //return multigrad(f, [argnum])[0];
  const g = multigrad(f, [argnum]);
  return function(...args: TensorLike[]): ChainableTensor {
    return g(...args)[0];
  };
}

function imperativeGrad(target: ChainableTensor, sources: ChainableTensor[]):
  ChainableTensor[] {
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

    log("backprop %s", op);
    log("- outGrad %s", outGrad);

    const inGrads = getBackwardFuncs(op.name).map((bwFunc) => {
      return bwFunc(outGrad, op.ans, ...op.inputs);
    });

    log("- inGrad %s", inGrads);

    for (let i = 0; i < op.inputIds.length; i++) {
      const t = op.inputIds[i];

      gradients.append(t, inGrads[i]);

      if (usageCounts.get(t) > 0) {
        usageCounts.dec(t);
        if (tape.tensorToOp.has(t) && usageCounts.get(t) == 0 &&
          !sourceIds.has(t)) {
          const inOp = tape.tensorToOp.get(t);
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
  const result: ChainableTensor[] = [];
  for (const t of sources) {
    const r = gradients.aggregate(t.id);
    log("- result %s", r);
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
  Map<number, TapeEntry>] {
  const tensorStack = [target.id];
  const oidLookup = new Map<number, TapeEntry>();
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
      if (usageCounts.get(inputId) == 1 && tape.tensorToOp.has(inputId) &&
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
function watch(t: ChainableTensor) {
  for (const tape of tapeStack) {
    tape.watch(t);
  }
}

export function recordOp(tapeEntry: TapeEntry) {
  for (const tape of tapeStack) {
    tape.recordOp(tapeEntry);
  }
}

export class GradientCollector {
  // Maps tensor id -> gradient tensor array
  private map = new Map<number, ChainableTensor[]>();

  append(tid: number, grad: ChainableTensor): void {
    if (this.map.has(tid)) {
      this.map.get(tid).push(grad);
    } else {
      this.map.set(tid, [grad]);
    }
  }

  // Sum up the gradients for a given tensor id.
  aggregate(tid: number): ChainableTensor {
    if (!this.map.has(tid) || this.map.get(tid).length == 0) {
      // TODO(scalar) Handle non-scalar shapes.
      return convertChainable(0);
    }
    const grads = this.map.get(tid);
    //log('aggregate tid %d ngrads %d', tid, grads.length);
    let sum = grads[0];
    for (let i = 1; i < grads.length; i++) {
      sum = sum.add(grads[i]);
    }
    return sum;
  }
}
