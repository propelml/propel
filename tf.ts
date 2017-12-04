import { assertEqual } from "./util"; 

export function maybeRequireBinding() {
  // If we're in the browser, don't even attempt it.
  if (typeof window !== 'undefined') return null;

  // This is to set the backend to either web or tensorflow.
  // Use this on the command line:
  //   PROPEL=web node myprogram.js
  // This is used in tools/presubmit.sh to run tests on both backends through
  // node.
  const opts = (process.env.PROPEL || "").split(",");
  if (opts.indexOf("web") >= 0) return null;

  // Now require the compiled tensorflow-binding.node
  // When using ts-node, we are in the root dir, after compiling to
  // javascript, we are in the dist dir.
  const toAttempt = [
    '../build/Debug/tensorflow-binding.node',
    '../build/Release/tensorflow-binding.node',
    './build/Debug/tensorflow-binding.node',
    './build/Release/tensorflow-binding.node',
  ];
  const fs = require("fs");
  const path = require("path");
  for (const fn of toAttempt) {
    if (fs.existsSync(path.join(__dirname, fn))) {
      return require(fn);
    }
  }
  return null;
}

export let binding = maybeRequireBinding();

// Auto create context for now.
export let ctx;
if (binding) {
  ctx = new binding.Context();
}

// Sugar for single value ops.
export function execute0(opName, inputs, attrs) {
  const r = binding.execute(ctx, opName, attrs, inputs);
  assertEqual(r.length, 1);
  return r[0];
}
