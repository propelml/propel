let binding;
try {
  binding = require('./build/Debug/tensorflow-binding.node');
} catch (e) {
  binding = require('./build/Release/tensorflow-binding.node');
}
let ctx = new binding.Context();
console.assert(ctx instanceof binding.Context);

let typedArray = new Float32Array([1, 2, 3, 4, 5, 6]);
let a = new binding.Tensor(typedArray, [2, 3]);
let b = new binding.Tensor(typedArray, [3, 2]);
console.assert(a.device == "CPU:0");
console.assert(b.device == "CPU:0");

let opAttrs = { transpose_a: false, transpose_b: false };
let retvals = binding.execute(ctx, "MatMul", opAttrs, [a, b]);
let r = retvals[0];
console.log("r", r);
console.assert(r.device == "CPU:0");

console.log("PASS");
