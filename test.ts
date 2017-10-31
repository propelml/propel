import * as xx from './xx';

function tanh(x) {
  let y = xx.exp(xx.mul(-2.0, x));
  return xx.div(xx.sub(1.0, y), xx.add(1.0, y));
}

xx.assertClose(tanh(1.0).getValues()[0], 0.7615, "tanh broken");


let gradTanh = xx.grad(tanh);
let r = gradTanh(1.0)

console.log("actual", r);
console.log("expected", 0.419);
