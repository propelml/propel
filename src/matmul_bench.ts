
import { ones } from "./api";

const t1 = ones([100, 100]);
const t2 = ones([100, 100]).add(1);

let count = 100;
let total = 0;
const begin = Date.now() / 1000;
let end = begin;

for (let i = 0; i < 20; i++) {
  const start = end;

  for (let i = 0; i < count; i++) {
    t1.matmul(t2);
  }

  end = Date.now() / 1000;
  const elapsed = end - start;
  const throughput = count / elapsed;

  total += count;
  count = Math.round(total / (end - begin));

  console.log(
    `time: ${elapsed}s  count: ${count}  throughput: ${throughput}/s`);
}
