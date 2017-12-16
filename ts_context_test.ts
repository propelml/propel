import * as assert from "assert";
import { Context, fileext } from "./ts_context";

assert.strictEqual(fileext(""), "");
assert.strictEqual(fileext(".dotfile"), "dotfile");
assert.strictEqual(fileext("a/dir.weird/.dotfile"), "dotfile");
assert.strictEqual(fileext("a/dir.weird/file"), "");
assert.strictEqual(fileext("a/dir.weird/file"), "");
assert.strictEqual(fileext("a/dir.weird/file.ext"), "ext");
assert.strictEqual(fileext("a/dir/"), "");
assert.strictEqual(fileext("file"), "");
assert.strictEqual(fileext("file."), "");
assert.strictEqual(fileext("file.ext"), "ext");
assert.strictEqual(fileext("file.foo."), "");
assert.strictEqual(fileext("file.foo.bar"), "bar");

const ctx = new Context();

async function pass(code, retval = undefined) {
  const { result, error } = await ctx.eval(code);
  if (error !== undefined) {
    console.log(error);
  }
  assert(error === undefined);
  if (retval !== undefined) {
    assert(result === retval);
  }
}

async function fail(code) {
  const { result, error } = await ctx.eval(code);
  assert(error);
}

(async() => {
  await pass("var a = 3;");
  await pass("let b = 42");
  await pass("a", 3);
  await pass("b", 42);
  await pass("Math.PI", Math.PI);
  await pass("console.log('hi')");
  await pass("console.log('one');console.log('two');42");
  await fail("import * as x from 'does not exist'; x");
  await pass(`import * as jquery from
              'https://code.jquery.com/jquery-3.2.1.slim.min.js'; jquery`);
  await pass("typeof jquery", "function");
  await pass("import * as u from './util'; u");
  await pass(`// Multi-line inputs are supported.
              (async () => {
                console.log('In here!');
                await new Promise((res, rej) => setTimeout(res, 1));
              })();`);
  await pass("let s: string = '44'; s", "44");
  await pass("import * as dl from './dl'; dl");
  await pass("import * as ut from './util_test'; ut");
  // await fail("const a; const a;"); -- TODO: enable type checking.
  await fail("throw new Error('wut')");
  await fail("sdfk.gjhsdk;lfghskl,dfjgkl");
  console.log("PASS");
})().catch((error) => {
  // Ridiculous this is necessary
  console.error(error);
  process.exit(1);
});
