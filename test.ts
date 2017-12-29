
import { IS_NODE } from "./util";

const filterExpr = IS_NODE ? process.argv[2]
                           : new URL(location.href).hash.slice(1);
const filterRegExp = filterExpr ? new RegExp(filterExpr, "i") : null;
const tests = [];

function filter(fn) {
  if (filterRegExp) {
    return filterRegExp.test(fn.name);
  } else {
    return true;
  }
}

export function test(fn) {
  if (!fn.name) {
    throw new Error("Test function may not be anonymous");
  }
  if (filter(fn)) {
    tests.push(fn);
  }
}

async function runTests() {
  let passed = 0;
  let failed = 0;

  for (let i = 0; i < tests.length; i++) {
    const fn = tests[i];

    console.warn("%d/%d +%d -%d: %s",
                 i + 1,
                 tests.length,
                 passed,
                 failed,
                 fn.name);

    try {
      const r = await fn();
      passed++;
    } catch (e) {
      console.error(e);
      failed++;
    }
  }

  console.warn(`DONE. passed: ${passed}, failed: ${failed}`);

  if (failed === 0) {
    // All good.
  } else if (IS_NODE) {
    process.exit(1);
  } else {
    throw new Error(`There were ${failed} test failures.`);
  }
}

setTimeout(runTests, 0);
