import { test } from "../tools/tester";
import * as pr from "./api";
import * as npy from "./npy";
import * as util from "./tensor_util";
import * as types from "./types";
import { IS_NODE } from "./util";

// Because the python interop test requires python and numpy installed
// it is turned off by default here. The option is left for debugging
// purposes.
// TODO Make this work on our CI servers.
const testPython = false;

test(async function npy_load() {
  // python -c "import numpy as np; np.save('1.npy', [1.5, 2.5])"
  let t = await npy.load("src/testdata/1.npy");
  util.assertAllEqual(t, [ 1.5, 2.5 ]);
  util.assertShapesEqual(t.shape, [2]);
  util.assert(t.dtype === "float32");

  // python -c "import numpy as np; np.save('2.npy', [[1.5, 43], [13, 2.5]])"
  t = await npy.load("src/testdata/2.npy");
  util.assertAllEqual(t, [[1.5, 43], [13, 2.5]]);
  util.assertShapesEqual(t.shape, [2, 2]);
  util.assert(t.dtype === "float32");

  // python -c "import numpy as np; np.save('3.npy', [[[1,2,3],[4,5,6]]])"
  t = await npy.load("src/testdata/3.npy");
  util.assertAllEqual(t, [[[1, 2, 3], [4, 5, 6]]]);
  util.assertShapesEqual(t.shape, [1, 2, 3]);
  util.assert(t.dtype === "int32");

  /*
   python -c "import numpy as np; np.save('4.npy', \
          np.array([0.1, 0.2], 'float32'))"
  */
  t = await npy.load("src/testdata/4.npy");
  util.assertAllClose(t, [0.1, 0.2]);
  util.assertShapesEqual(t.shape, [2]);
  util.assert(t.dtype === "float32");

  /*
   python -c "import numpy as np; np.save('uint8.npy', \
          np.array([0, 127], 'uint8'))"
  */
  t = await npy.load("src/testdata/uint8.npy");
  util.assertAllClose(t, [0, 127]);
  util.assertShapesEqual(t.shape, [2]);
  util.assert(t.dtype === "uint8");
});

test(async function npy_serialize() {
  const t = pr.tensor([ 1.5, 2.5 ]);
  const ab = await npy.serialize(t);
  // Now try to parse it.
  const tt = npy.parse(ab);
  util.assertAllEqual(t, tt);
});

if (IS_NODE && testPython) {
  test(async function npy_pythonInterop() {
    await checkPython("[ 1.5  2.5]", [ 1.5, 2.5 ]);
    await checkPython("[ 1.  2.]", [ 1, 2 ]);
    await checkPython("[[1 2]\n [3 4]]",
      pr.tensor([[ 1, 2 ], [3, 4]], { dtype: "int32" }));
  });

  async function checkPython(expected: string,
                             t: types.TensorLike): Promise<void> {
    const ab = await npy.serialize(pr.tensor(t));
    const actual = parsePython(new Buffer(ab));
    if (expected !== actual) {
      throw Error(`expected: "${expected}" actual: "${actual}"`);
    }
  }

  function parsePython(npyBuffer: Buffer): string {
    const { execSync } = require("child_process");
    const code = pyCode.trim().replace("\n", "; ");
    const cmd = `python -c "${code}"`;
    const stdout = execSync(cmd, {
      encoding: "ascii",
      input: npyBuffer
    });
    return stdout.trim();
  }
}

const pyCode = `
import io
import numpy as np
import sys
buf = io.BytesIO(sys.stdin.read())
print np.load(buf)
`;
