/*!
   Copyright 2018 Propel http://propel.site/.  All rights reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

import { assert, assertEqual, global, globalEval } from "../src/util";
import { test } from "../tools/tester";
import { Transpiler } from "./nb_transpiler";

test(function nb_transpiler_transpile() {
  const t = (src: string, expectedBody: string): void => {
    const expected = [
      "(async function __transpiled_top_level_1__" +
        "(__global, __import, console) {",
      expectedBody,
      "})",
      "//# sourceURL=__transpiled_source_1__"
    ].join("\n");

    const transpiler = new Transpiler();
    const transpiled = transpiler.transpile(src, "test");

    assertEqual(transpiled, expected,
          "unexpected transpiled source code\n" +
          `actual:   ${JSON.stringify(transpiled)}\n` +
          `expected: ${JSON.stringify(expected)}`
    );

    // Try to eval the transpiled source code. Note that this only verifies that
    // the transpiled code can be parsed correctly; the function body isn't run.
    const fn = globalEval(transpiled);
    assertEqual(typeof fn, "function");
  };

  /* tslint:disable:max-line-length */
  t("", "");
  t("import defaultExport from 'module-name';",
    "void (({_:{default:__global.defaultExport}} = {_:await __import('module-name')}));");
  t("import * as name from 'module-name';",
    "void (({_:__global.name} = {_:await __import('module-name')}));");
  t("import { export1 } from 'module-name';",
    "void (({_:{export1:__global.export1}} = {_:await __import('module-name')}));");
  t("import { export1 as alias1 } from 'module-name';",
    "void (({_:{export1:__global.alias1}} = {_:await __import('module-name')}));");
  t("import { export1, export2 } from 'module-name';",
    "void (({_:{export1:__global.export1},_:{export2:__global.export2}} = {_:await __import('module-name')}));");
  t("import { export1, export2 as alias2 } from 'module-name';",
    "void (({_:{export1:__global.export1},_:{export2:__global.alias2}} = {_:await __import('module-name')}));");
  t("import defaultExport, { export1 } from 'module-name';",
    "void (({_:{default:__global.defaultExport},_:{export1:__global.export1}} = {_:await __import('module-name')}));");
  t("import defaultExport, * as name from 'module-name';",
    "void (({_:{default:__global.defaultExport},_:__global.name} = {_:await __import('module-name')}));");
  t("import 'module-name';",
    "return (await __import('module-name'));");
  t("var x = y;",
    "void ((__global.x = y));");
  t("const x = y;",
    "void ((__global.x = y));");
  t("let x = y;",
    "void ((__global.x = y));");
  t("let a, b, c, d;",
    "void ((__global.a= undefined), (__global.b= undefined), (__global.c= undefined), (__global.d= undefined));");
  t("var { a: e, b: f, c: { g } } = { a: 1 };",
    "void (({ a: __global.e, b: __global.f, c: { g:__global.g } } = { a: 1 }));");
  t("var [d, e] = [2, 3];",
    "void (([__global.d, __global.e] = [2, 3]));");
  t("var [[e], [[f]]] = [];",
    "void (([[__global.e], [[__global.f]]] = []));");
  t("0",
    "return (0)");
  t("({ a: b.c } = {});",
    "return (({ a: b.c } = {}));");
  t("[a, b] = [c, d];",
    "return ([a, b] = [c, d]);");
  t("a = 3;",
    "return (a = 3);");
  t("function get_me_global() {}",
    "void (__global.get_me_global=function get_me_global() {});");
  t("function a() { var leave_me_alone; }",
    "void (__global.a=function a() { var leave_me_alone; });");
  t("function a() { var leave_me_alone = 1; }",
    "void (__global.a=function a() { var leave_me_alone = 1; });");
  t("function a() { var { leave_me_alone } = {}; }",
    "void (__global.a=function a() { var { leave_me_alone } = {}; });");
  t("class GlobalClass {}",
    "void (__global.GlobalClass=class GlobalClass {});");
  t("class GlobalMyArray extends Array {}",
    "void (__global.GlobalMyArray=class GlobalMyArray extends Array {});");
  t("var X = class ClassExpression {};",
    "void ((__global.X = class ClassExpression {}));");
  t("if (true) { var make_me_a_global = 1; }",
    "if (true) { void ((__global.make_me_a_global = 1)); }");
  t("if (true) { var { make_me_a_global } = {}; }",
    "if (true) { void (({ make_me_a_global:__global.make_me_a_global } = {})); }");
  t("if (true) { let leave_me_alone = 1; }",
    "if (true) { let leave_me_alone = 1; }");
  t("if (true) { let { leave_me_alone } = {}; }",
    "if (true) { let { leave_me_alone } = {}; }");
  t("if (true) { const leave_me_alone = 1; }",
    "if (true) { const leave_me_alone = 1; }");
  t("if (true) { const { leave_me_alone } = {}; }",
    "if (true) { const { leave_me_alone } = {}; }");
  /* tslint:enable:max-line-length */
});

test(async function nb_transpiler_formatException() {
  const dummyImport = () => ({});
  const transpiler = new Transpiler();
  let chain = Promise.resolve();

  const t = (src: string, name: string, ...positions: string[]): void => {
    chain = chain.then(async() => {
      const transpiled = transpiler.transpile(src, name);
      const fn = globalEval(transpiled);

      let error;
      try {
        await fn(global, dummyImport, console);
      } catch (e) {
        error = e;
      }
      assert(error !== undefined, `function should throw\n${src}`);
      const stack = transpiler.formatException(error);
      for (const pos of positions) {
        assert(stack.indexOf(pos) !== -1,
               `stack should contain ${pos}\n${stack}`);
      }
    });
  };

  // 0........1.........2.........3.........4
  // 1234567890123456789012345678901234567890
  t("badfn()", "test1", "test1:1:1");
  t("var asdfg=42;badfn()", "test2", "test2:1:14");
  t("\n" +
    "import * as foo from 'bar';badfn()",
    "test3", "test3:2:28");
  t("function wierd() {\n" +
    "  return badfn();\n" +
    "} 1+2+3+wierd()",
    "test4", "test4:2:3", "test4:3:9");
  t("wierd()", "test5", "test4:2:3", "test5:1:1");

  await chain;
});

test(async function nb_transpiler_getEntryPoint() {
  const transpiler = new Transpiler();
  const getEntryPoint = () => transpiler.getEntryPoint();
  let checkEntryPointCalls = 0;
  const checkEntryPoint = (actual, expected) => {
    assertEqual(actual, expected,
           "unexpected entry point name\n" +
           `actual:   ${actual}\n` +
           `expected: ${expected}`);
    checkEntryPointCalls++;
  };
  const importHelpers = async() => ({ checkEntryPoint, getEntryPoint });
  let chain = Promise.resolve();

  const t = (src: string, name: string): void => {
    chain = chain.then(async() => {
      const transpiled = transpiler.transpile(src, name);
      const fn = globalEval(transpiled);
      await fn(global, importHelpers, console);
    });
  };

  t("import { checkEntryPoint, getEntryPoint } from 'aaa'", "setup");
  t("checkEntryPoint(getEntryPoint(), 'file1')",
    "file1");
  t("function elsewhere() { return getEntryPoint() }\n" +
    "checkEntryPoint(elsewhere(), 'file2');",
    "file2");
  t("checkEntryPoint(elsewhere(), 'file3')", "file3");
  t("const promise = new Promise(\n" +
    "  res => setTimeout(() => res(getEntryPoint()), 100));\n" +
    "checkEntryPoint(await promise, 'file4')",
    "file4");

  await chain;

  assertEqual(checkEntryPointCalls, 4,
         "unexpected number of checkEntryPoint calls were made\n");
});
