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

import { assert } from "../src/util";
import { test } from "../tools/tester";
import { transpile } from "./nb_transpiler";

const t = (src, out) => test(
  function nb_transpiler() {
    const transpiled = transpile(src);
    assert(transpiled === out,
           "Actual:   " + JSON.stringify(transpiled) + "\n" +
           "Expected: " + JSON.stringify(out));
    // Try to eval the transpiled source code. Note that this only verifies that
    // the transpiled code can be parsed correctly; the function body isn't run.
    const fn = eval(transpiled);
    assert(typeof fn === "function");
  });

/* tslint:disable:max-line-length */

t("",
  "(async (__global, __import, console) => {\n\n})");
t("import defaultExport from 'module-name';",
  "(async (__global, __import, console) => {\nvoid (({_:{default:__global.defaultExport}} = {_:await __import('module-name')}))\n})");
t("import * as name from 'module-name';",
  "(async (__global, __import, console) => {\nvoid (({_:__global.name} = {_:await __import('module-name')}))\n})");
t("import { export1 } from 'module-name';",
  "(async (__global, __import, console) => {\nvoid (({_:{export1:__global.export1}} = {_:await __import('module-name')}))\n})");
t("import { export1 as alias1 } from 'module-name';",
  "(async (__global, __import, console) => {\nvoid (({_:{export1:__global.alias1}} = {_:await __import('module-name')}))\n})");
t("import { export1, export2 } from 'module-name';",
  "(async (__global, __import, console) => {\nvoid (({_:{export1:__global.export1},_:{export2:__global.export2}} = {_:await __import('module-name')}))\n})");
t("import { export1, export2 as alias2 } from 'module-name';",
  "(async (__global, __import, console) => {\nvoid (({_:{export1:__global.export1},_:{export2:__global.alias2}} = {_:await __import('module-name')}))\n})");
t("import defaultExport, { export1 } from 'module-name';",
  "(async (__global, __import, console) => {\nvoid (({_:{default:__global.defaultExport},_:{export1:__global.export1}} = {_:await __import('module-name')}))\n})");
t("import defaultExport, * as name from 'module-name';",
  "(async (__global, __import, console) => {\nvoid (({_:{default:__global.defaultExport},_:__global.name} = {_:await __import('module-name')}))\n})");
t("import 'module-name';",
  "(async (__global, __import, console) => {\nreturn (await __import('module-name'))\n})");
t("var x = y;",
  "(async (__global, __import, console) => {\nvoid ((__global.x = y));\n})");
t("const x = y;",
  "(async (__global, __import, console) => {\nvoid ((__global.x = y));\n})");
t("let x = y;",
  "(async (__global, __import, console) => {\nvoid ((__global.x = y));\n})");
t("let a, b, c, d;",
  "(async (__global, __import, console) => {\nvoid ((__global.a= undefined), (__global.b= undefined), (__global.c= undefined), (__global.d= undefined));\n})");
t("var { a: e, b: f, c: { g } } = { a: 1 };",
  "(async (__global, __import, console) => {\nvoid (({ a: __global.e, b: __global.f, c: { g:__global.g } } = { a: 1 }));\n})");
t("var [d, e] = [2, 3];",
  "(async (__global, __import, console) => {\nvoid (([__global.d, __global.e] = [2, 3]));\n})");
t("var [[e], [[f]]] = [];",
  "(async (__global, __import, console) => {\nvoid (([[__global.e], [[__global.f]]] = []));\n})");
t("({ a: b.c } = {});",
  "(async (__global, __import, console) => {\nreturn (({ a: b.c } = {}));\n})");
t("[a, b] = [c, d];",
  "(async (__global, __import, console) => {\nreturn ([a, b] = [c, d]);\n})");
t("a = 3;",
  "(async (__global, __import, console) => {\nreturn (a = 3);\n})");
t("function get_me_global() {}",
  "(async (__global, __import, console) => {\nvoid (__global.get_me_global=function get_me_global() {});\n})");
t("function a() { var leave_me_alone; }",
  "(async (__global, __import, console) => {\nvoid (__global.a=function a() { var leave_me_alone; });\n})");
t("function a() { var leave_me_alone = 1; }",
  "(async (__global, __import, console) => {\nvoid (__global.a=function a() { var leave_me_alone = 1; });\n})");
t("function a() { var { leave_me_alone } = {}; }",
  "(async (__global, __import, console) => {\nvoid (__global.a=function a() { var { leave_me_alone } = {}; });\n})");
t("if (true) { var make_me_a_global = 1; }",
  "(async (__global, __import, console) => {\nif (true) { void ((__global.make_me_a_global = 1)); }\n})");
t("if (true) { var { make_me_a_global } = {}; }",
  "(async (__global, __import, console) => {\nif (true) { void (({ make_me_a_global:__global.make_me_a_global } = {})); }\n})");
t("if (true) { let leave_me_alone = 1; }",
  "(async (__global, __import, console) => {\nif (true) { let leave_me_alone = 1; }\n})");
t("if (true) { let { leave_me_alone } = {}; }",
  "(async (__global, __import, console) => {\nif (true) { let { leave_me_alone } = {}; }\n})");
t("if (true) { const leave_me_alone = 1; }",
  "(async (__global, __import, console) => {\nif (true) { const leave_me_alone = 1; }\n})");
t("if (true) { const { leave_me_alone } = {}; }",
  "(async (__global, __import, console) => {\nif (true) { const { leave_me_alone } = {}; }\n})");
