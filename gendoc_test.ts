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
import * as gendoc from "./gendoc";
import { test } from "../test";
import { assert } from "../util";

test(async function gendoc_smoke() {
  const docs = gendoc.genJSON();
  console.log("length", docs.length);
  assert(docs.length > 5);
  assert(docs.map(e => e.name).indexOf("Tensor") >= 0);
  const html = gendoc.toHTML(docs);
  assert(html.length > 0);
});

test(async function gendoc_markupDocStr() {
  const docstr = [
    "hello",
    "",
    "  x = 1 + 2;",
    "",
    "world",
  ].join("\n");
  const actual = gendoc.markupDocStr(docstr);
  const expected = [
    "<p class='docstr'>hello",
    "",
    "</p><script type=notebook>",
    "x = 1 + 2;",
    "</script><p>",
    "",
    "world</p>",
  ].join("\n");
  // console.log("actual", JSON.stringify(actual));
  // console.log("expected", JSON.stringify(expected));
  assert(actual === expected);
});

test(async function gendoc_markupDocStr2() {
  const docstr = [
    "like this:",
    "",
    "  params.forEach((tensor, name) => {",
    "    console.log(tensor);",
    "  });",
  ].join("\n");
  const actual = gendoc.markupDocStr(docstr);
  const expected = [
    "<p class='docstr'>like this:",
    "",
    "</p><script type=notebook>",
    "params.forEach((tensor, name) => {",
    "  console.log(tensor);",
    "});",
    "</script>",
  ].join("\n");
  // console.log("actual", JSON.stringify(actual));
  // console.log("expected", JSON.stringify(expected));
  assert(actual === expected);
});
