import * as gendoc from "./gendoc";
import { test } from "./test";
import { assert } from "./util";

test(function testSmoke() {
  const docs = gendoc.genJSON();
  assert(docs.length > 5);
  assert(docs.map(e => e.name).indexOf("Tensor") >= 0);
  const html = gendoc.toHTML(docs);
  assert(html.length > 0);
});

test(function testMarkupDocStr() {
  const docstr = [
    "hello",
    "",
    "  x = 1 + 2;",
    "",
    "world",
  ].join("\n");
  const actual = gendoc.markupDocStr(docstr);
  const expected = [
    "hello",
    "",
    "<script type=notebook>",
    "x = 1 + 2;",
    "</script>",
    "",
    "world",
  ].join("\n");
  // console.log("actual", JSON.stringify(actual));
  // console.log("expected", JSON.stringify(expected));
  assert(actual === expected);
});

test(function testMarkupDocStr2() {
  const docstr = [
    "like this:",
    "",
    "  params.forEach((tensor, name) => {",
    "    console.log(tensor);",
    "  });",
  ].join("\n");
  const actual = gendoc.markupDocStr(docstr);
  const expected = [
    "like this:",
    "",
    "<script type=notebook>",
    "params.forEach((tensor, name) => {",
    "  console.log(tensor);",
    "});",
    "</script>",
  ].join("\n");
  // console.log("actual", JSON.stringify(actual));
  // console.log("expected", JSON.stringify(expected));
  assert(actual === expected);
});
