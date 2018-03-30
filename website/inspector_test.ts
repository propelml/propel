import { h, render } from "preact";
import { ones } from "../src/api";
import { assert, assertEqual, delay } from "../src/util";
import { test, testBrowser } from "../tools/tester";
import { Inspector } from "./inspector";
import { serialize, unserialize } from "./serializer";

// tslint:disable:object-literal-sort-keys

test(async function inspector_serilizer() {
  const testData = {
    number: 12,
    object: {
      string: "Test"
    }
  };
  const data = unserialize(serialize(testData));
  assertEqual(data, {
    "type": "object",
    "data": {
      "number": {
        "type": "number",
        "data": 12
      },
      "object": {
        "type": "object",
        "data": {
          "string": {
            "type": "string",
            "data": "Test"
          }
        },
        "cons": "Object"
      }
    },
    "cons": "Object"
  });
});

test(async function inspector_serilizerRecursive() {
  const a = { b: 5, a: null };
  a.a = a;
  const data = unserialize(serialize(a));
  assert(data.data.a.data.b.data === 5);
});

test(async function inspector_serilizerSymbols() {
  const s = Symbol("test");
  const a = {[s]: "Propel"};
  const data = unserialize(serialize(a));
  assertEqual(data, {
    "type": "object",
    "data": {
      "Symbol(test)": {
        "type": "string",
        "data": "Propel"
      }
    },
    "cons": "Object"
  });
});

testBrowser(async function inspector_component() {
  const object = {
    number: 21,
    tensor: ones([24, 24]),
    x: 24,
    r: null
  };
  object.r = object;
  const element = h(Inspector, {
    object: unserialize(serialize(object))
  });
  render(element, document.body);
  // It should render.
  const root = document.getElementsByClassName("depth-0")[0];
  assert(root.className === "tree-node collapsed depth-0");
  // It should contain 2 elements.
  assert(root.childElementCount === 2, "2. Wrong number of childs.");
  // Root element should be expanded by default.
  assert(root.children[1].childElementCount === 4, "3. Root is not expanded.");
  const tensorRoot = root.children[1].children[1];
  const classList = tensorRoot.classList;
  // It should not be expanded by default.
  assert(!classList.contains("collapsed"), "4. Sub-node collapsed by default.");
  // Clicking on arrow should work.
  (tensorRoot.children[0] as HTMLElement).click();
  // Preact setState is asynchronous so we wait.
  await delay(100);
  assert(classList.contains("collapsed"), "5. Arrow does not work.");
});
