import { h, render } from "preact";
import { ones } from "../src/api";
import { assert, assertEqual } from "../src/util";
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
        "data": 12,
        "isCircular": false
      },
      "object": {
        "type": "object",
        "data": {
          "string": {
            "type": "string",
            "data": "Test",
            "isCircular": false
          }
        },
        "cons": "Object",
        "isCircular": false
      }
    },
    "cons": "Object"
  });
});

test(async function inspector_serializerRecursive() {
  const a = { b: 5, a: null };
  a.a = a;
  const data = unserialize(serialize(a));
  assert(data.data.a.isCircular);
  assertEqual(data.data.a.data.b.data, 5);
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
        "data": "Propel",
        "isCircular": false
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
  const root = document.getElementsByClassName("depth-0")[0] as HTMLElement;
  assertEqual(root.innerText.replace(/\s/g, ""), `Object{number:21,tensor:` +
    `Tensor(dtype="float32",shape=[24,24]),x:24,r:[Circular]}`);
});
