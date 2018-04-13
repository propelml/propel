import { h, render } from "preact";
import { ones } from "../src/api";
import { assertEqual } from "../src/util";
import { test, testBrowser } from "../tools/tester";
import { Inspector } from "./inspector";
import { describe, DescriptorSet, InspectorOptions } from "./serializer";

test(async function inspector_describe() {
  const t = (value: any, descriptors: DescriptorSet,
             options?: InspectorOptions): void => {
    const actual = describe([value], options);
    const expected = { descriptors, roots: [0] };
    assertEqual(actual, expected);
  };

  // A random date used in the test.
  const sometime = "2090-01-02T02:04:05.678Z";

  // tslint:disable:max-line-length
  // tslint:disable:no-construct
  // tslint:disable:object-literal-sort-keys

  // Simple primitives.
  t(12.34,
    [{ type: "number", value: "12.34" }]);
  t("abcd",
    [{ type: "string", value: "abcd" }]);
  t(false,
    [{ type: "boolean", value: "false" }]);
  t(true,
    [{ type: "boolean", value: "true" }]);
  t(null,
    [{ type: "null" }]);
  t(undefined,
    [{ type: "undefined" }]);
  t(Symbol(),
    [{ type: "symbol", value: "Symbol()" }]);

  // Objects and boxed primitives.
  t([1, 2, 3],
    [{ type: "array", length: 3, ctor: "Array", props: [
       { key: 1, value: 2, hidden: false },
       { key: 3, value: 4, hidden: false },
       { key: 5, value: 6, hidden: false }] },
     { type: "string", value: "0" },
     { type: "number", value: "1" },
     { type: "string", value: "1" },
     { type: "number", value: "2" },
     { type: "string", value: "2" },
     { type: "number", value: "3" }]);
  t({ key1: "value1", key2: "value2" },
    [{ type: "object", ctor: "Object", props: [
       { key: 1, value: 2, hidden: false },
       { key: 3, value: 4, hidden: false }] },
     { type: "string", value: "key1" },
     { type: "string", value: "value1" },
     { type: "string", value: "key2" },
     { type: "string", value: "value2" }]);
  t(new Date(sometime),
    [{ type: "box", primitive: { type: "date", value: sometime }, ctor: "Date", props: [] }]);
  t(/acme/gi,
    [{ type: "box", primitive: { type: "regexp", value: "/acme/gi" }, ctor: "RegExp", props: [] }]);
  t(new Uint32Array(3),
    [{ type: "array", length: 3, ctor: "Uint32Array", props: [
       { key: 1, value: 2, hidden: false },
       { key: 3, value: 2, hidden: false },
       { key: 4, value: 2, hidden: false }] },
     { type: "string", value: "0" },
     { type: "number", value: "0" },
     { type: "string", value: "1" },
     { type: "string", value: "2" }]);
  t(new Boolean(false),
    [{ type: "box", primitive: { type: "boolean", value: "false" }, ctor: "Boolean", props: [] }]);
  t(new Number(3),
    [{ type: "box", primitive: { type: "number", value: "3" }, ctor: "Number", props: [] }]);
  t(new String("hello world"),
    [{ type: "box", primitive: { type: "string", value: "hello world" }, ctor: "String", props: [] }]);
  // Object without prototype.
  t(Object.create(null),
    [{ type: "object", ctor: null, props: [] }]);

  // Hidden properties.
  t([1],
    [{ type: "array", length: 1, ctor: "Array", props: [
       { key: 1, value: 2, hidden: false },
       { key: 3, value: 2, hidden: true }] },
     { type: "string", value: "0" },
     { type: "number", value: "1" },
     { type: "string", value: "length" }],
    { showHidden: true });
  t(() => {},
    [{ type: "function", name: "", async: false, class: false, generator: false, ctor: "Function", props: [
       { key: 1, value: 2, hidden: true },
       { key: 3, value: 4, hidden: true }] },
     { type: "string", value: "length" },
     { type: "number", value: "0" },
     { type: "string", value: "name" },
     { type: "string", value: "" }],
    { showHidden: true });

  // An object with some unusual keys.
  const obj = {
    "042": "not numerical",
    "142": "numerical",
    normal: 19,
    [Symbol("my symbol")]: "symbolic key comes last",
    get getp() { return 0; },
    set setp(v) {},
    get getsetp() { return 0; },
    set getsetp(v) {}
  };
  // ...and a circular reference.
  obj.cycle = obj;
  t(obj,
    [{ type: "object", ctor: "Object", props: [
       { key: 1, value: 2, hidden: false },
       { key: 3, value: 4, hidden: false },
       { key: 5, value: 6, hidden: false },
       { key: 7, value: 8, hidden: false },
       { key: 9, value: 10, hidden: false },
       { key: 11, value: 12, hidden: false },
       { key: 13, value: 0, hidden: false },
       { key: 14, value: 15, hidden: false }] },
     { type: "string", value: "142" },
     { type: "string", value: "numerical" },
     { type: "string", value: "042" },
     { type: "string", value: "not numerical" },
     { type: "string", value: "normal" },
     { type: "number", value: "19" },
     { type: "string", value: "getp" },
     { type: "getter" },
     { type: "string", value: "setp" },
     { type: "setter" },
     { type: "string", value: "getsetp" },
     { type: "gettersetter" },
     { type: "string", value: "cycle" },
     { type: "symbol", value: "Symbol(my symbol)" },
     { type: "string", value: "symbolic key comes last" }]);

  // Built-in objects with some extra properties tucked on.
  const curry = o => Object.assign(o, { "77": "xx", yy: "zz" });
  t(curry(new Date(sometime)),
    [{ type: "box", primitive: { type: "date", value: sometime }, ctor: "Date", props: [
       { key: 1, value: 2, hidden: false },
       { key: 3, value: 4, hidden: false }] },
     { type: "string", value: "77" },
     { type: "string", value: "xx" },
     { type: "string", value: "yy" },
     { type: "string", value: "zz" }]);
  t(curry(new String("hello world")),
    [{ type: "box", primitive: { type: "string", value: "hello world" }, ctor: "String", props: [
       { key: 1, value: 2, hidden: false },
       { key: 3, value: 4, hidden: false }] },
     { type: "string", value: "77" },
     { type: "string", value: "xx" },
     { type: "string", value: "yy" },
     { type: "string", value: "zz" }]);
  // Numerical key can't be assigned to TypedArray.
  t(curry(new Uint32Array(3)),
    [{ type: "array", length: 3, ctor: "Uint32Array", props: [
       { key: 1, value: 2, hidden: false },
       { key: 3, value: 2, hidden: false },
       { key: 4, value: 2, hidden: false },
       { key: 5, value: 6, hidden: false }] },
     { type: "string", value: "0" },
     { type: "number", value: "0" },
     { type: "string", value: "1" },
     { type: "string", value: "2" },
     { type: "string", value: "yy" },
     { type: "string", value: "zz" }]);

  // Subclasses of built-in objects.
  class MyArray extends Array {}
  t(MyArray.of("first", "second"),
    [{ type: "array", length: 2, ctor: "MyArray", props: [
       { key: 1, value: 2, hidden: false },
       { key: 3, value: 4, hidden: false }] },
     { type: "string", value: "0" },
     { type: "string", value: "first" },
     { type: "string", value: "1" },
     { type: "string", value: "second" }]);
  class MyFunction extends Function {}
  t(new MyFunction("a", "b", "return a + b"),
    [{ type: "function", name: "anonymous", async: false, class: false, generator: false, ctor: "MyFunction", props: [] }]);
  class MyNumber extends Number {}
  t(new MyNumber(11),
    [{ type: "box", primitive: { type: "number", value: "11" }, ctor: "MyNumber", props: [] }]);
  class MyString extends String {}
  t(new MyString("hello mars"),
    [{ type: "box", primitive: { type: "string", value: "hello mars" }, ctor: "MyString", props: [] }]);

  // Functions of different flavors.
  t(function add(a, b) { return a + b; },
    [{ type: "function", name: "add", async: false, class: false, generator: false, ctor: "Function", props: [] }]);
  t(function(a, b) { return a + b; },
    [{ type: "function", name: "", async: false, class: false, generator: false, ctor: "Function", props: [] }]);
  t((a, b) => a + b,
    [{ type: "function", name: "", async: false, class: false, generator: false, ctor: "Function", props: [] }]);
  t(a => a,
    [{ type: "function", name: "", async: false, class: false, generator: false, ctor: "Function", props: [] }]);
  t(() => 0,
    [{ type: "function", name: "", async: false, class: false, generator: false, ctor: "Function", props: [] }]);
  t(async function add(a, b) { return a + b; },
    [{ type: "function", name: "add", async: true, class: false, generator: false, ctor: "AsyncFunction", props: [] }]);
  t(async function(a, b) { return a + b; },
    [{ type: "function", name: "", async: true, class: false, generator: false, ctor: "AsyncFunction", props: [] }]);
  t(async(a, b) => a + b,
    [{ type: "function", name: "", async: true, class: false, generator: false, ctor: "AsyncFunction", props: [] }]);
  t(async a => a,
    [{ type: "function", name: "", async: true, class: false, generator: false, ctor: "AsyncFunction", props: [] }]);
  t(async() => 0,
    [{ type: "function", name: "", async: true, class: false, generator: false, ctor: "AsyncFunction", props: [] }]);
  t(function* Generator() {},
    [{ type: "function", name: "Generator", async: false, class: false, generator: true, ctor: "GeneratorFunction", props: [] }]);
  t(class Foo {},
    [{ type: "function", name: "Foo", async: false, class: true, generator: false, ctor: "Function", props: [] }]);
  t(class Bar extends String {},
    [{ type: "function", name: "Bar", async: false, class: true, generator: false, ctor: "Function", props: [] }]);
  t(new Function("a", "b", "return a + b"),
    [{ type: "function", name: "anonymous", async: false, class: false, generator: false, ctor: "Function", props: [] }]);

  // tslint:enable:max-line-length
  // tslint:enable:no-construct
  // tslint:enable:object-literal-sort-keys
});

testBrowser(async function inspector_component() {
  // We'll simulate console.log(val1, val2, val3).
  const val1 = {
    list: [9, 8, 7],
    number: 21,
    self: null,
    string: "text",
    tensor: ones([3, 3]).cast("int32")
  };
  val1.self = val1;
  const val2 = 42;
  const val3 = { "hello": "world" };

  // Expected output in textual form.
  const output = `
    {
      list: Array(3) [
        0: 9
        1: 8
        2: 7
      ]
      number: 21
      self: [circular]
      string: "text"
      tensor: Tensor(int32 3✕3) [
        [1, 1, 1]
        [1, 1, 1]
        [1, 1, 1]
      ]
    }
    42
    {
      hello: "world"
    }`;

  // Render.
  const el = h(Inspector, describe([val1, val2, val3]));
  render(el, document.body);

  // Check the textual representation.
  const root = document.getElementsByClassName("inspector")[0] as HTMLElement;
  const actual = root.innerText.trim().replace(/\s+/g, " ");
  const expected = output.trim().replace(/\s+/g, " ");
  assertEqual(actual, expected);
});
