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

import { Component, h } from "preact";
import { IS_WEB, isNumericalKey, randomString } from "../src/util";
import {
  BaseObjectDescriptor,
  InspectorData,
  PropertyDescriptor,
  ValueDescriptor
} from "./serializer";

type ElementLike = JSX.Element | string | null;
type ElementList = ElementLike[];

export class Inspector extends Component<InspectorData, void>{
  private parents = new Set(); // Used for circular reference detection.

  private renderKey(d: ValueDescriptor,
                    arrayIndex: number | null = null): ElementList {
    switch (d.type) {
      case "string":
        const value = d.value;
        // Keys that look like javascript identifiers are rendered without
        // quote marks. Numerical keys can be omitted entirely.
        const isSimpleKey = /^[a-z$_][\w$_]*$/i.test(value);
        const isArrayKey = (typeof arrayIndex === "number")
                           && isNumericalKey(value);
        if (isArrayKey || isSimpleKey) {
          // 12: {...}
          // simple_key: {...}
          let elements = [
            <span class="inspector__key cm-property">{ value }</span>,
            ": "
          ];
          // If the array keys are in the right order without holes, hide them
          // from the collapsed view.
          if (isArrayKey && Number(value) === arrayIndex) {
            elements = addClass(elements, "inspector--show-if-parent-expanded");
          }
          return elements;
        } else {
          // "my!weird*key\n": {...}
          return [
            <span class="inspector__key cm-string">
              { JSON.stringify(value) }
            </span>,
            ": "
          ];
        }
      case "symbol":
        // [Symbol(some symbol)]: {...}
        return [
          "[",
          <span class="inspector__key cm-property">{ d.value }</span>,
          "] :"
        ];
      default:
        // Keys are always either a string, symbol, or prototype key.
        throw new Error("Unexpected key type");
    }
  }

  private renderValue(d: ValueDescriptor): ElementList {
    // Render primitive values.
    switch (d.type) {
      case "boolean":
        return [<span class="cm-atom">{ d.value }</span>];
      case "date":
        return [<span class="cm-string-2">{ d.value }</span>];
      case "getter":
        return [<span class="cm-keyword">[getter]</span>];
      case "gettersetter":
        return [<span class="cm-keyword">[getter/setter]</span>];
      case "null":
        return [<span class="cm-atom">null</span>];
      case "number":
        return [<span class="cm-number">{ d.value }</span>];
      case "regexp":
        return [<span class="cm-string-2">{ d.value }</span>];
      case "setter":
        return [<span class="cm-keyword">[setter]</span>];
      case "string":
        return [<span class="cm-string">{ JSON.stringify(d.value) }</span>];
      case "symbol":
        return [<span class="cm-string">{ d.value }</span>];
      case "undefined":
        return [<span class="cm-atom">undefined</span>];
    }

    // Detect circular references.
    if (this.parents.has(d)) {
      return [<span class="cm-keyword">[circular]</span>];
    }
    this.parents.add(d);

    // Render the list of properties and other content that is expanded when the
    // expand button is clicked.
    const listItems: ElementList = [];

    if (d.type === "tensor") {
      // TODO: the tensor is currently formatted by Tensor.toString(),
      // and displayed in a monospace font. Make it nicer by using a table.
      // Remove the outer layer of square brackets added by toString().
      const formatted = d.formatted.replace(/(^\[)|(\]$)/g, "")
                                   .replace(/,\n+ /g, "\n");
      listItems.push(
        <li class="inspector__li inspector__prop">
          <div class="inspector__content inspector__content--pre">
            { formatted }
          </div>
        </li>
      );
    }
    // Add regular array/object properties.
    const isArray = d.type === "array";
    listItems.push(...d.props.map((prop, index) =>
      this.renderProperty(prop, isArray ? index : null)
    ));
    // Wrap the items in an <ul> element, but only if the list contains at
    // least one item.
    const list: ElementLike =
      (listItems.length > 0)
        ? <ul class="inspector__list inspector__prop-list">
            { ...listItems }
          </ul>
        : null;

    // Render the object header, containing the class name, sometimes
    // some metadata between parentheses, and boxed primitive content.
    const elements: ElementList = [];
    switch (d.type) {
      case "array":
        // If the constructor is "Array", it is hidden when collapsed.
        // Square brackets are used and they are always visible.
        //   [1, 2, 3, 4, 5, foo: "bar"]
        //   Array(3) [1, 2, 3, ...properties...]
        //   Float32Array(30) [...]
        let ctorElements = [
          <span class="cm-def">{ d.ctor }</span>,
          "(", <span class="cm-number">{ d.length }</span>, ") "
        ];
        if (d.ctor === "Array") {
          ctorElements = addClass(ctorElements, "inspector--show-if-expanded");
        }
        elements.push(...ctorElements, "[", list, "]");
        break;
      case "box":
        // The class name is omitted when it is "Date" or "RegExp".
        // Number, Boolean, String are shown to distinguish from primitives.
        // Properties are wrapped in curly braces; these are hidden when
        // there are no properties..
        //   Number:42 {...properties...}
        //   /abcd/i {...properties...}
        if (d.ctor !== "Date" && d.ctor !== "RegExp") {
          elements.push(<span class="cm-def">{ d.ctor }</span>, ":");
        }
        elements.push(
          ...this.renderValue(d.primitive),
          ...this.renderPropListWithBraces(d, list)
        );
        break;
      case "function":
        // Constructor name is only shown if it is nontrivial.
        //   function() {...properties...}
        //   class Foo
        //   async function* bar()
        //   MyFunction:function baz()
        if (!/^(Async)?(Generator)?Function$/.test(d.ctor)) {
          elements.push(<span class="cm-def">{ d.ctor }</span>, ":");
        }
        elements.push(
          <span class="cm-keyword">
            { d.async ? "async " : "" }
            { d.class ? "class" : "function"}
            { d.generator ? "*" : "" }
          </span>,
          d.name ? " " : "",
          d.name ? <span class="cm-def">{ d.name }</span> : "",
          d.class ? "" : "()",
          ...this.renderPropListWithBraces(d, list)
        );
        break;
      case "tensor":
        // Tensor(float32) [1]
        // Tensor(float32 15) [1, 2, 3, ...etc...]
        // Tensor(float32 2✕2) [[1, 2], [3, 4]]
        // Tensor(uint32 3✕4✕4) [[[[...tensor...]]], foo: bar]
        const isScalar = d.shape.length === 0 ||
                         (d.shape.length === 1 && d.shape[0] === 1);
        elements.push(
          <span class="cm-def">{ d.ctor }</span>,
          "(",
          <span class="cm-string-2">{ d.dtype }</span>
        );
        if (!isScalar) {
          for (const [dim, size] of d.shape.entries()) {
            elements.push(
              dim === 0 ? " " : "✕",
              <span class="cm-number">{ size }</span>
            );
          }
        }
        elements.push(") [", list, "]");
        break;
      case "object":
        // Constructor name is omitted when it's "Object", or when the
        // object doesn't have a prototype at all (ctor === null).
        // Curly braces are always shown, even when there are no properties.
        //   {a: "hello", b: "world"}
        //   MyObject {"#^%&": "something"}
        if (d.ctor !== null && d.ctor !== "Object") {
          elements.push(<span class="cm-def">{ d.ctor }</span>, " ");
        }
        elements.push("{", list, "}");
        break;
    }

    // Pop cycle detection stack.
    this.parents.delete(d);

    return elements;
  }

  private renderPropListWithBraces(d: BaseObjectDescriptor,
                                   list?: ElementLike): ElementList {
    if (!list) return [];
    // If the list contains items that are visible in collapsed mode, always
    // show the curly braces. Otherwise show them only when expanded.
    const hasVisibleProps = d.props.some(prop => !prop.hidden);
    const braceClass = hasVisibleProps ? "" : "inspector--show-if-expanded";
    return [
       ...addClass([" {"], braceClass),
       list,
       ...addClass(["}"], braceClass),
    ];
  }

  // Returns true if a descriptor needs an expand button.
  private isExpandable(d: ValueDescriptor): boolean {
    // If a reference is circular it can't be expanded.
    if (this.parents.has(d)) {
      return false;
    }
    // Tensors can be always expanded to show its values.
    if (d.type === "tensor") {
      return true;
    }
    // Objects can be expanded if they have properties.
    if ((d.type === "array" || d.type === "box" || d.type === "function" ||
         d.type === "object") && d.props.length > 0) {
      return true;
    }
    return false;
  }

  private renderExpandButton(): ElementLike {
    // Use a simple <a href="javascript:"> link, so it "just works" on pages
    // that have been pre-rendered into static html.
    const id = randomString();
    return <a id={ id } class="inspector__toggle"
              href={ `javascript:Inspector.toggle("${id}")` } />;
  }

  private renderProperty(prop: PropertyDescriptor,
                         arrayIndex: number | null = null): ElementLike {
    const key = this.props.descriptors[prop.key];
    const value = this.props.descriptors[prop.value];
    const attr = { class: "inspector__li inspector__prop" };
    if (prop.hidden) attr.class += " inspector__prop--hidden";
    return (
      <li { ...attr }>
       { this.isExpandable(value) && this.renderExpandButton() }
       <div class="inspector__content">
          { ...this.renderKey(key, arrayIndex) }
          { ...this.renderValue(value) }
        </div>
      </li>
    );
  }

  private renderRoot(): ElementLike {
    if (this.props.roots.length === 0) return "";
     // Map root ids to descriptors.
    const descriptors = this.props.descriptors;
    const roots = this.props.roots.map(id => descriptors[id]);
    // Count the number of items with children.
    const rootsWithChildren = roots.map(d => +this.isExpandable(d))
                                   .reduce((count, d) => count + d);
    let rootElement: ElementLike;
    if (rootsWithChildren <= 1) {
      // With at most one expandable item, place all roots in a single div.
      const elements = [].concat(...roots.map((d, index) => [
        index > 0 ? " " : "", // Space between items.
        ...this.renderValue(d)
      ]));
      rootElement = <div class="inspector__content">{ ...elements }</div>;
    } else {
      // With more than one expandable root, place all of them in a separate.
      // list item.
      const listItems = roots.map(d =>
        <li class="inspector__li inspector__root">
          { this.isExpandable(d) && this.renderExpandButton() }
          <div class="inspector__content">
            { ...this.renderValue(d) }
          </div>
        </li>
      );
      rootElement =
        <div class="inspector__content">
          <ul class="inspector__list">
            { ...listItems }
          </ul>
        </div>;
    }
    // If there are expandable items, add a top-level expand button.
    if (rootsWithChildren > 0) {
      rootElement =
        <ul class="inspector__list">
          <li class="inspector__li">
            { this.renderExpandButton() }
            { rootElement }
          </li>
        </ul>;
    }
    return rootElement;
  }

  render(): JSX.Element {
    return <div class="inspector cm-s-syntax">{ this.renderRoot() }</div>;
  }

  static toggle(id: string): void {
    const el = document.getElementById(id);
    const parent = el.parentNode as HTMLElement;
    parent.classList.toggle("inspector__li--expanded");
  }
}

// Helper function to attach a css class to a list of elements.
// The element list may contain strings; they are converted to spans.
function addClass(elements: ElementList, classes: string): ElementList {
  // Remove redundant whitespace from classes.
  classes = classes.split(/\s+/).filter(s => s !== "").join();
  // Return early when no classes are added.
  if (classes === "") return elements;
  const result: ElementList = [];
  let stringBuffer = "";
  for (const el of elements) {
    if (el === null) {
      // Remove empty elements.
    } else if (typeof el === "string") {
      // Merge adjacent strings so we don't create so many spans.
      stringBuffer += el;
    } else {
      if (stringBuffer) {
        // Wrap the buffered-up string in a <span> element and flush.
        result.push(<span class={ classes }>{ stringBuffer }</span>);
        stringBuffer = "";
      }
      // Add our css class to the existing element.
      const elClass = el.attributes.class;
      el.attributes.class = elClass ? `${elClass} ${classes}` : classes;
      result.push(el);
    }
  }
  if (stringBuffer) {
    result.push(<span class={ classes }>{ stringBuffer }</span>);
  }
  return result;
}

// Make Inspector available on the global object. This allows expand button
// to use a simple textual click handler, which survives serialization.
// We make the property non-enumerable to minimize interaction with other
// components.
if (IS_WEB) {
  Object.defineProperty(window, "Inspector", {
    enumerable: false,
    value: Inspector
  });
}
