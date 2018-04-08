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
  PropertyDescriptor,
  ValueDescriptor
} from "./serializer";

type ElementLike = JSX.Element | string;
type ElementList = ElementLike[];
type Visibility = "always" | "collapsed" | "expanded" | "parent-expanded";

export interface InspectorProps { descriptors: ValueDescriptor[]; }

export class Inspector extends Component<InspectorProps, void>{
  private renderKey(d: ValueDescriptor,
                    arrayIndex: number | null = null): ElementList {
    switch (d.type) {
      case "string":
        const value = d.value;
        // Keys that look like javascript identifiers are rendered without
        // quote marks. Numerical keys can be omitted entirely.
        const isSimpleKey =  /^[a-z$_][\w$_]*$/i.test(value);
        const isArrayKey = (typeof arrayIndex === "number")
                           && isNumericalKey(value);
        if (isArrayKey || isSimpleKey) {
          // 12: {...}
          // simple_key: {...}
          // If the array keys are in the right order without holes, hide them
          // from the collapsed view.
          const keyVisible = (isArrayKey && Number(value) === arrayIndex)
                              ? "parent-expanded" : "always";
          return showWhen(keyVisible,
                          <span class="cm-property">{ value }</span>, ": ");
        } else {
          // "my!weird*key\n": {...}
          const quoted = JSON.stringify(value);
          return [<span class="cm-property">{ quoted }</span>, ": "];
        }
      case "symbol":
        // [Symbol(some symbol)]: {...}
        return ["[", <span class="cm-property">{ d.value }</span>, "]: "];
      case "proto":
        // [prototype]: {...}
        return ["[", <span class="cm-property">prototype</span>, "]: "];
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
      case "circular":
        return [<span class="cm-keyword">[circular]</span>];
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
      case "proto":
        return [<span class="cm-keyword">[prototype]</span>];
      case "regexp":
        return [<span class="cm-string-2">{ d.value }</span>];
      case "setter":
        return [<span class="cm-keyword">[Setter]</span>];
      case "string":
        return [<span class="cm-string">{ JSON.stringify(d.value) }</span>];
      case "symbol":
        return [<span class="cm-string">{ d.value }</span>];
      case "undefined":
        return [<span class="cm-atom">undefined</span>];
    }

    // Render the list of properties and other content that is expanded when the
    // expand button is clicked.
    const listItems: ElementList = [];
    // TODO: the tensor is currently formatted by Tensor.prototype.toString()
    // and displayed in a monospace font. Make it nicer by using a table.
    if (d.type === "tensor") {
      listItems.push(
        <li>
          <div class="inspector__content inspector__content--pre">
            { d.formatted }
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
    const list =
      (listItems.length > 0)
        ? <ul class="inspector__list inspector__prop-list">
            { ...listItems }
          </ul>
        : undefined;

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
        const ctorVisible = d.ctor === "Array" ? "expanded" : "always";
        elements.push(
          ...showWhen(ctorVisible,
                      <span class="cm-def">{ d.ctor }</span>,
                      "(", <span class="cm-number">{ d.length }</span>, ") "),
          "[", list || "", "]"        );
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
          ...this.renderPropList(d, list)
        );
        break;
      case "function":
        // Constructor name is only shown if it is nontrivial.
        //   function() {…} {...properties...}
        //   class Foo {…}
        //   async function* bar() {…}
        //   MyFunction:function baz() {…}
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
          d.class ? " {…}" : "() {…}",
          ...this.renderPropList(d, list)
        );
        break;
      case "tensor":
        // Tensor(float32 2✕2) [[1, 2], [3, 4]]
        // Tensor(uint32 3✕4✕4) { [[[tensor]]]], ...properties... }
        elements.push(
          <span class="cm-def">{ d.ctor }</span>,
          "(",
          <span class="cm-string-2">{ d.dtype }</span>
        );
        for (const [dim, size] of d.shape.entries()) {
          elements.push(
            dim === 0 ? " " : "✕",
            <span class="cm-number">{ size }</span>
          );
        }
        elements.push(
          ")",
          ...showWhen("collapsed", " "),
          ...this.renderPropList(d, list)
        );
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
        elements.push("{", list || "", "}");
        break;
    }

    return elements;
  }

  private renderPropList(d: BaseObjectDescriptor,
                         list?: ElementLike): ElementList {
    if (!list) return [];
    // If the list contains items that are visible in collapsed mode, always
    // show the curly braces. Otherwise show them only when expanded.
    const hasVisibleProps = d.props.some(prop => !prop.hidden);
    const bracesVisible = hasVisibleProps ? "always" : "expanded";
    return [
       ...showWhen(bracesVisible, " {"),
       list,
       ...showWhen(bracesVisible, "}"),
    ];
  }

  private hasChildren(d: ValueDescriptor): boolean {
    // Returns true if a descriptor has properties that warrant it getting an
    // expand/collapse button.
    return d.type === "tensor" ||
           (d.type === "array" && d.props.length > 0) ||
           (d.type === "box" && d.props.length > 0) ||
           (d.type === "function" && d.props.length > 0) ||
           (d.type === "object" && d.props.length > 0);
  }

  private renderExpandButton(): ElementLike {
    const id = randomString();
    return <a id={ id } class="inspector__toggle"
              href={ `javascript:Inspector.toggle("${id}")` } />;
  }

  private renderProperty(d: PropertyDescriptor,
                         arrayIndex: number | null = null): ElementLike {
    const hasChildren = this.hasChildren(d.value);
    const attr = d.hidden ? { class: "inspector__li--hidden" } : {};
    return (
      <li { ...attr }>
       { hasChildren && this.renderExpandButton() }
       <div class="inspector__content">
          { ...this.renderKey(d.key, arrayIndex) }
          { ...this.renderValue(d.value) }
        </div>
      </li>
    );
  }

  private renderRoot(descriptors: ValueDescriptor[]): ElementLike {
    if (descriptors.length === 0) return "";
    // When inspecting a only non-expandable values, simply lay them out
    // next to one another in a single div.
    if (descriptors.every(d => !this.hasChildren(d))) {
      const elements = [].concat(...descriptors.map(
        (d, index) => [
          index > 0 ? " " : "", // Space between items.
          ...this.renderValue(d)
        ]));
      return <div class="inspector__content">{ ...elements }</div>;
    }
    // Since there is at least something expandable at the top level, make <li>
    // elements for all items.
    let rootItems = descriptors.map(d =>
      <li>
        { this.hasChildren(d) && this.renderExpandButton() }
        <div class="inspector__content">
          { ...this.renderValue(d) }
        </div>
      </li>
    );
    // If there is more than one descriptor, wrap the list in another list
    // and add an expand button that allows the user to display all descriptors
    // on a separate line.
    if (descriptors.length > 1) {
      rootItems = [
        <li>
          { this.renderExpandButton() }
          <div class="inspector__content">
            { ...showWhen("expanded", "(") }
            <ul class="inspector__list inspector__arg-list">
              { ...rootItems }
            </ul>
            { ...showWhen("expanded", ")") }
          </div>
        </li>
      ];
    }
    // Create the tree-root list of one element.
    return (
      <ul class="inspector__list inspector__tree-root">
        { ...rootItems }
      </ul>
    );
  }

  render(): JSX.Element {
    const d = this.props.descriptors;
    return <div class="inspector cm-s-syntax">{ this.renderRoot(d) }</div>;
  }

  static toggle(id: string): void {
    const el = document.getElementById(id);
    const parent = el.parentNode as HTMLElement;
    parent.classList.toggle("inspector__li--expanded");
  }
}

// Helper function to attach a css class to a list of elements, which makes
// them visible/hidden depending on tree expanded/collapsed state.
// The element list may contain strings; they are converted to spans.
function showWhen(when: Visibility, ...elements: ElementList): ElementList {
  if (when === "always") {
    return elements; // No transformation needed.
  }
  const visClass = `inspector--show-if-${when}`;
  const result: ElementList = [];
  let stringBuffer = "";
  for (const el of elements) {
    if (typeof el === "string") {
      // Merge adjacent strings so we don't create so many spans.
      stringBuffer += el;
    } else {
      if (stringBuffer) {
        // Wrap the buffered-up string in a <span> element and flush.
        result.push(<span class={ visClass }>{ stringBuffer }</span>);
        stringBuffer = "";
      }
      // Add our css class to the existing element.
      const elClass = el.attributes.class;
      el.attributes.class = elClass ? `${elClass} ${visClass}` : visClass;
      result.push(el);
    }
  }
  if (stringBuffer) {
    result.push(<span class={ visClass }>{ stringBuffer }</span>);
  }
  return result;
}

if (IS_WEB) {
  // Make Inspector available on the global object. This allows expand button
  // to use a simple textual click handler, which survives serialization.
  // We make the property non-enumerable to minimize interaction with other
  // components.
  Object.defineProperty(window, "Inspector", {
    enumerable: false,
    value: Inspector
  });
}
