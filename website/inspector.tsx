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

export interface InspectorProps {
  object: any;
  name?: string;
  depth?: number;
  inline?: boolean;
}

export interface InspectorState {
  isCollapsed: boolean;
  isCollapsible: boolean;
}

export class Inspector extends Component<InspectorProps, InspectorState>{
  inlineRendered;
  constructor(props) {
    super();
    const isCollapsible = this.isCollapsible(props);
    this.state = {
      isCollapsed: this.shouldExpandByDefault(props, isCollapsible),
      isCollapsible
    };
  }

  shouldExpandByDefault({ depth = 0, object }, isCollapsible) {
    if (!isCollapsible) return false;
    if (object.isCircular) return false;
    if (depth === 0) return true;
    return Object.getOwnPropertyNames(object.data).length <= 15;
  }

  toggle = () => {
    this.setState(s => ({
      isCollapsed: !s.isCollapsed
    }));
  }

  renderChild() {
    const { object: { data, type }, depth = 0 } = this.props;
    const properties = [];
    switch (type){
      case "object":
      case "array":
        for (const key in data) {
          if (data.hasOwnProperty(key)) {
            properties.push({
              name: key,
              object: data[key]
            });
          }
        }
        break;
      case "tensor":
        properties.push({
          object: {
            data: data.data,
            type: "tensorValue"
          }
        });
        break;
    }
    const nextDepth = depth + 1;
    return (
      <div class="tree-group">
        {properties.map(({ name, object }) => (
          <Inspector depth={ nextDepth } name={ name } object={ object } />
        ))}
      </div>
    );
  }

  isCollapsible(props) {
    const { object: { data, type } } = props;
    switch (type) {
      case "array":
      case "object":
        let keys = Object.getOwnPropertyNames(data);
        if (type === "array") {
          keys = keys.filter(x => x !== "length");
        }
        this.inlineRendered = this.renderInlinePreview(data, type);
        return this.inlineRendered.length < keys.length;
      case "tensor":
        return true;
    }
    return false;
  }

  renderInlinePreview(data, type) {
    let keys = Object.getOwnPropertyNames(data);
    let elements = null;
    if (type === "array") {
      keys = keys.filter(x => x !== "length");
    }
    const maxElements = type === "array" ? 20 : 10;
    const num = Math.min(Math.max(keys.length * 2 - 1, 0), maxElements);
    elements = new Array(num)
      .fill(null).map((x, i) => {
        if (i === maxElements - 1) return <span>,...</span>;
        if (i % 2) return <span>, </span>;
        return (
          <Inspector
            object={ data[keys[i / 2]] }
            inline={ true }
            name= { type === "object" && keys[i / 2] } />
        );
      });
    return elements;
  }

  render() {
    const { object: { data, type, cons, isCircular }, name } = this.props;
    const { depth = 0, inline = false } = this.props;
    const { isCollapsed, isCollapsible } = this.state;
    let value = null;
    if (inline && isCircular) {
      value = (
        <span class="cm-keyword">[Circular]</span>
      );
    } else {
      switch (type) {
        case "number":
          value = (
            <span class="cm-number">
              { data }
            </span>
          );
          break;
        case "string":
          value = (
            <span class="cm-string">
              { JSON.stringify(data) }
            </span>
          );
          break;
        case "symbol":
          value = (
            <span class="cm-atom">
              { data }
            </span>
          );
          break;
        case "function":
          value = (
            <span>
              <span class="cm-keyword">function</span>&nbsp;
              <span class="cm-def">
                { data.name }
              </span>
              <span>
              { data.isNative ? "() { [native code] }" : "() {...}" }
              </span>
            </span>
          );
          break;
        case "undefined":
          value = <span class="cm-atom">undefined</span>;
          break;
        case "null":
          value = <span class="cm-atom">null</span>;
          break;
        case "object":
          value = (
            <span>
              <span class="cm-variable">
              { inline && cons === "Object" ? "" : cons + " "}
              </span>
              &#123;
                { ...this.inlineRendered }
              &#125;
            </span>
          );
          break;
        case "array":
          value = (
            <span>
              <span class="cm-variable">
                { inline ? "" : "Array " }
              </span>
              [
                { ...this.inlineRendered }
              ]
            </span>
          );
          break;
        case "date":
          value = (
            <span class="cm-atom">
              { data }
            </span>
          );
          break;
        case "regexp":
          value = (
            <span class="cm-string-2">
              { data }
            </span>
          );
          break;
        case "boolean":
          value = (
            <span class="cm-atom">
              { data ? "true" : "false" }
            </span>
          );
          break;
        case "promise":
          value = (
            <span class="cm-atom">
              Promise
            </span>
          );
          break;
        case "tensor":
          value = (
            <span>
              <span class="cm-variable">
                Tensor(dtype="{data.dtype}", shape=[{data.shape.join(", ")}])
              </span>
            </span>
          );
          break;
        case "tensorValue":
          value = (
            <pre>
              { data }
            </pre>
          );
          break;
        default:
          return null;
      }
    }
    const isCollapsedClass = isCollapsed ? " collapsed" : "";
    const isCollapsibleClass = isCollapsible ? " collapsible" : "";
    const inlineClass = inline ? " inline" : "";
    const depthClass = " depth-" + depth;
    const className = "tree-node" + isCollapsedClass + inlineClass + depthClass;
    return (
      <div class={ className } >
        <div
          class={ "tree-header cm-s-syntax" + isCollapsibleClass }
          onClick={ !inline && isCollapsible && this.toggle } >
          { depth === 0 && !isCollapsible ? null : (
            <div class="tree-arrow-wrapper">
              { !isCollapsible ? null : (
                <div class="tree-arrow" />
              ) }
            </div>
          ) }
          { name && <span class="cm-property">{ name }</span> }
          { name && ": " }
          { value }
        </div>
        { !inline && isCollapsed && this.renderChild() }
      </div>
    );
  }
}
