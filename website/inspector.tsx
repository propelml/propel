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
  constructor(props) {
    super();
    const { depth } = props;
    const isCollapsible = this.isCollapsible(props);
    this.state = {
      isCollapsed: isCollapsible && !depth,
      isCollapsible
    };
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

  isTypeCollapsible(type) {
    return type === "tensor" || type === "object" || type === "array";
  }

  isCollapsible(props) {
    const { object: { data, type } } = props;
    switch (type) {
      case "array":
        const keys = Object.getOwnPropertyNames(data)
                     .filter(x => x !== "length");
        let isInline = true;
        let length = 0;
        for (let i = 0; i < keys.length; ++i) {
          if (!isInline) break;
          const t = data[keys[i]].type;
          if (this.isTypeCollapsible(t)) isInline = false;
          length += JSON.stringify(data[keys[i]].data).length + 2;
        }
        isInline = isInline && length < 100;
        return !isInline;
      case "object":
        const isEmpty = Object.getOwnPropertyNames(data).length === 0;
        return !isEmpty;
      case "tensor":
        return true;
    }
    return false;
  }

  render() {
    const { object: { data, type, cons }, name, depth = 0 } = this.props;
    const { inline = false } = this.props;
    const { isCollapsed, isCollapsible } = this.state;
    let value = null;
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
        // TODO inline preview for objects.
        value = (
          <span class="cm-variable">
            { cons }
            { isCollapsible ? "" : " {}"}
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
      case "array":
        if (isCollapsible) {
          value = (
            <span>
              <span class="cm-variable">
                Array
              </span>
              (
                <span class="cm-number">
                  { data.length.data }
                </span>
              )
            </span>
          );
        } else {
          const keys = Object.getOwnPropertyNames(data)
                       .filter(x => x !== "length");
          const elements = new Array(Math.max(keys.length * 2 - 1, 0))
            .fill(null).map((x, i) => {
              if (i % 2) return <span>,</span>;
              return <Inspector object={ data[keys[i / 2]] } inline={ true } />;
            });
          value = (
            <span>
              <span class="cm-variable">
                Array
              </span>
              [
                { ...elements }
              ]
            </span>
          );
        }
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
    const isCollapsedClass = isCollapsed ? " collapsed" : "";
    const isCollapsibleClass = isCollapsible ? " collapsible" : "";
    const inlineClass = inline ? " inline" : "";
    const depthClass = " depth-" + depth;
    const className = "tree-node" + isCollapsedClass + inlineClass + depthClass;
    return (
      <div class={ className } >
        <div
          class={ "tree-header cm-s-syntax" + isCollapsibleClass }
          onClick={ isCollapsible && this.toggle } >
          <div class="tree-arrow-wrapper">
            { !isCollapsible ? null : (
              <div class="tree-arrow" />
            )}
          </div>
          { name && <span class="cm-property">{ name }</span> }
          { name && ": " }
          { value }
        </div>
        { isCollapsed && this.renderChild() }
      </div>
    );
  }
}
