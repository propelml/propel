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
// Preact CodeMirror wrapper.
// tslint:disable-next-line:no-reference
/// <reference path="../node_modules/@types/codemirror/index.d.ts" />
import { Component, h } from "preact";
import { assert, IS_WEB } from "../src/util";
import { normalizeCode } from "./common";

interface CMProps {
  code: string;
  onCtrlEnter: () => void | Promise<void>;
  onShiftEnter: () => void | Promise<void>;
  onFocusChange: (focused: boolean) => void | Promise<void>;
}

interface CMState { }

export class CMComponent extends Component<CMProps, CMState> {
  editor: CodeMirror.Editor;
  pre: HTMLPreElement;

  // This render is only called the first time the component is loaded
  // see shouldComponentUpdate.
  render() {
    // This pre is replaced by CodeMirror if the browser has JavaScript enabled.
    return h("div", { "class": "cm-parent" },
      h("pre", {
        "class": "cm-pre",
        "ref": ((ref) => { this.pre = ref as HTMLPreElement; }),
      }, this.code));
  }

  // Because CodeMirror has complex internal state, we avoid react updates.
  shouldComponentUpdate(nextProps, nextState) {
    return false;
  }

  // Because CodeMirror has a lot of state that is not managed through
  // React, manually apply prop changes.
  componentWillReceiveProps(nextProps) {
    const nextCode = normalizeCode(nextProps.code);
    if (nextCode !== this.code) {
      this.editor.setValue(nextCode);
    }
  }

  get code(): string {
    return normalizeCode(this.editor ? this.editor.getValue()
                                     : this.props.code);
  }

  onFocusChange(focused: boolean) {
    if (this.props.onFocusChange) this.props.onFocusChange(focused);
  }

  componentDidMount() {
    const defaults = {
      lineNumbers: false,
      lineWrapping: true,
      mode: "javascript",
      scrollbarStyle: null,
      theme: "syntax",
      viewportMargin: Infinity,
    };
    const options = Object.assign(defaults, { value: this.code });
    assert(IS_WEB); // The following doesn't work in Node.

    // tslint:disable-next-line:variable-name
    const CodeMirror = require("codemirror");
    require("codemirror/mode/javascript/javascript.js");

    const parentEl = this.pre.parentElement;
    parentEl.removeChild(this.pre); // Delete existing pre.

    this.editor = CodeMirror(parentEl, options);
    this.editor.setOption("extraKeys", {
      "Ctrl-Enter": () => {
        if (this.props.onCtrlEnter) this.props.onCtrlEnter();
        return true;
      },
      "Shift-Enter": () => {
        if (this.props.onShiftEnter) this.props.onShiftEnter();
        return true;
      },
    });
    this.editor.on("focus", () => this.onFocusChange(true));
    this.editor.on("blur", () => this.onFocusChange(false));
  }

  blur() {
    this.editor.getInputField().blur();
    this.onFocusChange(false);
  }

  focus() {
    this.editor.getInputField().focus();
    this.onFocusChange(true);
  }

  setValue(code: string): void {
    this.editor.setValue(code);
  }
}
