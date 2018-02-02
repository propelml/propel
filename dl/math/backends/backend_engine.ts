/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as util from '../../util';
import {DataType, NDArray} from '../ndarray';

import {MathBackend} from './backend';
import * as kernel_registry from './kernel_registry';
import {KernelConfigRegistry} from './kernel_registry';
import * as tape_util from './tape_util';
import {ScopeResult, ScopeResultImmediate} from './tape_util';

export class BackendEngine {
  private activeScope: NDArray[];
  private scopeStack: NDArray[][];

  private debugMode = false;

  constructor(private backend: MathBackend, private safeMode: boolean) {
    // Create a default outer scope.
    this.activeScope = [];
    this.scopeStack = [this.activeScope];
  }

  enableDebugMode() {
    this.debugMode = true;
  }

  executeKernel<K extends keyof KernelConfigRegistry,
                          C extends KernelConfigRegistry[K]['inputAndArgs']>(
      kernelName: K, config: C):
      KernelConfigRegistry[K]['output'] {
    const kernelFn = () =>
        kernel_registry.executeKernel(this.backend, kernelName, config);

    let start: number;
    if (this.debugMode) {
      start = performance.now();
    }
    const result = kernelFn();
    if (this.debugMode) {
      const vals = result.dataSync();
      const time = util.rightPad(`${performance.now() - start}ms`, 9);
      const paddedName = util.rightPad(kernelName, 25);
      const rank = result.rank;
      const size = result.size;
      const shape = util.rightPad(result.shape.toString(), 14);
      console.log(
          `%c${paddedName}\t%c${time}\t%c${rank}D ${shape}\t%c${size}`,
          'font-weight:bold', 'color:red', 'color:blue', 'color: orange');
      util.checkForNaN(vals, result.dtype, name);
    }

    return result;
  }

  /**
   * Create a new math scope. Put chained math operations inside a scope
   * function closure so that the library automatically cleans up NDArrays
   * from intermediate math operations. You must create a scope in safe mode
   * to call math operations. If a result is returned from the scope, it will
   * also be tracked, which means there must be yet another wrapping scope.
   * @param name The name of the scope. Used for logging.
   * @param scopeFn The function to execute with chained math operations.
   */
  scope<T extends ScopeResult>(name: string, scopeFn: () => T): T {
    this.startScope();

    const result = scopeFn();

    if (result instanceof Promise) {
      result.then(r => this.endScope(r));
      return result;
    } else {
      this.endScope(result as ScopeResultImmediate);
      return result;
    }
  }

  /**
   * Start a scope. Use this with endScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  startScope() {
    const newScopeArrays: NDArray[] = [];
    this.scopeStack.push(newScopeArrays);
    this.activeScope = newScopeArrays;
  }

  /**
   * End a scope. Use this with startScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  endScope(result: ScopeResultImmediate) {
    const arraysToTrackInParent =
      tape_util.extractNDArraysFromScopeResult(result);

    // Dispose the arrays tracked in this scope.
    for (let i = 0; i < this.activeScope.length; i++) {
      const ndarray = this.activeScope[i];
      if (util.isNDArrayInList(ndarray, arraysToTrackInParent)) {
        continue;
      }

      ndarray.dispose();
    }

    this.scopeStack.pop();
    this.activeScope = this.scopeStack.length === 0 ?
        null :
        this.scopeStack[this.scopeStack.length - 1];

    // Track the current result in the parent scope.
    arraysToTrackInParent.forEach(ndarray => {
      this.track(ndarray);
    });
  }

  /**
   * Tracks an NDArray in the current scope to be automatically cleaned up
   * when the current scope ends, and returns the value.
   *
   * @param result The NDArray to track in the current scope.
   */
  track<D extends DataType, T extends NDArray<D>>(result: T): T {
    if (this.scopeStack.length === 1) {
      if (this.safeMode) {
        throw new Error(
            'You are using math in safe mode. Enclose all ' +
            'math.method() calls inside a scope: ' +
            'math.scope(() => {math.method();...}) to avoid memory ' +
            'leaks.');
      }
    } else {
      // Only track NDArrays that are created in an explicit scope; when
      // created in the global scope, tracking them makes no sense -- the
      // global scope never gets cleaned up, and adding them to an array
      // just prevents the NDArray from ever being garbage collected.
      this.activeScope.push(result);
    }
    return result;
  }

  getBackend(): MathBackend {
    return this.backend;
  }
}
