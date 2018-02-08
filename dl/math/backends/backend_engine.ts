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

import * as util from "../../util";
import { DataType, NDArray } from "../ndarray";

export type ScopeResult =
    void | NDArray | NDArray[] | {[key: string]: NDArray};

export function extractNDArraysFromScopeResult(result: ScopeResult):
    NDArray[] {
  if (result == null) {
    return [];
  }
  if (result instanceof NDArray) {
    return [result];
  }

  const list: NDArray[] = [];
  const resultObj = result as {[key: string]: NDArray};
  // Iteration over keys works also for arrays.
  for (const k in resultObj) {
    if (!resultObj.hasOwnProperty(k)) continue;
    const val = resultObj[k];
    if (val instanceof NDArray) {
      list.push(val);
    }
  }
  return list;
}

export class BackendEngine {
  private activeScope: NDArray[];
  private scopeStack: NDArray[][];

  constructor() {
    // Create a default outer scope.
    this.activeScope = [];
    this.scopeStack = [this.activeScope];
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
    this.endScope(result);
    return result;
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
  endScope(result: ScopeResult) {
    const arraysToTrackInParent = extractNDArraysFromScopeResult(result);

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
    if (this.scopeStack.length > 1) {
      // Only track NDArrays that are created in an explicit scope; when
      // created in the global scope, tracking them makes no sense -- the
      // global scope never gets cleaned up, and adding them to an array
      // just prevents the NDArray from ever being garbage collected.
      this.activeScope.push(result);
    }
    return result;
  }
}
