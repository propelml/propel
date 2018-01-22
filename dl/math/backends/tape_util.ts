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

import {NDArray} from '../ndarray';

export type ScopeResultImmediate =
    void|NDArray|NDArray[]|{[key: string]: NDArray};
export type ScopeResult = ScopeResultImmediate|Promise<ScopeResultImmediate>;

export function extractNDArraysFromScopeResult(result: ScopeResultImmediate):
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
    const val = resultObj[k];
    if (val instanceof NDArray) {
      list.push(val);
    }
  }
  return list;
}
