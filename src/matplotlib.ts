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

import { Tensor } from "./api";
import { toUint8Image } from "./im";
import { PlotData } from "./output_handler";
import { assertEqual } from "./tensor_util";
import { getOutputHandler } from "./util";

export function plot(...args) {
  if (!getOutputHandler()) {
    console.warn("plot: no output handler");
    return;
  }

  const xs = [];
  const ys = [];
  let state = "x";
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (state) {
      case "x":
        xs.push(arg);
        state = "y";
        break;

      case "y":
        ys.push(arg);
        state = "x";
        break;
    }
  }

  assertEqual(xs.length, ys.length);
  const data: PlotData = [];
  for (let i = 0; i < xs.length; ++i) {
    // TODO line = $.stack([xs[i], ys[i]], 1)
    const xv = xs[i].dataSync();
    const yv = ys[i].dataSync();
    assertEqual(xv.length, yv.length);
    const line = [];
    for (let j = 0; j < xv.length; ++j) {
      line.push({ x: xv[j], y: yv[j] });
    }
    data.push(line);
  }

  getOutputHandler().plot(data);
}

/** Displays a tensor as an image.
 *
 *    import { randn, imshow } from "propel"
 *    img = randn([300, 300]).relu().mul(255);
 *    imshow(img)
 */
export function imshow(tensor: Tensor): void {
  if (!getOutputHandler()) {
    console.warn("imshow: no output handler");
    return;
  }
  const image = toUint8Image(tensor);
  getOutputHandler().imshow(image);
}
