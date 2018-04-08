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

// This file is intentionally decoupled from matplotlib.ts, so the dom output
// handler can be loaded without pulling in the entire math library.

// FIXME
// tslint:disable:object-literal-sort-keys

import { h , render } from "preact";
import * as vega from "vega-lib";
import { Inspector } from "../website/inspector";
import { ValueDescriptor } from "../website/serializer";
import { createCanvas, Image } from "./im";

export type PlotData = Array<Array<{ x: number, y: number }>>;

export type VegaConfig = vega.Config;
export type VegaSpec = vega.Spec;

export interface Progress {
  job: string;
  loaded: number | null;
  total: number | null;
}

export interface OutputHandler {
  imshow(image: Image): void;
  plot(data: PlotData): void;
  print(text: object[] | string): void;
  downloadProgress(progress: Progress);
}

const progressOutputHandlerMap = new Map<string, OutputHandler>();

export class OutputHandlerDOM implements OutputHandler {
  // TODO colors should match those used by the syntax highlighting.
  private progressJobs = new Map<string, Progress>();

  constructor(private element: Element) {}

  imshow(image: Image): void {
    const canvas = createCanvas(image);
    this.element.appendChild(canvas);
  }

  vega(spec: VegaSpec, config: VegaConfig = {}) {
    const runtime = vega.parse(spec, config);
    const view = new vega.View(runtime, {
      loader: vega.loader(),
      logLevel: vega.Warn,
      renderer: "svg"
    }).initialize(this.element);
    view.run();
  }

  plot(data: PlotData): void {
    const d = data.map((line, i) => {
      return line.map(({ x, y }) => {
        const obj = { x, y, c: i };
        return obj;
      });
    });
    let vegaData = [];
    for (const line of d) {
      vegaData = vegaData.concat(line);
    }

    this.vega({
      $schema: "https://vega.github.io/schema/vega/v3.json",
      width: 440,
      height: 300,

      data: [{
        name: "table",
        values: vegaData,
      }],

      scales: [
        {
          name: "x",
          type: "linear",
          domain: {"data": "table", "field": "x"},
          range: "width",
        },
        {
          name: "y",
          type: "linear",
          domain: {"data": "table", "field": "y"},
          range: "height",
          nice: true,
          zero: true,
        },
        {
          name: "color",
          type: "ordinal",
          domain: {"data": "table", "field": "c"},
          range: "category",
        }
      ],

      axes: [
        {"orient": "bottom", "scale": "x"},
        {"orient": "left", "scale": "y"}
      ],

      marks: [
        {
          "type": "group",
          "from": {
            "facet": {
              "name": "series",
              "data": "table",
              "groupby": "c"
            }
          },
          "marks": [
            {
              "type": "line",
              "from": {"data": "series"},
              "encode": {
                "enter": {
                  "x": {"scale": "x", "field": "x"},
                  "y": {"scale": "y", "field": "y"},
                  "stroke": {"scale": "color", "field": "c"},
                  "strokeWidth": {"value": 2}
                },
              }
            }
          ]
        }
      ]
    });
  }

  print(data: ValueDescriptor[] | string): void {
    if (typeof data === "string") {
      const element = this.element;
      const last = element.lastChild;
      let s = (last && last.nodeType !== Node.TEXT_NODE) ? "\n" : "";
      s += data + "\n";
      const el = document.createTextNode(s);
      element.appendChild(el);
      return;
    }
    try {
      const elem = h(Inspector, { descriptors: data });
      render(elem, this.element);
    } catch (e) {}
  }

  downloadProgress(progress: Progress) {
    const job = progress.job;

    // Ensure that the progress handler jobs are always handled by the same
    // output handler, even when the notebook guesses the cell wrong.
    // TODO: this is really hacky and should not be the responsibility of the
    // output handler at all.
    const outputHandler = progressOutputHandlerMap.get(job);
    if (outputHandler === undefined) {
      progressOutputHandlerMap.set(job, this);
    } else if (outputHandler !== this) {
      return outputHandler.downloadProgress(progress);
    }

    if (progress.loaded === null) {
      // When loaded equals null, this indicates that the job is done.
      // TODO: this isn't really correct - the progress bar might go backwards
      // when multiple parallel jobs are present and one completes before
      // the other.
      this.progressJobs.delete(job);
      progressOutputHandlerMap.delete(job);
    } else {
      this.progressJobs.set(job, progress);
    }

    const outputContainer = this.element.parentNode as HTMLElement;
    const progressBar = outputContainer.previousElementSibling as HTMLElement;

    if (this.progressJobs.size === 0) {
      // If there are no more downloads in progress, hide the progress bar.
      progressBar.style.width = "0"; // Prevent it from going backwards.
      progressBar.style.display = "none";
      return;
    }

    let sumLoaded = 0;
    let sumTotal = 0;
    for (const [_, {loaded, total}] of this.progressJobs) {
      // Total may be null if the size of the download isn't known yet.
      if (total === null) {
        continue;
      }
      sumLoaded += loaded;
      sumTotal += total;
    }

    // Avoid division by zero.
    const percent = sumTotal > 0 ? sumLoaded / sumTotal * 100 : 0;
    progressBar.style.display = "block";
    progressBar.style.width = `${percent}%`;
  }
}
