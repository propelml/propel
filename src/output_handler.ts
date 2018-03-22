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

import * as d3 from "d3";
import { createCanvas, Image } from "./im";

export type PlotData = Array<Array<{ x: number, y: number }>>;

export interface Progress {
  job: string;
  loaded: number | null;
  total: number | null;
}

export interface OutputHandler {
  imshow(image: Image): void;
  plot(data: PlotData): void;
  print(text: string): void;
  downloadProgress(progress: Progress);
}

const progressOutputHandlerMap = new Map<string, OutputHandler>();

export class OutputHandlerDOM implements OutputHandler {
  // TODO colors should match those used by the syntax highlighting.
  private color = d3.scaleOrdinal(d3.schemeCategory10);
  private progresses = new Map<string, Progress>();

  constructor(private element: Element) {}

  private makeAxis(svg, margin, xScale, yScale, width, height) {
    const axisBottom = d3.axisBottom(xScale);
    axisBottom.tickSizeOuter(0);
    svg.append("g")
      .attr("transform", `translate(${margin.left},${height + margin.top})`)
      .call(axisBottom);

    const axisLeft = d3.axisLeft(yScale);
    axisLeft.tickSizeOuter(0);
    svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`)
      .call(axisLeft);

    const axisRight = d3.axisRight(yScale);
    axisRight.ticks(0);
    axisRight.tickSizeOuter(0);
    svg.append("g")
      .attr("transform", `translate(${width + margin.left},${margin.top})`)
      .call(axisRight);

    const axisTop = d3.axisTop(xScale);
    axisTop.ticks(0);
    axisTop.tickSizeOuter(0);
    svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`)
      .call(axisTop);
  }

  private getLimits(lines): number[] {
    // TODO Replace this with real ops on tensors.
    let xMin, xMax, yMin, yMax;
    for (const line of lines) {
      for (const point of line) {
        if (xMin === undefined || point.x < xMin) {
          xMin = point.x;
        }

        if (yMin === undefined || point.y < yMin) {
          yMin = point.y;
        }

        if (xMax === undefined || point.x > xMax) {
          xMax = point.x;
        }

        if (yMax === undefined || point.y > yMax) {
          yMax = point.y;
        }
      }
    }
    return [xMin, xMax, yMin, yMax];
  }

  imshow(image: Image): void {
    const canvas = createCanvas(image);
    this.element.appendChild(canvas);
  }

  plot(data: PlotData): void {
    const outputId_ = "#" + this.element.id;

    // Make an SVG Container
    let width = 440;
    let height = 300;

    // No append.
    d3.select(outputId_).select("svg").remove();

    const svg = d3.select(outputId_).append("svg")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMinYMin meet")
      .attr("width", width)
      .attr("height", height);
    const m = 30;
    const margin = { top: m, right: m, bottom: m, left: m };
    width = +svg.attr("width") - margin.left - margin.right;
    height = +svg.attr("height") - margin.top - margin.bottom;
    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const [xMin, xMax, yMin, yMax] = this.getLimits(data);

    // A small inner margin prevents the plot lines from touching the axes.
    const xMargin = (xMax - xMin) * 0.02;
    const yMargin = (yMax - yMin) * 0.02;

    const xScale = d3.scaleLinear()
      .domain([xMin - xMargin, xMax + xMargin])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([yMin - yMargin, yMax + yMargin])
      .range([height, 0]);

    this.makeAxis(svg, margin, xScale, yScale, width, height);

    const line = (d3.line() as any)
      .x(d => xScale(d.x))
      .y(d => yScale(d.y));

    g.selectAll("path").data(data)
      .enter()
      .append("path")
      .attr("d", line as any)
      .style("fill", "none")
      .style("stroke-width", "2px")
      .style("stroke", (d, i) => {
        return this.color(i as any);
      });
  }

  print(text: string): void {
    const element = this.element;
    const last = element.lastChild;
    let s = (last && last.nodeType !== Node.TEXT_NODE) ? "\n" : "";
    s += text + "\n";
    const el = document.createTextNode(s);
    element.appendChild(el);
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
      this.progresses.delete(job);
      progressOutputHandlerMap.delete(job);
    } else {
      this.progresses.set(job, progress);
    }

    const outputContainer = this.element.parentNode as HTMLElement;
    const progressBar = outputContainer.previousElementSibling as HTMLElement;

    if (this.progresses.size === 0) {
      // If there are no more downloads in progress, hide the progress bar.
      progressBar.style.width = "0"; // Prevent it from going backwards.
      progressBar.style.display = "none";
      return;
    }

    let sumLoaded = 0;
    let sumTotal = 0;
    for (const [_, {loaded, total}] of this.progresses) {
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
