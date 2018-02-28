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

export type PlotData = Array<Array<{ x: number, y: number }>>;
export type ImshowData = { height: number, width: number, values: number[] };

export interface OutputHandler {
  plot(data: PlotData): void;
  imshow(data: ImshowData): void;
}

export class OutputHandlerDOM implements OutputHandler {
  // TODO colors should match those used by the syntax highlighting.
  private color = d3.scaleOrdinal(d3.schemeCategory10);

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

  imshow({ width, height, values }: ImshowData): void {
    const canvas = document.createElement("canvas");
    canvas.height = height;
    canvas.width = width;
    const ctx = canvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        let index = (y * width + x) * 4;
        const value = values[y * width + x];
        data[index]   = value; // red
        data[++index] = value; // green
        data[++index] = value; // blue
        data[++index] = 255;   // alpha
      }
    }
    ctx.putImageData(imageData, 0, 0);
    this.element.appendChild(canvas);
  }
}
