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
import * as d3 from "d3";
import { Tensor } from "./api";
import { assertEqual } from "./util";

export interface OutputHandler {
  (): Element;
}

let h: OutputHandler;

/** This function allows different systems to register the HTML divs that will
 * produce output. It is used inside of notebook.ts but can be used by
 * stand-alone pages for plotting.  Note currently there is only one global
 * handler. It gets replaced it register is called twice.
 */
export function register(handler: OutputHandler): void {
  if (h == null) {
    console.warn("Warning replacing existing OutputHandler.");
  }
  h = handler;
}

export function outputEl(): null | Element {
  return h ? h() : null;
}

// TODO colors should match those used by the syntax highlighting.
const color = d3.scaleOrdinal(d3.schemeCategory10);

function makeAxis(svg, margin, xScale, yScale, width, height) {
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

function getLimits(lines): number[] {
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

function plotLines(data) {
  const el = outputEl();
  // Don't try to plot if there's not output registered.
  if (!el) return;
  const outputId_ = "#" + el.id;

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

  const [xMin, xMax, yMin, yMax] = getLimits(data);

  // A small inner margin prevents the plot lines from touching the axes.
  const xMargin = (xMax - xMin) * 0.02;
  const yMargin = (yMax - yMin) * 0.02;

  const xScale = d3.scaleLinear()
    .domain([xMin - xMargin, xMax + xMargin])
    .range([0, width]);

  const yScale = d3.scaleLinear()
    .domain([yMin - yMargin, yMax + yMargin])
    .range([height, 0]);

  makeAxis(svg, margin, xScale, yScale, width, height);

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
      return color(i as any);
    });
}

export function plot(...args) {
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
  const data = [];
  for (let i = 0; i < xs.length; ++i) {
    // TODO line = $.stack([xs[i], ys[i]], 1)
    const xv = xs[i].getData();
    const yv = ys[i].getData();
    assertEqual(xv.length, yv.length);
    const line = [];
    for (let j = 0; j < xv.length; ++j) {
      line.push({ x: xv[j], y: yv[j] });
    }
    data.push(line);
  }

  plotLines(data);
}

export function imshow(image: Tensor): void {
  const output = outputEl();
  if (!output) return;
  const canvas = document.createElement("canvas");
  // Assuming image shape is [3, height, width] for RGB.
  // [height, width] for monochrome.
  assertEqual(image.shape.length, 2, "Assuming monochrome for now");
  const tensorData = image.getData();
  const h = canvas.height = image.shape[0];
  const w = canvas.width = image.shape[1];
  const ctx = canvas.getContext("2d");
  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data;
  for (let y = 0; y < h; ++y) {
    for (let x = 0; x < w; ++x) {
      let index = (y * w + x) * 4;
      // TODO image.get(y, x);
      const value = tensorData[y * w + x];
      data[index]   = value; // red
      data[++index] = value; // green
      data[++index] = value; // blue
      data[++index] = 255;   // alpha
    }
  }
  ctx.putImageData(imageData, 0, 0);
  output.appendChild(canvas);
}
