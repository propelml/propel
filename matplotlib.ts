import * as d3 from "d3";
import { $ } from "./api";
import * as notebook from "./notebook";
import { assertEqual } from "./util";

const currentPlot = null;
// TODO colors should match those used by the syntax highlighting.
const color = d3.scaleOrdinal(d3.schemeCategory10);

function makeAxis(svg, margin, xScale, yScale, width, height) {
  const axisBottom = d3.axisBottom(xScale);
  axisBottom.tickSizeOuter(0);
  svg.append("g")
    .attr("transform", `translate(${margin.left},${height+margin.top})`)
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
    .attr("transform", `translate(${width+margin.left},${margin.top})`)
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
  const outputId = notebook.outputId();
  // Make an SVG Container
  const svg = d3.select(outputId).append("svg")
    .attr("width", 400)
    .attr("height", 250);
  const m = 30;
  const margin = { top: m, right: m, bottom: m, left: m };
  const width = +svg.attr("width") - margin.left - margin.right;
  const height = +svg.attr("height") - margin.top - margin.bottom;
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

export function show(...args) {
  //notebook.appendOutput(currentPlot);
}
