import * as d3 from "d3";
import $ from "./propel";
import * as repl from "./repl";
import { assertEqual } from "./util";

const currentPlot = null;

// TODO colors should match those used in highlight.js.
const color = d3.scaleOrdinal(d3.schemeCategory10);

function plotLines(data) {
  const outputId = repl.outputId();
  // Make an SVG Container
  const svg = d3.select(outputId).append("svg")
    .attr("width", 400)
    .attr("height", 200);
  const margin = { top: 10, right: 10, bottom: 10, left: 10 };
  const width = +svg.attr("width") - margin.left - margin.right;
  const height = +svg.attr("height") - margin.top - margin.bottom;
  const g = svg.append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  const xScale = d3.scaleLinear()
    .domain([-7, 7])
    .range([0, width]);

  const yScale = d3.scaleLinear()
    .domain([-1, 1])
    .range([height, 0]);

  const line = (d3.line() as any)
    .x((d) => xScale(d.x))
    .y((d) => yScale(d.y));

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

function plot(...args) {
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
    const xv = xs[i].ndarray.getValues();
    const yv = ys[i].ndarray.getValues();
    assertEqual(xv.length, yv.length);
    const line = [];
    for (let j = 0; j < xv.length; ++j) {
      line.push({ x: xv[j], y: yv[j] });
    }
    data.push(line);
  }

  plotLines(data);
  //currentPlot = svg;
  //return svg
}

function show(...args) {
  //repl.appendOutput(currentPlot);
}

export default {
  plot,
  show,
};
