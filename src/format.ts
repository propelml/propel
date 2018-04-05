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
// This module is just to implement tensor.toString.
import { Tensor } from "./tensor";

export type FormatterFunction = (arg: any) => string;

export interface Formatter {
  int?: FormatterFunction;
  float?: FormatterFunction;
}

export interface FormatOptions {
  edgeitems?: number;
  threshold?: number;
  precision?: number;
  linewidth?: number;
  formatter?:  FormatterFunction | Formatter;
  // TODO
  // floatmode?: "fixed" | "unique" | "maxprec" | "maxprec_equal";
  // nanstr?: string;
  // infstr?: string;
  // sign?: string;
  // suppress?: boolean;
}

let defaultFormatOptions: FormatOptions = {
  // repr N leading and trailing items of each dimension
  "edgeitems": 3,
  // total items > triggers array summarization
  "threshold": 1000,
  // Precision of floating point representations.
  "precision": 6,
  "linewidth": 75,
  "formatter": undefined,
};

export function setPrintoptions(options: FormatOptions) {
  defaultFormatOptions = {
    ...defaultFormatOptions,
    ...options
  };
}

export function getPrintoptions(): FormatOptions {
  return { ...defaultFormatOptions };
}

function floatFormatter(tensor, precision): FormatterFunction {
  // TODO - This function needs some optimizations
  // Also it does not generate same output as NumPy in somecases
  const format = x => String(Number(x).toFixed(precision)).replace(/0+$/, "");
  const data = tensor.dataSync();
  let maxBefore = 0;
  let maxAfter = 0;
  for (let i = 0; i < data.length; ++i) {
    const [before, after] = format(data[i]).replace(/^-/, "").split(".", 2);
    if (maxBefore < before.length) maxBefore = before.length;
    if (maxAfter < after.length) maxAfter = after.length;
  }
  maxBefore += 1;
  return function(x) {
    const [before, after] = format(x).split(".", 2);
    return " ".repeat(Math.max(maxBefore - before.length, 0)) + before +
           "." + after + " ".repeat(Math.max(maxAfter - after.length, 0));
  };
}

function IntegerFormatter(tensor: Tensor): FormatterFunction {
  let maxStrLen = 0;
  if (tensor.size > 0) {
    maxStrLen = Math.max(String(tensor.reduceMax()).replace(/^-/, "").length,
                         String(tensor.reduceMin()).replace(/^-/, "").length);
  }
  return function(x) {
    x = String(x);
    return (" ").repeat(Math.max(maxStrLen - x.length, 0)) + x;
  };
}

// This function provides a way to get an element from a tensor by it's index.
// indexableTensor(pr.range(200).reshape(2, 100))([1, 5])
function indexableTensor(tensor: Tensor) {
  const data = tensor.dataSync();
  return function(index: number[]) {
    const shape = [...tensor.shape];
    shape.shift();
    let flatIndex = 0;
    for (let i = 0; i < index.length; ++i) {
      flatIndex += shape.reduce((a, b) => a * b, 1) * index[i];
      shape.shift();
    }
    return data[flatIndex];
  };
}

function extendLine(s, line, word, lineWidth, nextLinePrefix) {
  const needsWrap = (line.length + word.length) > lineWidth;
  if (needsWrap) {
    s += line.trimRight() + "\n";
    line = nextLinePrefix;
  }
  line += word;
  return [s, line];
}

function formatTensor(t: Tensor, formatFunction: FormatterFunction,
                      lineWidth: number, nextLinePrefix: string,
                      separator: string, edgeItems: number,
                      summaryInsert: string): string {
  const get = indexableTensor(t);
  const recurser = (index: number[], hangingIndent: string,
                    currWidth: number) => {
    const axis = index.length;
    const axesLeft = t.rank - axis;

    if (axesLeft === 0) {
      return formatFunction(get(index));
    }

    // When recursing, add a space to align with the [ added, and reduce the
    // length of the line by 1.
    const nextHangingIndent = hangingIndent + " ";
    const nextWidth = currWidth - ("]").length;
    const tLen = t.shape[axis];
    const showSummary = summaryInsert.length > 0 && 2 * edgeItems < tLen;

    let leadingItems = 0;
    let trailingItems = tLen;
    if (showSummary) {
      leadingItems = edgeItems;
      trailingItems = edgeItems;
    }

    // Stringify the array with the hanging indent on the first line too.
    let s = "";

    // Last axis (rows) - wrap elements if they would not fit on one line.
    if (axesLeft === 1) {
      // 1 for (']').length
      const elemWidth = currWidth - Math.max(separator.trimRight().length, 1);
      let line = hangingIndent;
      for (let i = 0; i < leadingItems; ++i) {
        const word = recurser([...index, i], nextHangingIndent, nextWidth);
        [s, line] = extendLine(s, line, word, elemWidth, hangingIndent);
        line += separator;
      }
      if (showSummary) {
        [s, line] = extendLine(s, line, summaryInsert,
                               elemWidth, hangingIndent);
        line += separator;
      }
      for (let i = trailingItems; i > 1; --i) {
        const word = recurser([...index, (t.shape[axis] - i)],
                              nextHangingIndent, nextWidth);
        [s, line] = extendLine(s, line, word, elemWidth, hangingIndent);
        line += separator;
      }
      const word = recurser([...index, (t.shape[axis] - 1)],
                            nextHangingIndent, nextWidth);
      [s, line] = extendLine(s, line, word, elemWidth, hangingIndent);
      s += line;
    } else {
      // Other axes - insert newlines between rows.
      const lineSep = separator.trimRight() + "\n".repeat(axesLeft - 1);
      for (let i = 0; i < leadingItems; ++i) {
        const nested = recurser([...index, i], nextHangingIndent, nextWidth);
        s += hangingIndent + nested + lineSep;
      }
      if (showSummary) {
        s += hangingIndent + summaryInsert + lineSep;
      }
      for (let i = trailingItems; i > 1; --i) {
        const nested = recurser([...index, (t.shape[axis] - i)],
                                nextHangingIndent, nextWidth);
        s += hangingIndent + nested + lineSep;
      }
      const nested = recurser([...index, (t.shape[axis] - 1)],
                              nextHangingIndent, nextWidth);
      s += hangingIndent + nested;
    }
    // Remove the hanging indent, and wrap in [].
    return `[${s.slice(hangingIndent.length)}]`;
  };
  return recurser([], nextLinePrefix, lineWidth);
}

function getFormatFunction(tensor: Tensor, opts: FormatOptions)
    : FormatterFunction {
  if (opts.formatter && opts.formatter[tensor.dtype]) {
    return opts.formatter[tensor.dtype];
  }
  switch (tensor.dtype){
    case "int32":
    case "uint8":
      return IntegerFormatter(tensor);
    case "float32":
      return floatFormatter(tensor, opts.precision);
  }
  throw new Error("Unsupported dtype.");
}

export function toString(tensor: Tensor, opts: FormatOptions = {},
                         separator = ", ", prefix = ""): string {
  // The str of 0d arrays is a special case: It should appear like a scalar.
  if (tensor.rank === 0) {
    return tensor.dataSync()[0] + "";
  }
  opts = {
    ...defaultFormatOptions,
    ...opts
  };

  let summaryInsert;
  if (tensor.size > opts.threshold) {
    // TODO summarize the output.
    summaryInsert = "...";
  } else {
    summaryInsert = "";
  }

  // Get formatter for this dtype.
  const formatFunction = getFormatFunction(tensor, opts);

  const nextLinePrefix =  " ";
  return formatTensor(tensor, formatFunction, opts.linewidth,
                      nextLinePrefix, separator, opts.edgeitems,
                      summaryInsert);
}
