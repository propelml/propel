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
// tslint:disable:variable-name
import { readFileSync } from "fs";
import { h } from "preact";
import { GlobalHeader } from "./common";

export const PropelIndex = (props) => {
  let md = readFileSync(__dirname + "/../README.md", "utf8");
  md = "<p>" + md.replace(/\n\n/g, "\n\n<p>");
  return (
    <div class="index">
      <GlobalHeader />
      <div class="intro flex-row">
        <div class="flex-cell">
          <div dangerouslySetInnerHTML={ { __html: md } } />
          <p><a href="http://github.com/propelml/propel"
            >github.com/propelml/propel</a>
          </p>
        </div>
      </div>
    </div>
  );
};
