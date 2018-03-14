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
import { h } from "preact";
import * as db from "./db";

export function Loading(props) {
  return h("div", { "class": "loading-screen"},
    h("h1", null, "Loading"),
  );
}

export function PropelLogo(props) {
  let subtitle = null;
  if (props.subtitle) {
    subtitle = h("h2", null,
      h("a", { href: props.subtitleLink || "/" }, props.subtitle));
  }
  return h("div", { "class": "propel-logo" },
    h("div", { "class": "logo" },
      h("svg", {
        height: 24,
        viewBox: "0 0 24 24",
        width: 24,
        xmlns: "http://www.w3.org/2000/svg", },
          h("circle", { cx: 12, cy: 12, r: 12 }),
      ),
    ),
    h("div", { "class": "global-title" },
      h("div", { "class": "global-main-title" },
        h("h1", null, h("a", { href: "/"  }, "Propel")),
      ),
      h("div", { "class": "global-sub-title" },
        subtitle,
      ),
    ),

  );
}

export function Footer(props) {
  return h("div", { "class": "footer" },
    h("a", { "href": "/references" }, "References"),
    h("a", { "href": "/docs" }, "Documentation"),
    h("a", { "href": "https://github.com/propelml/propel" }, "GitHub"),
    h("a", { "href": "mailto:propelml@gmail.com" }, "Contact"),
  );
}

export function GlobalHeader(props) {
  return h("div", { "class": "global-header" },
    h("div", { "class": "global-header-inner" },
      h(PropelLogo, {
        subtitle: props.subtitle,
        subtitleLink: props.subtitleLink,
      }),
      h("div", { "class": "global-header-right" }, ...props.children)
    ),
  );
}

export function div(className, ...children) {
  return h("div", { "class": className }, ...children);
}

export function p(...children) {
  return h("p", null, ...children);
}

export function UserMenu(props) {
  if (props.userInfo) {
    return h("div", { "class": "dropdown" },
      h(Avatar, { size: 32, userInfo: props.userInfo }),
      h("div", { "class": "dropdown-content" },
        h("a", {
          "href": "#",
          "onclick": db.active.signOut,
        }, "Sign out"),
      )
    );
  } else {
    return h("a", {
      "href": "#",
      "onclick": db.active.signIn,
    }, "Sign in");
  }
}

export function Avatar(props: { size?: number, userInfo: db.UserInfo }) {
  const size = props.size || 50;
  return h("img", {
    "class": "avatar",
    "height": size,
    "src": props.userInfo.photoURL + "&size=" + size,
    "width": size,
  });
}

// Trims whitespace.
export function normalizeCode(code: string): string {
  return code.trim();
}
