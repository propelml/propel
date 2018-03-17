import { h } from "preact";
import * as db from "./db";

export function Loading(props) {
  return h("div", { "class": "notification-screen"},
    h("div", { "class": "notification-container"},
      h("h1", null, "Loading"),
    ),
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
