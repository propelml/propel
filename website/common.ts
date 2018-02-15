import { h } from "preact";

export function Loading(props) {
  return h("h1", null, "Loading");
}

export function PropelLogo(props) {
  let subtitle = null;
  if (props.subtitle) {
    subtitle = h("h2", null,
      h("a", { href: props.subtitleLink || "/" }, props.subtitle));
  }
  return h("div", { "class": "propel-logo" },
    h("svg", {
      height: 24,
      viewBox: "0 0 24 24",
      width: 24,
      xmlns: "http://www.w3.org/2000/svg",
    }, h("circle", { cx: 12, cy: 12, r: 12 })),
    h("h1", null, h("a", { href: "/"  }, "Propel")),
    subtitle,
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
