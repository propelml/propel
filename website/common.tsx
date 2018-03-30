// tslint:disable:variable-name
import { h } from "preact";
import * as db from "./db";

export function Loading(props) {
  return (
    <div class="notification-screen">
      <div class="notification-container">
        <h1>Loading</h1>
      </div>
    </div>
  );
}

export function PropelLogo(props) {
  let Subtitle = null;
  if (props.subtitle) {
    Subtitle = (
      <h2>
        <a href={ props.subtitleLink || "/" }>{ props.subtitle }</a>
      </h2>
    );
  }
  return (
    <div class="propel-logo">
      <div class="logo">
        <svg
          height={ 24 }
          viewBox="0 0 24 24"
          width={ 24 }
          xmlns="http://www.w3.org/2000/svg" >
          <circle cx={ 12 } cy={ 12 } r={ 12 } />
        </svg>
      </div>
      <div class="global-title">
        <div class="global-main-title">
          <h1><a href="/">Propel</a></h1>
        </div>
        <div class="global-sub-title">
          { Subtitle }
        </div>
      </div>
    </div>
  );
}

export function Footer(props) {
  return (
    <div class="footer">
      <a href="/references">References</a>
      <a href="/docs">Documentation</a>
      <a href="https://github.com/propelml/propel">GitHub</a>
      <a href="mailto:propelml@gmail.com">Contact</a>
    </div>
  );
}

export function GlobalHeader(props) {
  return (
    <div class="global-header">
      <div class="global-header-inner">
        <PropelLogo
          subtitle={ props.subtitle }
          subtitleLink={ props.subtitleLink } />
        <div class="global-header-right">
          { props.children }
        </div>
      </div>
    </div>
  );
}

export function UserMenu(props) {
  if (props.userInfo) {
    return (
      <div class="dropdown">
        <Avatar size={ 32 } userInfo={ props.userInfo } />
        <div class="dropdown-content">
          <a href="#" onClick={ db.active.signOut } >
            Sign out
          </a>
        </div>
      </div>
    );
  }
  return <a href="#" onClick={ db.active.signIn }>Sign in</a>;
}

export function Avatar(props: { size?: number, userInfo: db.UserInfo }) {
  const size = props.size || 50;
  return (
    <img
      class="avatar"
      height={ size }
      src={ props.userInfo.photoURL + "&size=" + size }
      width={ size } />
  );
}
