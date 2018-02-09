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

// This file contains routines for accessing the firebase database (firestore).
// This is used to save and restore notebooks.
// These routines are run only on the browser.

// tslint:disable:no-reference
/// <reference path="firebase.d.ts" />

export interface Database {
  getDoc(nbId): Promise<NotebookDoc>;
  updateDoc(nbId: string, doc: NotebookDoc): Promise<void>;
  clone(existingDoc: NotebookDoc): Promise<string>;
  queryLatest(): Promise<NbInfo[]>;
  signIn(): void;
  signOut(): void;
  ownsDoc(doc: NotebookDoc): boolean;
  subscribeAuthChange(cb: (user: UserInfo) => void): UnsubscribeCb;
}

export interface UserInfo {
  displayName: string;
  photoURL: string;
  uid: string;
}

// Defines the scheme of the notebooks collection.
export interface NotebookDoc {
  cells: string[];
  owner: UserInfo;
  updated: Date;
  created: Date;
}

export interface UnsubscribeCb {
  (): void;
}

export interface NbInfo {
  nbId: string;
  doc: NotebookDoc;
}

// These are shared by all functions and are lazily constructed by lazyInit.
let db: firebase.firestore.Firestore;
let nbCollection: firebase.firestore.CollectionReference;
let auth: firebase.auth.Auth;
const firebaseConfig = {
  apiKey: "AIzaSyAc5XVKd27iXdGf1ZEFLWudZbpFg3nAwjQ",
  authDomain: "propel-ml.firebaseapp.com",
  databaseURL: "https://propel-ml.firebaseio.com",
  messagingSenderId: "587486455356",
  projectId: "propel-ml",
  storageBucket: "propel-ml.appspot.com",
};

class DatabaseFB implements Database {
  async getDoc(nbId): Promise<NotebookDoc> {
    // We have one special doc that is loaded from memory, used for testing and
    // debugging.
    if (nbId === "default") {
      return defaultDoc;
    }
    const docRef = nbCollection.doc(nbId);
    const snap = await docRef.get();
    if (snap.exists) {
      return snap.data() as NotebookDoc;
    } else {
      throw Error(`Notebook does not exist ${nbId}`);
    }
  }

  // Caller must catch errors.
  async updateDoc(nbId: string, doc: NotebookDoc): Promise<void> {
    if (nbId === "default") return; // Don't save the default doc.
    if (!this.ownsDoc(doc)) return;
    const docRef = nbCollection.doc(nbId);
    await docRef.update({
      cells: doc.cells,
      updated: firebase.firestore.FieldValue.serverTimestamp(),
    });
  }

  // Attempts to clone the given notebook given the Id.
  // Promise resolves to the id of the new notebook which will be owned by the
  // current user.
  async clone(existingDoc: NotebookDoc): Promise<string> {
    lazyInit();
    const u = auth.currentUser;
    if (!u) throw Error("Cannot clone. User must be logged in.");

    if (existingDoc.cells.length === 0) {
      throw Error("Cannot clone empty notebook.");
    }

    const newDoc = {
      cells: existingDoc.cells,
      created: firebase.firestore.FieldValue.serverTimestamp(),
      owner: {
        displayName: u.displayName,
        photoURL: u.photoURL,
        uid: u.uid,
      },
      updated: firebase.firestore.FieldValue.serverTimestamp(),
    };
    console.log({newDoc});
    const docRef = await nbCollection.add(newDoc);
    return docRef.id;
  }

  async queryLatest(): Promise<NbInfo[]> {
    lazyInit();
    const query = nbCollection.orderBy("updated").limit(100);
    const snapshots = await query.get();
    const out = [];
    snapshots.forEach(snap => {
      const nbId = snap.id;
      const doc = snap.data();
      out.unshift({ nbId, doc });
    });
    return out;
  }

  signIn() {
    lazyInit();
    const provider = new firebase.auth.GithubAuthProvider();
    auth.signInWithPopup(provider);
  }

  signOut() {
    lazyInit();
    auth.signOut();
  }

  ownsDoc(d: NotebookDoc): boolean {
    const u = auth.currentUser;
    return u && d && u.uid === d.owner.uid;
  }

  subscribeAuthChange(cb: (user: UserInfo) => void): UnsubscribeCb {
    lazyInit();
    return auth.onAuthStateChanged(cb);
  }
}

class DatabaseMock implements Database {
  counts = {};
  inc(method) {
    if (method in this.counts) {
      this.counts[method] += 1;
    } else {
      this.counts[method] = 1;
    }
  }

  async getDoc(nbId: string): Promise<NotebookDoc> {
    this.inc("getDoc");
    return defaultDoc;
  }

  async updateDoc(nbId: string, doc: NotebookDoc): Promise<void> {
    this.inc("updateDoc");
  }

  async clone(existingDoc: NotebookDoc): Promise<string> {
    this.inc("clone");
    return "clonedNbId";
  }

  async queryLatest(): Promise<NbInfo[]> {
    this.inc("queryLatest");
    return [];
  }

  signIn(): void {
    this.inc("signIn");
  }

  signOut(): void {
    this.inc("signOut");
  }

  ownsDoc(doc: NotebookDoc): boolean {
    this.inc("ownsDoc");
    return false;
  }

  subscribeAuthChange(cb: (user: UserInfo) => void): UnsubscribeCb {
    this.inc("subscribeAuthChange");
    return null;
  }
}

// Default to a mock, so none of the functions errors out during operation in
// Node. Firebase cannot be loaded in Node.
// We cannot load Firebase in Node because grpc has an unreasonable
// amount of dependencies, included a whole copy of OpenSSL.
// The firebase web client is very difficult to get working in Node
// even when trying to use the grpc-web-client library.
export let active: Database = new DatabaseMock();

export function enableFirebase() {
  active = new DatabaseFB();
}

export function enableMock(): DatabaseMock {
  const d = new DatabaseMock();
  active = d;
  return d;
}

function lazyInit() {
  if (db == null) {
    firebase.initializeApp(firebaseConfig);
    db = firebase.firestore();
    // firebase.firestore.setLogLevel("debug");
    auth = firebase.auth();
    nbCollection = db.collection("notebooks");
  }
  return true;
}

const defaultOwner: UserInfo = {
  displayName: "default owner",
  photoURL: "https://avatars1.githubusercontent.com/u/80?v=4",
  uid: "abc",
};

const defaultDocCells: string[] = [
`
import { T } from "propel";
t = T([[2, 3], [30, 20]])
t.mul(5)
`,
`
import { grad, linspace, plot } from "propel";
f = (x) => T(x).mul(x);
x = linspace(-4, 4, 200);
plot(x, f(x),
     x, grad(f)(x));
`,
`
f = await fetch('/static/mnist/README');
t = await f.text();
t;
`,
`
import { T } from "propel";
function f(x) {
  let y = x.sub(1);
  let z = T(-1).sub(x);
  return x.greater(0).select(y,z).relu();
}
x = linspace(-5, 5, 100)
plot(x, f(x))
plot(x, grad(f)(x))
grad(f)([-3, -0.5, 0.5, 3])
`
];

const defaultDoc: NotebookDoc = {
  cells: defaultDocCells,
  owner: defaultOwner,
  updated: new Date(),
  created: new Date(),
};
