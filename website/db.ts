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
import { assert } from "../src/util";

// tslint:disable:no-reference
/// <reference path="firebase.d.ts" />

export interface Database {
  getDoc(nbId): Promise<NotebookDoc>;
  updateDoc(nbId: string, doc: NotebookDoc): Promise<void>;
  clone(existingDoc: NotebookDoc): Promise<string>;
  create(): Promise<string>;
  queryLatest(): Promise<NbInfo[]>;
  signIn(): void;
  signOut(): void;
  subscribeAuthChange(cb: (user: UserInfo) => void): UnsubscribeCb;
}

export interface UserInfo {
  displayName: string;
  photoURL: string;
  uid: string;
}

export interface CellDoc {
  id: string;
  input: string;
  outputHTML: null | string;
}

// Defines the scheme of the notebooks collection.
export interface NotebookDoc {
  anonymous?: boolean;
  cells?: string[];
  cellDocs?: CellDoc[];  // Coming soon.
  owner: UserInfo;
  title: string;
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
    if (!ownsDoc(auth.currentUser, doc)) return;
    const docRef = nbCollection.doc(nbId);
    await docRef.update({
      cells: doc.cells,
      title: doc.title || "",
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
      title: "",
      updated: firebase.firestore.FieldValue.serverTimestamp(),
    };
    console.log({newDoc});
    const docRef = await nbCollection.add(newDoc);
    return docRef.id;
  }

  async create(): Promise<string> {
    lazyInit();
    const u = auth.currentUser;
    if (!u) return "anonymous";

    const newDoc = {
      cells: [ "// New Notebook. Insert code here." ],
      created: firebase.firestore.FieldValue.serverTimestamp(),
      owner: {
        displayName: u.displayName,
        photoURL: u.photoURL,
        uid: u.uid,
      },
      title: "",
      updated: firebase.firestore.FieldValue.serverTimestamp(),
    };
    console.log({newDoc});
    const docRef = await nbCollection.add(newDoc);
    return docRef.id;
  }

  async queryLatest(): Promise<NbInfo[]> {
    lazyInit();
    const query = nbCollection.orderBy("updated", "desc").limit(100);
    const snapshots = await query.get();
    const out = [];
    snapshots.forEach(snap => {
      const nbId = snap.id;
      const doc = snap.data();
      out.unshift({ nbId, doc });
    });
    return out.reverse();
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

  subscribeAuthChange(cb: (user: UserInfo) => void): UnsubscribeCb {
    lazyInit();
    return auth.onAuthStateChanged(cb);
  }
}

export class DatabaseMock implements Database {
  private currentUser: UserInfo = null;
  private docs: { [key: string]: NotebookDoc };
  counts = {};
  inc(method) {
    if (method in this.counts) {
      this.counts[method] += 1;
    } else {
      this.counts[method] = 1;
    }
  }

  constructor() {
    assert(defaultDoc != null);
    this.docs = { "default": Object.assign({}, defaultDoc) };
  }

  async getDoc(nbId: string): Promise<NotebookDoc> {
    this.inc("getDoc");
    if (this.docs[nbId] === null) {
      throw Error("getDoc called with bad nbId " + nbId);
    }
    return this.docs[nbId];
  }

  async updateDoc(nbId: string, doc: NotebookDoc): Promise<void> {
    this.inc("updateDoc");
    this.docs[nbId] = Object.assign(this.docs[nbId], doc);
  }

  async clone(existingDoc: NotebookDoc): Promise<string> {
    this.inc("clone");
    return "clonedNbId";
  }

  async create(): Promise<string> {
    this.inc("create");
    return "createdNbId";
  }

  async queryLatest(): Promise<NbInfo[]> {
    this.inc("queryLatest");
    return [];
  }

  signIn(): void {
    this.inc("signIn");
    this.currentUser = defaultOwner;
    this.makeAuthChangeCallbacks();
  }

  signOut(): void {
    this.inc("signOut");
    this.currentUser = null;
    this.makeAuthChangeCallbacks();
  }

  private authChangeCallbacks = [];
  private makeAuthChangeCallbacks() {
    for (const cb of this.authChangeCallbacks) {
      cb(this.currentUser);
    }
  }

  subscribeAuthChange(cb: (user: UserInfo) => void): UnsubscribeCb {
    this.inc("subscribeAuthChange");
    this.authChangeCallbacks.push(cb);
    return () => {
      const i = this.authChangeCallbacks.indexOf(cb);
      this.authChangeCallbacks.splice(i, 1);
    };
  }
}

export let active: Database = null;

export function enableFirebase() {
  active = new DatabaseFB();
}

export function enableMock(): DatabaseMock {
  const d = new DatabaseMock();
  active = d;
  return d;
}

export function ownsDoc(userInfo: UserInfo, doc: NotebookDoc): boolean {
  return userInfo && userInfo.uid === doc.owner.uid;
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

export const defaultOwner: UserInfo = Object.freeze({
  displayName: "default owner",
  photoURL: "https://avatars1.githubusercontent.com/u/80?v=4",
  uid: "abc",
});

const defaultDocCells: ReadonlyArray<string> = Object.freeze([
`
import { tensor } from "propel";
t = tensor([[2, 3], [30, 20]])
t.mul(5)
`,
`
import { grad, linspace, plot } from "propel";
f = (x) => tensor(x).mul(x);
x = linspace(-4, 4, 200);
plot(x, f(x),
     x, grad(f)(x));
`,
`
f = await fetch('/data/mnist/README');
t = await f.text();
t;
`,
`
import { tensor } from "propel";
function f(x) {
  let y = x.sub(1);
  let z = tensor(-1).sub(x);
  return x.greater(0).select(y,z).relu();
}
x = linspace(-5, 5, 100)
plot(x, f(x))
plot(x, grad(f)(x))
grad(f)([-3, -0.5, 0.5, 3])
`
]);

export const defaultDoc: NotebookDoc = Object.freeze({
  cells: defaultDocCells.slice(0).map(c => c.trim()),
  created: new Date(),
  owner: Object.assign({}, defaultOwner),
  title: "Sample Notebook",
  updated: new Date(),
});

// To bridge the old and new NotebookDoc scheme.
// In the old NotebookDoc we only had `doc.cells`, in the new
// scheme we have `cellDocs`.
export function getInputCodes(doc: NotebookDoc): string[] {
  if (doc.cells != null) {
    return doc.cells;
  } else if (doc.cellDocs != null) {
    return doc.cellDocs.map(cellDoc => cellDoc.input);
  } else {
    return [];
  }
}
