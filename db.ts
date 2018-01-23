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

import { assert } from "./util";

export interface UserInfo {
  uid: string;
  photoURL: string;
  displayName: string;
}
// Defines the scheme of the notebooks collection.
export interface NotebookDoc {
  cells: string[];
  owner: UserInfo;
  updated: Date;
  created: Date;
}

const firebaseConfig = {
  apiKey: "AIzaSyAc5XVKd27iXdGf1ZEFLWudZbpFg3nAwjQ",
  authDomain: "propel-ml.firebaseapp.com",
  databaseURL: "https://propel-ml.firebaseio.com",
  messagingSenderId: "587486455356",
  projectId: "propel-ml",
  storageBucket: "propel-ml.appspot.com",
};

// Constructor arguments to Handle.
interface HandleConfig {
  nbId: string;
  onAuthStateChange: (user: UserInfo) => void;
  onDocChange: (doc: NotebookDoc) => void;
  onError: (msg: string) => void;
}

export class Handle {
  private db: firebase.firestore.Firestore;
  private nbCollection: firebase.firestore.CollectionReference;
  private auth: firebase.auth.Auth;
  private nbDoc: NotebookDoc;
  private docRef: firebase.firestore.DocumentReference;
  private removeDocListener: () => void;
  private removeAuthListener: () => void;

  constructor(public config: HandleConfig) {
    firebase.initializeApp(firebaseConfig);
    firebase.firestore.setLogLevel("debug");
    this.db = firebase.firestore();
    this.auth = firebase.auth();
    this.nbDoc = null;
    this.nbCollection = this.db.collection("notebooks");
    this.docRef = this.nbCollection.doc(this.config.nbId);
  }

  dispose() {
    if (this.removeDocListener) this.removeDocListener();
    if (this.removeAuthListener) this.removeAuthListener();
  }

  signIn() {
    const provider = new firebase.auth.GithubAuthProvider();
    this.auth.signInWithPopup(provider);
  }

  signOut() {
    this.auth.signOut();
  }

  // Call this in order to receive the various callbacks.
  subscribe(): void {
    assert(this.removeAuthListener == null);
    assert(this.removeDocListener == null);

    this.removeAuthListener = this.auth.onAuthStateChanged(
        this.config.onAuthStateChange);

    this.removeDocListener = this.docRef.onSnapshot((doc) => {
      // console.log("onSnapshot", doc);
      if (doc.exists) {
        const d = doc.data() as NotebookDoc;
        this.nbDoc = d;
        this.config.onDocChange(d);
      } else {
        this.config.onError(`Notebook does not exist ${this.config.nbId}`);
      }
    }, (error) => {
      this.config.onError(error.message);
    });
  }

  ownsDoc(): boolean {
    const u = this.auth.currentUser;
    const d = this.nbDoc;
    return u && d && u.uid === d.owner.uid;
  }

  async update(cells: string[]): Promise<void> {
    if (!this.ownsDoc()) return;
    try {
      await this.docRef.update({
        cells,
        updated: firebase.firestore.FieldValue.serverTimestamp(),
      });
    } catch (e) {
      // TODO Display error in UI.
      this.config.onError(e.message);
    }
  }

  // Attempts to clone the given notebook given the Id.
  // Promise resolves to the id of the new notebook which will be owned by the
  // current user.
  async clone(existingDoc: NotebookDoc): Promise<string> {
    const u = this.auth.currentUser;
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
    const docRef = await this.nbCollection.add(newDoc);
    return docRef.id;
  }
}
