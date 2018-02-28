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

import { createResolvable, Resolvable } from "../src/util";

export type RpcHandler = (...args: any[]) => any;
export type RpcHandlers = { [name: string]: RpcHandler };

interface Message {
  type: "syn" | "ack" | "call" | "return";
}

interface CallMessage extends Message {
  type: "call";
  id: string;
  handler: string;
  args: any[];
}

interface ReturnMessage extends Message {
  type: "return";
  id: string;
  result?: any;
  exception?: any;
}

export class SandboxRPC {
  // TODO: better solution for filtering messages intended for other frames.
  private unique = Math.floor(Math.random() * 1 << 30).toString(16);
  private counter = 0;
  private ready = createResolvable();
  private returnHandlers = new Map<string, Resolvable<any>>();

  constructor(private remote: Window, private handlers: RpcHandlers) {
    // TODO: remove event listener when remote window disappears.
    window.addEventListener("message", event => this.onMessage(event));
    this.remote.postMessage({type: "syn"}, "*");
  }

  async call(handler: string, ...args: any[]): Promise<any> {
    await this.ready;

    const id = `${this.unique}_${this.counter++}`;
    const message: CallMessage = {
      type: "call",
      id,
      handler,
      args
    };

    const resolver = createResolvable<any>();
    this.returnHandlers.set(id, resolver);

    try {
      this.remote.postMessage(message, "*");
      return await resolver;
    } finally {
      this.returnHandlers.delete(id);
    }
  }

  private onMessage(event: MessageEvent): void {
    const { type } = event.data;
    switch (type) {
      case "syn":
      case "ack":
        this.onHandshake(event);
        break;
      case "call":
        this.onCall(event);
        break;
      case "return":
        this.onReturn(event);
        break;
    }
  }

  private onHandshake(event: MessageEvent) {
    const {type} = event.data;
    if (type === "syn") {
      this.remote.postMessage({ type: "ack" }, "*");
    }
    this.ready.resolve();
  }

  private async onCall(event: MessageEvent) {
    const {id, handler, args} = event.data;
    const ret: ReturnMessage = {
      type: "return",
      id
    };
    try {
      const result = await this.handlers[handler](...args);
      this.remote.postMessage({ result, ...ret }, "*");
    } catch (exception) {
      if (exception instanceof Error) {
        // Convert to a normal object.
        const { message, stack } = exception;
        exception = { message, stack, __error__: true };
      }
      this.remote.postMessage({ exception, ...ret }, "*");
    }
  }

  private onReturn(event: MessageEvent) {
    const message: ReturnMessage = event.data;
    const id = message.id;
    const resolver = this.returnHandlers.get(id);
    if (resolver === undefined) {
        return; // Not for us.
    }
    if (message.hasOwnProperty("exception")) {
      let { exception } = message;
      if (exception.__error__) {
        // Convert to Error object.
        exception = Object.assign(new Error(exception.message),
                                  { stack: exception.stack });
      }
      resolver.reject(exception);
    } else {
      resolver.resolve(message.result);
    }
  }
}
