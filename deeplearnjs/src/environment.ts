/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as device_util from './device_util';
import * as util from './util';

export enum Type {
  NUMBER,
  BOOLEAN
}

export interface Features {
  // Whether the disjoint_query_timer extension is an available extension.
  'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED'?: boolean;
  // Whether the timer object from the disjoint_query_timer extension gives
  // timing information that is reliable.
  'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE'?: boolean;
  // 0: No WebGL, 1: WebGL 1.0, 2: WebGL 2.0.
  'WEBGL_VERSION'?: number;
  // Whether writing & reading floating point textures is enabled. When
  // false, fall back to using unsigned byte textures.
  'WEBGL_FLOAT_TEXTURE_ENABLED'?: boolean;
  // Whether WEBGL_get_buffer_sub_data_async is enabled.
  'WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED'?: boolean;
}

export const URL_PROPERTIES: URLProperty[] = [
  {name: 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED', type: Type.BOOLEAN},
  {name: 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE', type: Type.BOOLEAN},
  {name: 'WEBGL_VERSION', type: Type.NUMBER},
  {name: 'WEBGL_FLOAT_TEXTURE_ENABLED', type: Type.BOOLEAN}, {
    name: 'WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED',
    type: Type.BOOLEAN
  }
];

export interface URLProperty {
  name: keyof Features;
  type: Type;
}

function getWebGLRenderingContext(webGLVersion: number): WebGLRenderingContext {
  if (webGLVersion === 0) {
    throw new Error('Cannot get WebGL rendering context, WebGL is disabled.');
  }

  const tempCanvas = document.createElement('canvas');

  if (webGLVersion === 1) {
    return (tempCanvas.getContext('webgl') ||
            tempCanvas.getContext('experimental-webgl')) as
        WebGLRenderingContext;
  }
  return tempCanvas.getContext('webgl2') as WebGLRenderingContext;
}

function loseContext(gl: WebGLRenderingContext) {
  if (gl != null) {
    const loseContextExtension = gl.getExtension('WEBGL_lose_context');
    if (loseContextExtension == null) {
      throw new Error(
          'Extension WEBGL_lose_context not supported on this browser.');
    }
    loseContextExtension.loseContext();
  }
}

function isWebGLVersionEnabled(webGLVersion: 1|2) {
  const gl = getWebGLRenderingContext(webGLVersion);
  if (gl != null) {
    loseContext(gl);
    return true;
  }
  return false;
}

function isWebGLDisjointQueryTimerEnabled(webGLVersion: number) {
  const gl = getWebGLRenderingContext(webGLVersion);

  const extensionName = webGLVersion === 1 ? 'EXT_disjoint_timer_query' :
                                             'EXT_disjoint_timer_query_webgl2';
  const ext = gl.getExtension(extensionName);
  const isExtEnabled = ext != null;
  if (gl != null) {
    loseContext(gl);
  }
  return isExtEnabled;
}

function isFloatTextureReadPixelsEnabled(webGLVersion: number): boolean {
  if (webGLVersion === 0) {
    return false;
  }

  const gl = getWebGLRenderingContext(webGLVersion);

  if (webGLVersion === 1) {
    if (gl.getExtension('OES_texture_float') == null) {
      return false;
    }
  } else {
    if (gl.getExtension('EXT_color_buffer_float') == null) {
      return false;
    }
  }

  const frameBuffer = gl.createFramebuffer();
  const texture = gl.createTexture();

  gl.bindTexture(gl.TEXTURE_2D, texture);

  // tslint:disable-next-line:no-any
  const internalFormat = webGLVersion === 2 ? (gl as any).RGBA32F : gl.RGBA;
  gl.texImage2D(
      gl.TEXTURE_2D, 0, internalFormat, 1, 1, 0, gl.RGBA, gl.FLOAT, null);

  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
  gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

  const frameBufferComplete =
      (gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE);

  loseContext(gl);

  return frameBufferComplete;
}

function isWebGLGetBufferSubDataAsyncExtensionEnabled(webGLVersion: number) {
  if (webGLVersion !== 2) {
    return false;
  }
  const gl = getWebGLRenderingContext(webGLVersion);
  const ext = gl.getExtension('WEBGL_get_buffer_sub_data_async');
  const isEnabled = ext != null;
  loseContext(gl);
  return isEnabled;
}

export class Environment {
  private features: Features = {};

  constructor(features?: Features) {
    if (features != null) {
      this.features = features;
    }
  }

  get<K extends keyof Features>(feature: K): Features[K] {
    if (feature in this.features) {
      return this.features[feature];
    }

    this.features[feature] = this.evaluateFeature(feature);

    return this.features[feature];
  }

  private evaluateFeature<K extends keyof Features>(feature: K): Features[K] {
    if (feature === 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED') {
      const webGLVersion = this.get('WEBGL_VERSION');

      if (webGLVersion === 0) {
        return false;
      }

      return isWebGLDisjointQueryTimerEnabled(webGLVersion);
    } else if (feature === 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') {
      return this.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED') &&
          !device_util.isMobile();
    } else if (feature === 'WEBGL_VERSION') {
      if (isWebGLVersionEnabled(2)) {
        return 2;
      } else if (isWebGLVersionEnabled(1)) {
        return 1;
      }
      return 0;
    } else if (feature === 'WEBGL_FLOAT_TEXTURE_ENABLED') {
      return isFloatTextureReadPixelsEnabled(this.get('WEBGL_VERSION'));
    } else if (
        feature === 'WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED') {
      return isWebGLGetBufferSubDataAsyncExtensionEnabled(
          this.get('WEBGL_VERSION'));
    }
    throw new Error(`Unknown feature ${feature}.`);
  }
}

// Expects flags from URL in the format ?dljsflags=FLAG1:1,FLAG2:true.
const DEEPLEARNJS_FLAGS_PREFIX = 'dljsflags';
function getFeaturesFromURL(): Features {
  const features: Features = {};

  if (typeof window === 'undefined') {
    return features;
  }

  const urlParams = util.getQueryParams(window.location.search);
  if (DEEPLEARNJS_FLAGS_PREFIX in urlParams) {
    const urlFlags: {[key: string]: string} = {};

    const keyValues = urlParams[DEEPLEARNJS_FLAGS_PREFIX].split(',');
    keyValues.forEach(keyValue => {
      const [key, value] = keyValue.split(':') as [string, string];
      urlFlags[key] = value;
    });

    URL_PROPERTIES.forEach(urlProperty => {
      if (urlProperty.name in urlFlags) {
        console.log(
            `Setting feature override from URL ${urlProperty.name}: ` +
            `${urlFlags[urlProperty.name]}`);
        if (urlProperty.type === Type.NUMBER) {
          features[urlProperty.name] = +urlFlags[urlProperty.name];
        } else if (urlProperty.type === Type.BOOLEAN) {
          features[urlProperty.name] = urlFlags[urlProperty.name] === 'true';
        } else {
          console.warn(`Unknown URL param: ${urlProperty.name}.`);
        }
      }
    });
  }

  return features;
}

export let ENV = new Environment(getFeaturesFromURL());

export function setEnvironment(environment: Environment) {
  ENV = environment;
}
