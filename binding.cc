/*
   Copyright 2017 propel authors. All Rights Reserved.
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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <node_api.h>

#include "deps/libtensorflow/include/tensorflow/c/c_api.h"
#include "deps/libtensorflow/include/tensorflow/c/eager/c_api.h"

#define COUNT_OF(array) (sizeof(array) / sizeof(array[0]))

static const size_t kMaxDims = 10;

struct ContextWrap {
  napi_env env;
  TFE_Context* tf_context;
  napi_ref js_typed_array;
};

struct TensorWrap {
  napi_env env;
  TF_Tensor* tf_tensor;
  TFE_TensorHandle* tf_tensor_handle;
  napi_ref js_typed_array;
};

static void ReleaseTypedArray(void* data, size_t len, void* tensor_wrap_ptr) {
  auto tensor_wrap = static_cast<TensorWrap*>(tensor_wrap_ptr);

  assert(tensor_wrap->js_typed_array != NULL);
  napi_status status =
      napi_delete_reference(tensor_wrap->env, tensor_wrap->js_typed_array);
  assert(status == napi_ok);
  tensor_wrap->js_typed_array = NULL;
}

static void DeleteTensor(napi_env env, void* tensor_wrap_ptr, void* hint) {
  auto tensor_wrap = static_cast<TensorWrap*>(tensor_wrap_ptr);
  napi_status status;

  if (tensor_wrap->tf_tensor_handle != NULL)
    TFE_DeleteTensorHandle(tensor_wrap->tf_tensor_handle);

  if (tensor_wrap->tf_tensor != NULL)
    TF_DeleteTensor(tensor_wrap->tf_tensor);

  // At this point, the typed array should no longer be referenced, because
  // tensorflow should have called ReleaseTypedArray(). But since it isn't
  // clear what happens when TF_NewTensor() fails, double check here and clean
  // up if necessary.
  if (tensor_wrap->js_typed_array != NULL) {
    status =
        napi_delete_reference(tensor_wrap->env, tensor_wrap->js_typed_array);
    assert(status == napi_ok);
  }

  delete tensor_wrap;
}

static void DeleteContext(napi_env env, void* wrap_ptr, void* hint) {
  auto wrap = static_cast<ContextWrap*>(wrap_ptr);
  auto tf_status = TF_NewStatus();
  TFE_DeleteContext(wrap->tf_context, tf_status);
  delete wrap;
  TF_DeleteStatus(tf_status);
}

static napi_value NewContext(napi_env env, napi_callback_info info) {
  napi_value js_this;
  auto napi_status = napi_get_cb_info(env, info, 0, NULL, &js_this, NULL);

  auto opts = TFE_NewContextOptions();

  auto tf_status = TF_NewStatus();
  assert(tf_status);
  auto tf_context = TFE_NewContext(opts, tf_status);
  TF_DeleteStatus(tf_status);
  TFE_DeleteContextOptions(opts);

  auto context_wrap = new ContextWrap();
  assert(context_wrap);

  context_wrap->tf_context = tf_context;
  context_wrap->env = env;

  napi_status = napi_wrap(env, js_this, context_wrap, DeleteContext, NULL,
                          NULL);
  assert(napi_status == napi_ok);

  return js_this;
}

static napi_value NewTensor(napi_env env, napi_callback_info info) {
  napi_status napi_status;

  // Check whether this function is called as a construct call.
  napi_value js_target;
  napi_status = napi_get_new_target(env, info, &js_target);
  assert(napi_status == napi_ok);
  if (js_target == NULL) {
    napi_throw_type_error(env, "EINVAL", "Function not used as a constructor");
    return NULL;
  }

  // Fetch JavaScript `this` object and function arguments.
  size_t argc = 2;
  napi_value args[2];
  napi_value js_this;
  napi_status = napi_get_cb_info(env, info, &argc, args, &js_this, NULL);
  assert(napi_status == napi_ok);

  napi_value js_array = args[0];
  napi_value js_dims = args[1];

  // Check whether the first argument is a typed array.
  bool is_typed_array;
  napi_status = napi_is_typedarray(env, js_array, &is_typed_array);
  assert(napi_status == napi_ok);

  if (!is_typed_array) {
    napi_throw_type_error(
        env, "EINVAL", "First argument should be a TypedArray");
    return NULL;
  }

  // Get information about the typed array.
  napi_typedarray_type js_array_type;
  size_t js_array_length;
  void* js_array_data;
  napi_status = napi_get_typedarray_info(env,
                                         js_array,
                                         &js_array_type,
                                         &js_array_length,
                                         &js_array_data,
                                         NULL,
                                         NULL);
  assert(napi_status == napi_ok);

  // Map to tensorflow type.
  size_t width;
  TF_DataType tf_type;

  switch (js_array_type) {
    case napi_int8_array:
      width = sizeof(int8_t);
      tf_type = TF_INT8;
      break;
    case napi_uint8_array:
    case napi_uint8_clamped_array:
      width = sizeof(uint8_t);
      tf_type = TF_UINT8;
      break;
    case napi_int16_array:
      width = sizeof(int16_t);
      tf_type = TF_INT16;
      break;
    case napi_uint16_array:
      width = sizeof(uint16_t);
      tf_type = TF_UINT16;
      break;
    case napi_int32_array:
      width = sizeof(int32_t);
      tf_type = TF_INT32;
      break;
    case napi_uint32_array:
      width = sizeof(uint32_t);
      tf_type = TF_UINT32;
    case napi_float32_array:
      width = sizeof(float);
      tf_type = TF_FLOAT;
      break;
    case napi_float64_array:
      width = sizeof(double);
      tf_type = TF_DOUBLE;
      break;
    default:
      napi_throw_type_error(env, "EINVAL", "Unsupported TypedArray type.");
      return 0;
  }

  // Build the array containing the dimensions.
  int64_t dims[kMaxDims];
  uint32_t i, num_dims;
  bool b;

  napi_status = napi_is_array(env, js_dims, &b);
  assert(napi_status == napi_ok);
  if (!b) {
    napi_throw_range_error(
        env, "EINVAL", "Second argument should be an Array");
    return NULL;
  }

  napi_status = napi_get_array_length(env, js_dims, &num_dims);
  assert(napi_status == napi_ok);
  if (num_dims < 1 || num_dims > COUNT_OF(dims)) {
    napi_throw_range_error(env, "ERANGE", "Invalid number of dimensions");
    return NULL;
  }

  for (i = 0; i < num_dims; i++) {
    napi_value element;
    int64_t value;

    napi_status = napi_get_element(env, js_dims, i, &element);
    assert(napi_status == napi_ok);

    napi_status = napi_get_value_int64(env, element, &value);
    if (napi_status == napi_number_expected) {
      napi_throw_range_error(
          env, "ERANGE", "Dimension size should be a number");
      return NULL;
    } else if (value <= 0) {
      napi_throw_range_error(env, "ERANGE", "Dimension size out of range");
      return NULL;
    }
    assert(napi_status == napi_ok);

    dims[i] = value;
  }

  // Construct the native wrap object.
  TensorWrap* tensor_wrap = new TensorWrap();
  if (tensor_wrap == NULL) {
    napi_throw_error(env, "ENOMEM", "Out of memory");
    return NULL;
  }

  // Attach native wrapper to the JavaScript object.
  tensor_wrap->env = env;
  napi_status = napi_wrap(env, js_this, tensor_wrap, DeleteTensor, NULL, NULL);
  assert(napi_status == napi_ok);

  // Store a TypedArray reference in the native wrapper. This must be done
  // before calling TF_NewTensor, because TF_NewTensor might recursively invoke
  // the ReleaseTypedArray function that clears the reference.
  napi_status =
      napi_create_reference(env, js_array, 1, &tensor_wrap->js_typed_array);
  assert(napi_status == napi_ok);

  // Construct the TF_Tensor object.
  size_t byte_length = js_array_length * width;
  TF_Tensor* tf_tensor = TF_NewTensor(tf_type,
                                      dims,
                                      num_dims,
                                      js_array_data,
                                      byte_length,
                                      ReleaseTypedArray,
                                      tensor_wrap);
  if (tf_tensor == NULL) {
    napi_throw_error(env, "ENOMEM", "Out of memory");
    return NULL;
  }
  tensor_wrap->tf_tensor = tf_tensor;

  // Create the TFE_TensorHandle object.
  TF_Status* tf_status = TF_NewStatus();
  if (tf_status == NULL) {
    napi_throw_error(env, "ENOMEM", "Out of memory");
    return NULL;
  }
  TFE_TensorHandle* tf_tensor_handle =
      TFE_NewTensorHandle(tf_tensor, tf_status);
  if (TF_GetCode(tf_status) != TF_OK) {
    napi_throw_error(env, NULL, TF_Message(tf_status));
    TF_DeleteStatus(tf_status);
    return NULL;
  }
  TF_DeleteStatus(tf_status);
  tensor_wrap->tf_tensor_handle = tf_tensor_handle;

  return js_this;
}

static napi_value TensorGetDevice(napi_env env, napi_callback_info info) {
  napi_status napi_status;

  // Fetch JavaScript `this` object.
  napi_value js_this;
  napi_status = napi_get_cb_info(env, info, NULL, NULL, &js_this, NULL);
  assert(napi_status == napi_ok);

  // Unwrap.
  TensorWrap* tensor_wrap;
  napi_status =
      napi_unwrap(env, js_this, reinterpret_cast<void**>(&tensor_wrap));
  assert(napi_status == napi_ok);

  // Ask tensorflow for the device name.
  const char* device =
      TFE_TensorHandleDeviceName(tensor_wrap->tf_tensor_handle);

  // Build JavaScript string containing the device name.
  napi_value js_device;
  napi_status =
      napi_create_string_utf8(env, device, NAPI_AUTO_LENGTH, &js_device);
  assert(napi_status == napi_ok);

  return js_device;
}

static napi_value InitBinding(napi_env env, napi_value exports) {
  napi_status status;
  napi_property_descriptor descriptor;

  // Define the Context JavaScript class.
  napi_value context_class;
  status = napi_define_class(
      env,
      "Context",                    // JavaScript class name
      NAPI_AUTO_LENGTH,             // JavasScript class name length
      NewContext,                   // Constructor
      NULL,                         // Constructor argument
      0,                            // Property count
      NULL,                         // Property descriptors
      &context_class);              // Out: js value representing the class
  assert(status == napi_ok);

  // Stick the Context class onto the exports object.
  descriptor = {"Context", NULL, NULL, NULL, NULL, context_class, napi_default,
    NULL};
  status = napi_define_properties(env, exports, 1, &descriptor);
  assert(status == napi_ok);

  // Define the Tensor JavaScript class.
  napi_value tensor_class;
  napi_property_descriptor tensor_properties[] = {
      {"device", NULL, NULL, TensorGetDevice, NULL, NULL, napi_default, NULL}};
  status = napi_define_class(
      env,
      "Tensor",                     // JavaScript class name
      NAPI_AUTO_LENGTH,             // JavasScript class name length
      NewTensor,                    // Constructor
      NULL,                         // Constructor argument
      COUNT_OF(tensor_properties),  // Property count
      tensor_properties,            // Property descriptors
      &tensor_class);               // Out: js value representing the class

  assert(status == napi_ok);

  // Stick the Tensor class onto the exports object.
  descriptor = {"Tensor", NULL, NULL, NULL, NULL, tensor_class, napi_default,
    NULL};
  status = napi_define_properties(env, exports, 1, &descriptor);
  assert(status == napi_ok);

  return exports;
}

NAPI_MODULE(tensorflow_binding, InitBinding)
