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
#include <node_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <string>
#include "deps/libtensorflow/include/tensorflow/c/c_api.h"
#include "deps/libtensorflow/include/tensorflow/c/eager/c_api.h"

#define COUNT_OF(array) (sizeof(array) / sizeof(array[0]))

enum AttrType {
  ATTR_STRING,
  ATTR_INT,
  ATTR_FLOAT,
  ATTR_BOOL,
  ATTR_TYPE,
  ATTR_SHAPE,
  ATTR_FUNCTION,
  ATTR_STRING_LIST,
  ATTR_INT_LIST,
  ATTR_FLOAT_LIST,
  ATTR_BOOL_LIST,
  ATTR_TYPE_LIST,
  ATTR_SHAPE_LIST,
};

static const size_t kMaxDims = 10;
static napi_ref tensor_class_ref;

struct ContextWrap {
  napi_env env;
  TFE_Context* tf_context;
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

void AssertConstructorCall(napi_env env, napi_callback_info info) {
#ifdef DEBUG
  napi_value js_target;
  auto napi_status = napi_get_new_target(env, info, &js_target);
  assert(napi_status == napi_ok);
  assert(js_target != NULL && "Function not used as a constructor");
#endif
}

// TFE_OpSetAttrType, TFE_OpSetAttrBool, and friends use attr_name parameter
// beyond the lifetime of the call. Because we are getting these strings
// from V8, we cannot simply get a constant pointer. So, rather than trying
// to dynamically allocate memory for these attribute names, we instead have
// this lookup map to fixed strings.  The value (int) in the map is unused.
// Add to this as needed.
const std::map<std::string, int> attrNameMap = {
    {"T", 0}, {"transpose_a", 0}, {"transpose_b", 0},
};

const char* AttrNameLookup(napi_env env, napi_value attr_name_js) {
  char attr_name[512];
  auto status =
      napi_get_value_string_utf8(env, attr_name_js, attr_name, 512, NULL);
  assert(status == napi_ok);

  auto it = attrNameMap.find(attr_name);
  if (it != attrNameMap.end()) {
    return it->first.c_str();
  }
  assert(false && "attr_name not found in attrNameMap");
}

void SetOpAttr(napi_env env, TFE_Op* op, napi_value attr) {
  // Check that the attr is an array.
  bool is_array;
  auto status = napi_is_array(env, attr, &is_array);
  assert(status == napi_ok && is_array);
  // Get length.
  uint32_t attr_len;
  status = napi_get_array_length(env, attr, &attr_len);
  assert(status == napi_ok);
  assert(attr_len >= 3);

  // attr[0] should be the name e.g. "transpose_a"
  napi_value attr_name_js;
  status = napi_get_element(env, attr, 0, &attr_name_js);
  assert(status == napi_ok);
  const char* attr_name = AttrNameLookup(env, attr_name_js);

  // attr[1] should be an integer in enum AttrType.
  napi_value attr_type_js;
  status = napi_get_element(env, attr, 1, &attr_type_js);
  assert(status == napi_ok);
  enum AttrType attr_type;
  status = napi_get_value_int32(
      env, attr_type_js, reinterpret_cast<int32_t*>(&attr_type));
  assert(status == napi_ok);

  napi_value attr2;
  status = napi_get_element(env, attr, 2, &attr2);
  assert(status == napi_ok);

  switch (attr_type) {
    case ATTR_BOOL: {
      bool v;
      status = napi_get_value_bool(env, attr2, &v);
      assert(status == napi_ok);
      TFE_OpSetAttrBool(op, attr_name, v);
      break;
    }

    case ATTR_TYPE: {
      TF_DataType v;
      status =
          napi_get_value_int32(env, attr2, reinterpret_cast<int32_t*>(&v));
      assert(status == napi_ok);
      TFE_OpSetAttrType(op, attr_name, v);
      break;
    }

    default:
      assert(false && "implement me");
  }
}

/* Set attribtues from arguments. Attrs will look something like this:
    [
      ["transpose_a", binding.ATTR_BOOL, false],
      ["transpose_b", binding.ATTR_BOOL, false],
      ["T", binding.ATTR_TYPE, binding.TF_FLOAT],
    ]
*/
void SetOpAttrs(napi_env env, TFE_Op* op, napi_value attrs) {
  uint32_t attrs_len;
  auto status = napi_get_array_length(env, attrs, &attrs_len);
  assert(status == napi_ok);

  for (uint32_t i = 0; i < attrs_len; ++i) {
    // Each element of the attrs list should be an array.
    napi_value attr;
    status = napi_get_element(env, attrs, i, &attr);
    assert(status == napi_ok);
    SetOpAttr(env, op, attr);
  }
}

static napi_value Execute(napi_env env, napi_callback_info info) {
  // Fetch JavaScript `this` object and function arguments.
  size_t argc = 4;
  napi_value args[4];
  napi_value js_this;
  auto napi_status = napi_get_cb_info(env, info, &argc, args, &js_this, NULL);
  assert(napi_status == napi_ok);

  // Get ContextWrap from args[0].
  ContextWrap* context_wrap;
  napi_status =
      napi_unwrap(env, args[0], reinterpret_cast<void**>(&context_wrap));
  assert(napi_status == napi_ok);

  // Get op_name (const char*) from args[1].
  char op_name[512];
  napi_status = napi_get_value_string_utf8(env, args[1], op_name, 512, NULL);
  assert(napi_status == napi_ok);

  // Get attrs from args[2].
  auto attrs = args[2];
  bool is_array;
  napi_status = napi_is_array(env, attrs, &is_array);
  assert(napi_status == napi_ok && is_array);

  // Get inputs from args[3].
  auto inputs = args[3];
  napi_status = napi_is_array(env, inputs, &is_array);
  assert(napi_status == napi_ok && is_array);
  uint32_t inputs_len;
  napi_status = napi_get_array_length(env, inputs, &inputs_len);
  assert(napi_status == napi_ok);

  // Create TFE_Op
  auto tf_status = TF_NewStatus();
  TFE_Op* op = TFE_NewOp(context_wrap->tf_context, op_name, tf_status);
  assert(TF_GetCode(tf_status) == TF_OK);

  SetOpAttrs(env, op, attrs);

  // Loop thru inputs and add them to Op.
  for (uint32_t i = 0; i < inputs_len; ++i) {
    napi_value input;
    napi_status = napi_get_element(env, inputs, i, &input);
    assert(napi_status == napi_ok);

    TensorWrap* tensor_wrap;
    napi_status =
        napi_unwrap(env, input, reinterpret_cast<void**>(&tensor_wrap));
    assert(napi_status == napi_ok);

    TFE_OpAddInput(op, tensor_wrap->tf_tensor_handle, tf_status);
    assert(TF_GetCode(tf_status) == TF_OK);
  }

  // TODO(ry) only handling a single return value currently.
  TFE_TensorHandle* retvals[1];
  int num_retvals = 1;
  TFE_Execute(op, retvals, &num_retvals, tf_status);
  if (TF_GetCode(tf_status) != TF_OK) {
    napi_throw_error(env, NULL, TF_Message(tf_status));
    return NULL;
  }

  // Create array to be returned.
  napi_value js_retvals;
  napi_status = napi_create_array_with_length(env, num_retvals, &js_retvals);
  assert(napi_status == napi_ok);

  // Get reference to Tensor class so we can call its constructor.
  napi_value tensor_class;
  napi_status = napi_get_reference_value(env, tensor_class_ref, &tensor_class);
  assert(napi_status == napi_ok);

  // For each retval, wrap the TensorHandle.
  for (int i = 0; i < num_retvals; ++i) {
    napi_value js_retval;
    // Create a new Tensor object.
    napi_status = napi_new_instance(env, tensor_class, 0, NULL, &js_retval);
    assert(napi_status == napi_ok);
    // Unwrap
    TensorWrap* tensor_wrap;
    napi_status =
        napi_unwrap(env, js_retval, reinterpret_cast<void**>(&tensor_wrap));
    assert(napi_status == napi_ok);
    assert(tensor_wrap->env == env);
    tensor_wrap->tf_tensor_handle = retvals[i];
    // Set created js object in output array.
    napi_status = napi_set_element(env, js_retvals, (uint32_t) i, js_retval);
    assert(napi_status == napi_ok);
  }

  TFE_DeleteOp(op);
  TF_DeleteStatus(tf_status);
  return js_retvals;
}

static void DeleteContext(napi_env env, void* wrap_ptr, void* hint) {
  auto wrap = static_cast<ContextWrap*>(wrap_ptr);
  auto tf_status = TF_NewStatus();
  assert(tf_status);
  TFE_DeleteContext(wrap->tf_context, tf_status);
  assert(TF_GetCode(tf_status) == TF_OK);
  delete wrap;
  TF_DeleteStatus(tf_status);
}

static napi_value NewContext(napi_env env, napi_callback_info info) {
  napi_value js_this;

  AssertConstructorCall(env, info);

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

  napi_status =
      napi_wrap(env, js_this, context_wrap, DeleteContext, NULL, NULL);
  assert(napi_status == napi_ok);

  return js_this;
}

static napi_value NewTensor(napi_env env, napi_callback_info info) {
  napi_status napi_status;

  AssertConstructorCall(env, info);

  // Fetch JavaScript `this` object and function arguments.
  size_t argc = 2;
  napi_value args[2];
  napi_value js_this;
  napi_status = napi_get_cb_info(env, info, &argc, args, &js_this, NULL);
  assert(napi_status == napi_ok);

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

  // Execute calls the Tensor constructor without any arguments.
  if (argc == 0) {
    tensor_wrap->tf_tensor_handle = NULL;
    tensor_wrap->tf_tensor = NULL;
    return js_this;
  }

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

static void DeleteTensorArrayBuffer(napi_env env,
                                    void* tensor_wrap_ptr,
                                    void* hint) {
  auto tensor_wrap = reinterpret_cast<TensorWrap*>(hint);
  // If this tensor's data originates from JavaScript, then it will
  // be freed by ReleaseTypedArray. If not, we should delete the Tensor.
  if (tensor_wrap->js_typed_array == NULL) {
    assert(tensor_wrap->tf_tensor != NULL);
    TF_DeleteTensor(tensor_wrap->tf_tensor);
    tensor_wrap->tf_tensor = NULL;
  }
}

static napi_value TensorAsArrayBuffer(napi_env env, napi_callback_info info) {
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

  // Resolve TFE_TensorHandle into TF_Tensor
  auto tf_status = TF_NewStatus();
  auto tensor =
      TFE_TensorHandleResolve(tensor_wrap->tf_tensor_handle, tf_status);
  assert(TF_GetCode(tf_status) == TF_OK);
  TF_DeleteStatus(tf_status);

  assert(tensor_wrap->tf_tensor == NULL || tensor_wrap->tf_tensor == tensor);
  tensor_wrap->tf_tensor = tensor;

  void* external_data = TF_TensorData(tensor);
  size_t byte_length = TF_TensorByteSize(tensor);
  napi_value array_buffer;
  napi_create_external_arraybuffer(env,
                                   external_data,
                                   byte_length,
                                   DeleteTensorArrayBuffer,
                                   tensor_wrap,
                                   &array_buffer);
  assert(napi_status == napi_ok);

  return array_buffer;
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

void AssignIntProperty(napi_env env,
                       napi_value exports,
                       const char* name,
                       int32_t value) {
  napi_value js_value;
  auto status = napi_create_int32(env, value, &js_value);
  assert(status == napi_ok);
  napi_property_descriptor d = {
      name, NULL, NULL, NULL, NULL, js_value, napi_default, NULL};
  status = napi_define_properties(env, exports, 1, &d);
  assert(status == napi_ok);
}

static napi_value InitBinding(napi_env env, napi_value exports) {
  napi_status status;

  // Define the Context JavaScript class.
  napi_value context_class;
  status = napi_define_class(
      env,
      "Context",         // JavaScript class name
      NAPI_AUTO_LENGTH,  // JavasScript class name length
      NewContext,        // Constructor
      NULL,              // Constructor argument
      0,                 // Property count
      NULL,              // Property descriptors
      &context_class);   // Out: js value representing the class
  assert(status == napi_ok);

  // Define the Tensor JavaScript class.
  napi_value tensor_class;
  napi_property_descriptor tensor_properties[] = {
      {"asArrayBuffer",
       NULL,
       TensorAsArrayBuffer,
       NULL,
       NULL,
       NULL,
       napi_default,
       NULL},
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

  // tensor_class is used Execute() to instanciate resulting Tensors. Thus
  // create a reference.
  status = napi_create_reference(env, tensor_class, 1, &tensor_class_ref);
  assert(status == napi_ok);

  // Fill the exports.
  napi_property_descriptor exports_properties[] = {
      {"Context", NULL, NULL, NULL, NULL, context_class, napi_default, NULL},
      {"execute", NULL, Execute, NULL, NULL, NULL, napi_default, NULL},
      {"Tensor", NULL, NULL, NULL, NULL, tensor_class, napi_default, NULL},
  };
  status = napi_define_properties(
      env, exports, COUNT_OF(exports_properties), exports_properties);
  assert(status == napi_ok);

#define EXPORT_ENUM(v) AssignIntProperty(env, exports, #v, v)
  // TF_DataType
  EXPORT_ENUM(TF_FLOAT);
  EXPORT_ENUM(TF_DOUBLE);
  EXPORT_ENUM(TF_INT32);
  EXPORT_ENUM(TF_UINT8);
  EXPORT_ENUM(TF_INT16);
  EXPORT_ENUM(TF_INT8);
  EXPORT_ENUM(TF_STRING);
  EXPORT_ENUM(TF_COMPLEX64);
  EXPORT_ENUM(TF_COMPLEX);
  EXPORT_ENUM(TF_INT64);
  EXPORT_ENUM(TF_BOOL);
  EXPORT_ENUM(TF_QINT8);
  EXPORT_ENUM(TF_QUINT8);
  EXPORT_ENUM(TF_QINT32);
  EXPORT_ENUM(TF_BFLOAT16);
  EXPORT_ENUM(TF_QINT16);
  EXPORT_ENUM(TF_QUINT16);
  EXPORT_ENUM(TF_UINT16);
  EXPORT_ENUM(TF_COMPLEX128);
  EXPORT_ENUM(TF_HALF);
  EXPORT_ENUM(TF_RESOURCE);
  EXPORT_ENUM(TF_VARIANT);
  EXPORT_ENUM(TF_UINT32);
  EXPORT_ENUM(TF_UINT64);
  // AttrType
  EXPORT_ENUM(ATTR_STRING);
  EXPORT_ENUM(ATTR_INT);
  EXPORT_ENUM(ATTR_FLOAT);
  EXPORT_ENUM(ATTR_BOOL);
  EXPORT_ENUM(ATTR_TYPE);
  EXPORT_ENUM(ATTR_SHAPE);
  EXPORT_ENUM(ATTR_FUNCTION);
  EXPORT_ENUM(ATTR_STRING_LIST);
  EXPORT_ENUM(ATTR_INT_LIST);
  EXPORT_ENUM(ATTR_FLOAT_LIST);
  EXPORT_ENUM(ATTR_BOOL_LIST);
  EXPORT_ENUM(ATTR_TYPE_LIST);
  EXPORT_ENUM(ATTR_SHAPE_LIST);
#undef EXPORT_ENUM

  return exports;
}

NAPI_MODULE(tensorflow_binding, InitBinding)
