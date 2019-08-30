/**
 * Copyright (C) 2019 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file	tensor_filter_single.c
 * @date	28 Aug 2019
 * @brief	Element to use general neural network framework directly without gstreamer pipeline
 * @see	  http://github.com/nnsuite/nnstreamer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug	  No known bugs except for NYI items
 *
 * This is the main element for per-NN-framework plugins.
 * Specific implementations for each NN framework must be written
 * in each framework specific files; e.g., tensor_filter_tensorflow_lite.c
 *
 */

/**
 * SECTION:element-tensor_filter_single
 *
 * An element that invokes neural network models and their framework or
 * an independent shared object implementing tensor_filter_custom.h.
 * The input and output are always in the format of other/tensor or
 * other/tensors. This element is going to be the basis of single shot api.
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <glib.h>
#include <string.h>

#include <nnstreamer/nnstreamer_plugin_api.h>
#include <nnstreamer/tensor_typedef.h>
#include "tensor_filter_single.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!self->silent)
#endif

#define g_free_const(x) g_free((void*)(long)(x))

#define silent_debug_info(i,msg) do { \
  if (DBG) { \
    guint info_idx; \
    gchar *dim_str; \
    g_debug (msg " total %d", (i)->num_tensors); \
    for (info_idx = 0; info_idx < (i)->num_tensors; info_idx++) { \
      dim_str = gst_tensor_get_dimension_string ((i)->info[info_idx].dimension); \
      g_debug ("[%d] type=%d dim=%s", info_idx, (i)->info[info_idx].type, dim_str); \
      g_free (dim_str); \
    } \
  } \
} while (0)

/**
 * @brief Validate filter sub-plugin's data.
 */
static gboolean
nnstreamer_filter_validate (const GstTensorFilterFramework * tfsp)
{
  if (!tfsp || !tfsp->name) {
    /* invalid fw name */
    return FALSE;
  }

  if (!tfsp->invoke_NN) {
    /* no invoke function */
    return FALSE;
  }

  if (!(tfsp->getInputDimension && tfsp->getOutputDimension) &&
      !tfsp->setInputDimension) {
    /* no method to get tensor info */
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Filter's sub-plugin should call this function to register itself.
 * @param[in] tfsp Tensor-Filter Sub-Plugin to be registered.
 * @return TRUE if registered. FALSE is failed or duplicated.
 */
int
nnstreamer_filter_probe (GstTensorFilterFramework * tfsp)
{
  g_return_val_if_fail (nnstreamer_filter_validate (tfsp), FALSE);
  return register_subplugin (NNS_SUBPLUGIN_FILTER, tfsp->name, tfsp);
}

/**
 * @brief Filter's sub-plugin may call this to unregister itself.
 * @param[in] name The name of filter sub-plugin.
 */
void
nnstreamer_filter_exit (const char *name)
{
  unregister_subplugin (NNS_SUBPLUGIN_FILTER, name);
}

/**
 * @brief Find filter sub-plugin with the name.
 * @param[in] name The name of filter sub-plugin.
 * @return NULL if not found or the sub-plugin object has an error.
 */
const GstTensorFilterFramework *
nnstreamer_filter_find (const char *name)
{
  return get_subplugin (NNS_SUBPLUGIN_FILTER, name);
}


/**
 * @brief Parse the string of model
 * @param info tensors info structure
 * @param model_file the prediction model path
 * @param model_file_sub the initialize model path
 * @return number of parsed model path
 * @todo Create a struct list to save multiple model files with key, value pair
 */
static guint
gst_tensors_parse_modelpaths_string (GstTensorFilterProperties * prop,
    const gchar * model_files)
{
  gchar **models;
  gchar **model_0;
  gchar **model_1;
  guint num_models = 0;
  guint num_model_0 = 0;
  guint num_model_1 = 0;

  g_return_val_if_fail (prop != NULL, 0);

  if (model_files) {
    models = g_strsplit_set (model_files, ",", -1);
    num_models = g_strv_length (models);

    if (num_models == 1) {
      prop->model_file = g_strdup (models[0]);
    } else if (num_models == 2) {
      model_0 = g_strsplit_set (models[0], "=", -1);
      model_1 = g_strsplit_set (models[1], "=", -1);

      num_model_0 = g_strv_length (model_0);
      num_model_1 = g_strv_length (model_1);

      if (num_model_0 == 1 && num_model_1 == 1) {
        prop->model_file_sub = g_strdup (model_0[0]);
        prop->model_file = g_strdup (model_1[0]);
      } else if (g_ascii_strncasecmp (model_0[0], "init", 4) == 0 ||
          g_ascii_strncasecmp (model_0[0], "Init", 4) == 0) {
        prop->model_file_sub = g_strdup (model_0[1]);

        if (num_model_1 == 2)
          prop->model_file = g_strdup (model_1[1]);
        else
          prop->model_file = g_strdup (model_1[0]);
      } else if (g_ascii_strncasecmp (model_0[0], "pred", 4) == 0 ||
          g_ascii_strncasecmp (model_0[0], "Pred", 4) == 0) {
        prop->model_file = g_strdup (model_0[1]);

        if (num_model_1 == 2)
          prop->model_file_sub = g_strdup (model_1[1]);
        else
          prop->model_file_sub = g_strdup (model_1[0]);
      } else if (g_ascii_strncasecmp (model_1[0], "init", 4) == 0 ||
          g_ascii_strncasecmp (model_1[0], "Init", 4) == 0) {
        prop->model_file_sub = g_strdup (model_1[1]);

        if (num_model_0 == 2)
          prop->model_file = g_strdup (model_0[1]);
        else
          prop->model_file = g_strdup (model_0[0]);
      } else if (g_ascii_strncasecmp (model_1[0], "pred", 4) == 0 ||
          g_ascii_strncasecmp (model_1[0], "Pred", 4) == 0) {
        prop->model_file = g_strdup (model_1[1]);

        if (num_model_0 == 2)
          prop->model_file_sub = g_strdup (model_0[1]);
        else
          prop->model_file_sub = g_strdup (model_0[0]);
      }
      g_strfreev (model_0);
      g_strfreev (model_1);
    }
    g_strfreev (models);
  }
  return num_models;
}

/**
 * @brief Open nn framework.
 */
#define gst_tensor_filter_open_fw(filter) do { \
      if (filter->prop.fw_opened == FALSE && filter->fw) { \
        if (filter->fw->open != NULL) {\
          if (filter->fw->open (&filter->prop, &filter->privateData) == 0) \
            filter->prop.fw_opened = TRUE; \
        } else {\
          filter->prop.fw_opened = TRUE; \
        } \
      } \
    } while (0)

/**
 * @brief Close nn framework.
 */
#define gst_tensor_filter_close_fw(filter) do { \
      if (filter->prop.fw_opened) { \
        if (filter->fw && filter->fw->close) \
          filter->fw->close (&filter->prop, &filter->privateData); \
        filter->prop.fw_opened = FALSE; \
        g_free_const (filter->prop.fwname); \
        filter->prop.fwname = NULL; \
        filter->fw = NULL; \
        filter->privateData = NULL; \
      } \
    } while (0)

/**
 * @brief Invoke callbacks of nn framework. Guarantees calling open for the first call.
 */
#define gst_tensor_filter_call(filter,ret,funcname,...) do { \
      gst_tensor_filter_open_fw (filter); \
      ret = -1; \
      if (filter->prop.fw_opened && filter->fw && filter->fw->funcname) { \
        ret = filter->fw->funcname (&filter->prop, &filter->privateData, __VA_ARGS__); \
      } \
    } while (0)

/**
 * @brief GTensorFilter properties.
 */
enum
{
  PROP_0,
  PROP_SILENT,
  PROP_FRAMEWORK,
  PROP_MODEL,
  PROP_INPUT,
  PROP_INPUTTYPE,
  PROP_INPUTNAME,
  PROP_OUTPUT,
  PROP_OUTPUTTYPE,
  PROP_OUTPUTNAME,
  PROP_CUSTOM,
  PROP_SUBPLUGINS
};

#define g_tensor_filter_single_parent_class parent_class
G_DEFINE_TYPE (GTensorFilterSingle, g_tensor_filter_single, G_TYPE_OBJECT);

/* GObject vmethod implementations */
static void g_tensor_filter_single_finalize (GObject * object);
static void g_tensor_filter_single_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void g_tensor_filter_single_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);

/* GTensorFilterSingle method implementations */
static gboolean g_tensor_filter_single_invoke (GTensorFilterSingle * self,
    GstTensorMemory * input, GstTensorMemory * output);
static gboolean g_tensor_filter_input_configured (GTensorFilterSingle * self);
static gboolean g_tensor_filter_output_configured (GTensorFilterSingle * self);

/* Private functions */
static gboolean g_tensor_filter_single_start (GTensorFilterSingle * self);
static gboolean g_tensor_filter_single_stop (GTensorFilterSingle * self);

/**
 * @brief initialize the tensor_filter's class
 */
static void
g_tensor_filter_single_class_init (GTensorFilterSingleClass * klass)
{
  GObjectClass *gobject_class;

  gobject_class = (GObjectClass *) klass;

  gobject_class->set_property = g_tensor_filter_single_set_property;
  gobject_class->get_property = g_tensor_filter_single_get_property;
  gobject_class->finalize = g_tensor_filter_single_finalize;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          FALSE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_FRAMEWORK,
      g_param_spec_string ("framework", "Framework",
          "Neural network framework", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_MODEL,
      g_param_spec_string ("model", "Model filepath",
          "File path to the model file. Separated with \
          ',' in case of multiple model files(like caffe2)", "", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_INPUT,
      g_param_spec_string ("input", "Input dimension",
          "Input tensor dimension from inner array, up to 4 dimensions ?", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_INPUTNAME,
      g_param_spec_string ("inputname", "Name of Input Tensor",
          "The Name of Input Tensor", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_INPUTTYPE,
      g_param_spec_string ("inputtype", "Input tensor element type",
          "Type of each element of the input tensor ?", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_OUTPUTNAME,
      g_param_spec_string ("outputname", "Name of Output Tensor",
          "The Name of Output Tensor", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_OUTPUT,
      g_param_spec_string ("output", "Output dimension",
          "Output tensor dimension from inner array, up to 4 dimensions ?", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_OUTPUTTYPE,
      g_param_spec_string ("outputtype", "Output tensor element type",
          "Type of each element of the output tensor ?", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_CUSTOM,
      g_param_spec_string ("custom", "Custom properties for subplugins",
          "Custom properties for subplugins ?", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_SUBPLUGINS,
      g_param_spec_string ("sub-plugins", "Sub-plugins",
          "Registrable sub-plugins list", "",
          G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  klass->invoke = g_tensor_filter_single_invoke;
  klass->start = g_tensor_filter_single_start;
  klass->input_configured = g_tensor_filter_input_configured;
  klass->output_configured = g_tensor_filter_output_configured;
}

/**
 * @brief initialize the new element
 */
static void
g_tensor_filter_single_init (GTensorFilterSingle * self)
{
  GstTensorFilterProperties *prop;

  prop = &self->prop;

  /* init NNFW properties */
  prop->fwname = NULL;
  prop->fw_opened = FALSE;
  prop->input_configured = FALSE;
  prop->output_configured = FALSE;
  prop->model_file = NULL;
  prop->custom_properties = NULL;
  gst_tensors_info_init (&prop->input_meta);
  gst_tensors_info_init (&prop->output_meta);

  /* init internal properties */
  self->fw = NULL;
  self->privateData = NULL;
  self->silent = TRUE;
  self->started = FALSE;
  gst_tensors_config_init (&self->in_config);
  gst_tensors_config_init (&self->out_config);
}

/**
 * @brief Function to finalize instance.
 */
static void
g_tensor_filter_single_finalize (GObject * object)
{
  gboolean status;
  GTensorFilterSingle *self;
  GstTensorFilterProperties *prop;

  self = G_TENSOR_FILTER_SINGLE (object);

  /** stop if not already stopped */
  if (self->started == TRUE) {
    status = g_tensor_filter_single_stop (self);
    g_debug ("Tensor filter single stop status: %d", status);
  }

  prop = &self->prop;

  g_free_const (prop->fwname);
  g_free_const (prop->model_file);
  g_free_const (prop->custom_properties);

  gst_tensors_info_free (&prop->input_meta);
  gst_tensors_info_free (&prop->output_meta);

  gst_tensors_info_free (&self->in_config.info);
  gst_tensors_info_free (&self->out_config.info);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Setter for tensor_filter_single properties.
 */
static void
g_tensor_filter_single_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GTensorFilterSingle *self;
  GstTensorFilterProperties *prop;

  self = G_TENSOR_FILTER_SINGLE (object);
  prop = &self->prop;

  g_debug ("Setting property for prop %d.\n", prop_id);

  switch (prop_id) {
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      g_debug ("Debug mode = %d", self->silent);
      break;
    case PROP_FRAMEWORK:
    {
      const gchar *fw_name = g_value_get_string (value);
      const GstTensorFilterFramework *fw;

      if (self->fw != NULL) {
        /* close old framework */
        g_tensor_filter_single_stop (self);
      }

      g_debug ("Framework = %s\n", fw_name);

      fw = nnstreamer_filter_find (fw_name);

      /* See if mandatory methods are filled in */
      if (nnstreamer_filter_validate (fw)) {
        self->fw = fw;
        prop->fwname = g_strdup (fw_name);
      } else {
        g_warning ("Cannot identify the given neural network framework, %s\n",
            fw_name);
      }
      break;
    }
    case PROP_MODEL:
    {
      const gchar *model_files = g_value_get_string (value);
      guint model_num;

      if (prop->model_file) {
        g_tensor_filter_single_stop (self);
        g_free_const (prop->model_file);
        prop->model_file = NULL;
      }

      if (prop->model_file_sub) {
        g_tensor_filter_single_stop (self);
        g_free_const (prop->model_file_sub);
        prop->model_file_sub = NULL;
      }

      /* Once configures, it cannot be changed in runtime */
      /** @todo by using `gst_element_get_state()`, reject configurations in RUNNING or other states */
      g_assert (model_files);
      model_num = gst_tensors_parse_modelpaths_string (prop, model_files);
      if (model_num == 1) {
        g_debug ("Model = %s\n", prop->model_file);
        if (!g_file_test (prop->model_file, G_FILE_TEST_IS_REGULAR))
          g_critical ("Cannot find the model file: %s\n",
              prop->model_file);
      } else if (model_num == 2) {
        g_debug ("Init Model = %s\n", prop->model_file_sub);
        g_debug ("Pred Model = %s\n", prop->model_file);
        if (!g_file_test (prop->model_file_sub, G_FILE_TEST_IS_REGULAR))
          g_critical ("Cannot find the init model file: %s\n",
              prop->model_file_sub);
        if (!g_file_test (prop->model_file, G_FILE_TEST_IS_REGULAR))
          g_critical ("Cannot find the pred model file: %s\n",
              prop->model_file);
      } else if (model_num > 2) {
        /** @todo if the new NN framework requires more than 2 model files, this area will be implemented */
        g_critical (
            "There is no NN framework that requires model files more than 2. Current Input model files are :%d\n",
            model_num);
      } else {
        g_critical ("Set model file path first\n");
      }
      break;
    }
    case PROP_INPUT:
      g_assert (!prop->input_configured && value);
      /* Once configures, it cannot be changed in runtime */
      {
        guint num_dims;

        num_dims = gst_tensors_info_parse_dimensions_string (&prop->input_meta,
            g_value_get_string (value));

        if (prop->input_meta.num_tensors > 0 &&
            prop->input_meta.num_tensors != num_dims) {
          g_warning (
              "Invalid input-dim, given param does not match with old value.");
        }

        prop->input_meta.num_tensors = num_dims;
      }
      break;
    case PROP_OUTPUT:
      g_assert (!prop->output_configured && value);
      /* Once configures, it cannot be changed in runtime */
      {
        guint num_dims;

        num_dims = gst_tensors_info_parse_dimensions_string (&prop->output_meta,
            g_value_get_string (value));

        if (prop->output_meta.num_tensors > 0 &&
            prop->output_meta.num_tensors != num_dims) {
          g_warning (
              "Invalid output-dim, given param does not match with old value.");
        }

        prop->output_meta.num_tensors = num_dims;
      }
      break;
    case PROP_INPUTTYPE:
      g_assert (!prop->input_configured && value);
      /* Once configures, it cannot be changed in runtime */
      {
        guint num_types;

        num_types = gst_tensors_info_parse_types_string (&prop->input_meta,
            g_value_get_string (value));

        if (prop->input_meta.num_tensors > 0 &&
            prop->input_meta.num_tensors != num_types) {
          g_warning (
              "Invalid input-type, given param does not match with old value.");
        }

        prop->input_meta.num_tensors = num_types;
      }
      break;
    case PROP_OUTPUTTYPE:
      g_assert (!prop->output_configured && value);
      /* Once configures, it cannot be changed in runtime */
      {
        guint num_types;

        num_types = gst_tensors_info_parse_types_string (&prop->output_meta,
            g_value_get_string (value));

        if (prop->output_meta.num_tensors > 0 &&
            prop->output_meta.num_tensors != num_types) {
          g_warning (
              "Invalid output-type, given param does not match with old value.");
        }

        prop->output_meta.num_tensors = num_types;
      }
      break;
    case PROP_INPUTNAME:
      /* INPUTNAME is required by tensorflow to designate the order of tensors */
      g_assert (!prop->input_configured && value);
      /* Once configures, it cannot be changed in runtime */
      {
        guint num_names;

        num_names = gst_tensors_info_parse_names_string (&prop->input_meta,
            g_value_get_string (value));

        if (prop->input_meta.num_tensors > 0 &&
            prop->input_meta.num_tensors != num_names) {
          g_warning (
              "Invalid input-name, given param does not match with old value.");
        }

        prop->input_meta.num_tensors = num_names;
      }
      break;
    case PROP_OUTPUTNAME:
      /* OUTPUTNAME is required by tensorflow to designate the order of tensors */
      g_assert (!prop->output_configured && value);
      /* Once configures, it cannot be changed in runtime */
      {
        guint num_names;

        num_names = gst_tensors_info_parse_names_string (&prop->output_meta,
            g_value_get_string (value));

        if (prop->output_meta.num_tensors > 0 &&
            prop->output_meta.num_tensors != num_names) {
          g_warning (
              "Invalid output-name, given param does not match with old value.");
        }

        prop->output_meta.num_tensors = num_names;
      }
      break;
    case PROP_CUSTOM:
      /* In case updated custom properties in runtime! */
      g_free_const (prop->custom_properties);
      prop->custom_properties = g_value_dup_string (value);
      g_debug ("Custom Option = %s\n", prop->custom_properties);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Getter for tensor_filter_single properties.
 */
static void
g_tensor_filter_single_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GTensorFilterSingle *self;
  GstTensorFilterProperties *prop;

  self = G_TENSOR_FILTER_SINGLE (object);
  prop = &self->prop;

  g_debug ("Getting property for prop %d.\n", prop_id);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    case PROP_FRAMEWORK:
      g_value_set_string (value, prop->fwname);
      break;
    case PROP_MODEL:
      g_value_set_string (value, prop->model_file);
      break;
    case PROP_INPUT:
      if (prop->input_meta.num_tensors > 0) {
        gchar *dim_str;

        dim_str = gst_tensors_info_get_dimensions_string (&prop->input_meta);

        g_value_set_string (value, dim_str);
        g_free (dim_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_OUTPUT:
      if (prop->output_meta.num_tensors > 0) {
        gchar *dim_str;

        dim_str = gst_tensors_info_get_dimensions_string (&prop->output_meta);

        g_value_set_string (value, dim_str);
        g_free (dim_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_INPUTTYPE:
      if (prop->input_meta.num_tensors > 0) {
        gchar *type_str;

        type_str = gst_tensors_info_get_types_string (&prop->input_meta);

        g_value_set_string (value, type_str);
        g_free (type_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_OUTPUTTYPE:
      if (prop->output_meta.num_tensors > 0) {
        gchar *type_str;

        type_str = gst_tensors_info_get_types_string (&prop->output_meta);

        g_value_set_string (value, type_str);
        g_free (type_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_INPUTNAME:
      if (prop->input_meta.num_tensors > 0) {
        gchar *name_str;

        name_str = gst_tensors_info_get_names_string (&prop->input_meta);

        g_value_set_string (value, name_str);
        g_free (name_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_OUTPUTNAME:
      if (prop->output_meta.num_tensors > 0) {
        gchar *name_str;

        name_str = gst_tensors_info_get_names_string (&prop->output_meta);

        g_value_set_string (value, name_str);
        g_free (name_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_CUSTOM:
      g_value_set_string (value, prop->custom_properties);
      break;
    case PROP_SUBPLUGINS:
    {
      GString *subplugins;
      subplugin_info_s sinfo;
      guint i, total;

      subplugins = g_string_new (NULL);

      /* add custom */
      g_string_append (subplugins, "custom");

      total = nnsconf_get_subplugin_info (NNSCONF_PATH_FILTERS, &sinfo);

      if (total > 0) {
        const gchar *prefix_str;
        gsize prefix, len;

        prefix_str = nnsconf_get_subplugin_name_prefix (NNSCONF_PATH_FILTERS);
        prefix = strlen (prefix_str);

        for (i = 0; i < total; ++i) {
          g_string_append (subplugins, ",");

          /* supposed .so files only */
          len = strlen (sinfo.names[i]) - prefix - 3;
          g_string_append_len (subplugins, sinfo.names[i] + prefix, len);
        }
      }

      g_value_take_string (value, g_string_free (subplugins, FALSE));
      break;
    }
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Printout the comparison results of two tensors.
 * @param[in] info1 The tensors to be shown on the left hand side
 * @param[in] info2 The tensors to be shown on the right hand side
 * @todo If this is going to be used by other elements, move this to nnstreamer/tensor_common.
 */
static void
gst_tensor_filter_compare_tensors (GstTensorsInfo * info1,
    GstTensorsInfo * info2)
{
  gchar *result = NULL;
  gchar *line, *tmp, *left, *right;
  guint i;

  for (i = 0; i < NNS_TENSOR_SIZE_LIMIT; i++) {
    if (info1->num_tensors <= i && info2->num_tensors <= i)
      break;

    if (info1->num_tensors > i) {
      tmp = gst_tensor_get_dimension_string (info1->info[i].dimension);
      left = g_strdup_printf ("%s [%s]",
          gst_tensor_get_type_string (info1->info[i].type), tmp);
      g_free (tmp);
    } else {
      left = g_strdup ("None");
    }

    if (info2->num_tensors > i) {
      tmp = gst_tensor_get_dimension_string (info2->info[i].dimension);
      right = g_strdup_printf ("%s [%s]",
          gst_tensor_get_type_string (info2->info[i].type), tmp);
      g_free (tmp);
    } else {
      right = g_strdup ("None");
    }

    line =
        g_strdup_printf ("%2d : %s | %s %s\n", i, left, right,
        g_str_equal (left, right) ? "" : "FAILED");

    g_free (left);
    g_free (right);

    if (result) {
      tmp = g_strdup_printf ("%s%s", result, line);
      g_free (result);
      g_free (line);

      result = tmp;
    } else {
      result = line;
    }
  }

  if (result) {
    /* print warning message */
    g_warning ("Tensor info :\n%s", result);
    g_free (result);
  }
}

/**
 * @brief Determine if input is configured
 * (both input and output tensor)
 */
static gboolean
g_tensor_filter_input_configured (GTensorFilterSingle * self)
{
  if (self->prop.input_configured)
    return TRUE;
  else
    return FALSE;
}

/**
 * @brief Determine if output is configured
 * (both input and output tensor)
 */
static gboolean
g_tensor_filter_output_configured (GTensorFilterSingle * self)
{
  if (self->prop.output_configured)
    return TRUE;
  else
    return FALSE;
}

/**
 * @brief Load tensor info from NN model.
 * (both input and output tensor)
 */
static void
g_tensor_filter_load_tensor_info (GTensorFilterSingle * self)
{
  GstTensorFilterProperties *prop;
  GstTensorsInfo in_info, out_info;
  int res;

  prop = &self->prop;

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  /* supposed fixed in-tensor info if getInputDimension is defined. */
  if (!prop->input_configured) {
    gst_tensor_filter_call (self, res, getInputDimension, &in_info);

    if (res == 0) {
      g_assert (in_info.num_tensors > 0);

      /** if set-property called and already has info, verify it! */
      if (prop->input_meta.num_tensors > 0) {
        if (!gst_tensors_info_is_equal (&in_info, &prop->input_meta)) {
          g_critical ("The input tensor is not compatible.");
          gst_tensor_filter_compare_tensors (&in_info, &prop->input_meta);
          goto done;
        }
      } else {
        gst_tensors_info_copy (&prop->input_meta, &in_info);
      }

      prop->input_configured = TRUE;
      silent_debug_info (&in_info, "input tensor");
    }
  }

  /* supposed fixed out-tensor info if getOutputDimension is defined. */
  if (!prop->output_configured) {
    gst_tensor_filter_call (self, res, getOutputDimension, &out_info);

    if (res == 0) {
      g_assert (out_info.num_tensors > 0);

      /** if set-property called and already has info, verify it! */
      if (prop->output_meta.num_tensors > 0) {
        if (!gst_tensors_info_is_equal (&out_info, &prop->output_meta)) {
          g_critical ("The output tensor is not compatible.");
          gst_tensor_filter_compare_tensors (&out_info, &prop->output_meta);
          goto done;
        }
      } else {
        gst_tensors_info_copy (&prop->output_meta, &out_info);
      }

      prop->output_configured = TRUE;
      silent_debug_info (&out_info, "output tensor");
    }
  }

done:
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

/**
 * @brief Called when the element starts processing, if fw not laoded
 * @param self "this" pointer
 * @return TRUE if there is no error.
 */
static gboolean
g_tensor_filter_single_start (GTensorFilterSingle * self)
{
  /** open framework, load model */
  if (self->fw == NULL)
    return FALSE;

  if (self->prop.fw_opened == FALSE && self->fw) {
    if (self->fw->open != NULL) {
      if (self->fw->open (&self->prop, &self->privateData) == 0)
        self->prop.fw_opened = TRUE;
    } else {
      self->prop.fw_opened = TRUE;
    }
  }

  if (self->prop.fw_opened == FALSE)
    return FALSE;

  g_tensor_filter_load_tensor_info (self);

  return TRUE;
}

/**
 * @brief Called when the element stops processing, if fw loaded
 * @param self "this" pointer
 * @return TRUE if there is no error.
 */
static gboolean
g_tensor_filter_single_stop (GTensorFilterSingle * self)
{
  /** close framework, unload model */
  if (self->prop.fw_opened) {
    if (self->fw && self->fw->close) {
      self->fw->close (&self->prop, &self->privateData);
    }
    self->prop.fw_opened = FALSE;
    g_free_const (self->prop.fwname);
    self->prop.fwname = NULL;
    self->fw = NULL;
    self->privateData = NULL;
  }
  return TRUE;
}


/**
 * @brief Called when an input supposed to be invoked
 * @param self "this" pointer
 * @param input memory containing input data to run processing on
 * @param output memory to put output data into after processing
 * @return TRUE if there is no error.
 */
static gboolean
g_tensor_filter_single_invoke (GTensorFilterSingle * self,
    GstTensorMemory * input, GstTensorMemory * output)
{
  gboolean status;
  int i;

  if (G_UNLIKELY (!self->fw) || G_UNLIKELY (!self->fw->invoke_NN))
    return FALSE;
  if (G_UNLIKELY (!self->fw->run_without_model) &&
      G_UNLIKELY (!self->prop.model_file))
    return FALSE;

  /** start if not already started */
  if (self->started == FALSE) {
    status = g_tensor_filter_single_start (self);
    if (status == FALSE) {
      return status;
    }
    self->started = TRUE;
  }

  /** Setup output buffer */
  for (i = 0; i < self->prop.output_meta.num_tensors; i++) {
    /* allocate memory if allocate_in_invoke is FALSE */
    if (self->fw->allocate_in_invoke == FALSE) {
      output[i].data = g_malloc (output[i].size);
      if (!output[i].data)
        goto error;
    }
  }

  status = self->fw->invoke_NN (&self->prop, &self->privateData, input, output);

  if (status == 0)
    return TRUE;

  return FALSE;

error:
  for (i = 0; i < self->prop.output_meta.num_tensors; i++)
    g_free (output[i].data);
  return FALSE;
}
