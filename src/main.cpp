#include <Arduino.h>
#include <TensorFlowLite.h>
#include <Arduino_OV767X.h>

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model.h"

namespace
{
  tflite::ErrorReporter *error_reporter = nullptr;
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;
  TfLiteTensor *output = nullptr;
  constexpr int kTensorArenaSize = 136 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

TfLiteStatus GetImage(tflite::ErrorReporter *error_reporter/*, int image_width,
                      int image_height, int channels*/, int8_t *image_data)
{

  byte data[176 * 144]; // Receiving QCIF grayscale from camera = 176 * 144 * 1

  static bool g_is_camera_initialized = false;
  static bool serial_is_initialized = false;

  // Initialize camera if necessary
  if (!g_is_camera_initialized)
  {
    if (!Camera.begin(QCIF, GRAYSCALE, 5))
    {
      TF_LITE_REPORT_ERROR(error_reporter, "Failed to initialize camera!");
      return kTfLiteError;
    }
    g_is_camera_initialized = true;
  }

  // Read camera data
  Camera.readFrame(data);

  int min_x = (176 - 96) / 2;
  int min_y = (144 - 96) / 2;
  int index = 0;

  // Crop 96x96 image. This lowers FOV, ideally we would downsample but this is simpler.
  for (int y = min_y; y < min_y + 96; y++)
  {
    for (int x = min_x; x < min_x + 96; x++)
    {
      image_data[index++] = static_cast<int8_t>(data[(y * 176) + x] - 128); // convert TF input image to signed 8-bit
    }
  }
  return kTfLiteOk;
}

void setup()
{
  Serial.begin(9600);
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(converted_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
   TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }
  static tflite::MicroMutableOpResolver<3> micro_op_resolver(error_reporter);
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddConv2D() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddMaxPool2D() != kTfLiteOk)
  {
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);

  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
  input = interpreter->input(0);
}

void loop()
{
  if (Serial.available())
  {
    char c = Serial.read();
    if (c == '1')
    {
      // Get image from provider.
      if (kTfLiteOk != GetImage(error_reporter/*, kNumCols, kNumRows, kNumChannels*/,
                                input->data.int8))
      {
        TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
      }

      // Run the model on this input and make sure it succeeds.
      if (kTfLiteOk != interpreter->Invoke())
      {
        TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
      }

      output = interpreter->output(0);
      int8_t score = output->data.uint8[0];
      if (score > 0.5)
      {
        Serial.println("It's a cat");
      }
      else
      {
        Serial.println("It's a person");
      }
    } else {
      Serial.println("No capture");
    }
  }
}