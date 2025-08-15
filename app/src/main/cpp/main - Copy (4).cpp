// %BANNER_BEGIN%
// ---------------------------------------------------------------------
// %COPYRIGHT_BEGIN%
// Copyright (c) 2022 Magic Leap, Inc. All Rights Reserved.
// Use of this file is governed by the Software License Agreement,
// located here: https://www.magicleap.com/software-license-agreement-ml2
// Terms and conditions applicable to third-party materials accompanying
// this distribution may also be found in the top-level NOTICE file
// appearing herein.
// %COPYRIGHT_END%
// ---------------------------------------------------------------------
// %BANNER_END%

#define ALOG_TAG "com.magicleap.capi.sample.camera_mixed_reality"

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <sstream>
#include <thread>
#ifdef ML_WINDOWS
#include <direct.h>
#define mkdir(a, b) _mkdir(a)
#else
#include <sys/stat.h>
#endif

#include <app_framework/application.h>
#include <app_framework/geometry/quad_mesh.h>
#include <app_framework/gui.h>
#include <app_framework/input/input_command_handler.h>
#include <app_framework/input/ml_input_handler.h>
#include <app_framework/logging.h>
#include <app_framework/material/textured_material.h>
#include <app_framework/registry.h>
#include <ml_camera_v2.h>
#include <ml_media_error.h>
#include <ml_media_format.h>
#include <ml_media_recorder.h>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include <iostream>
#ifdef ML_LUMIN
#include <EGL/egl.h>
#define EGL_EGLEXT_PROTOTYPES
#include <EGL/eglext.h>
#endif

#define UNWRAP_RET_MEDIARESULT(res) UNWRAP_RET_MLRESULT_GENERIC(res, UNWRAP_MLMEDIA_RESULT);

namespace EnumHelpers {
    const char *GetMLCameraErrorString(const MLCameraError &err) {
        switch (err) {
            case MLCameraError::MLCameraError_None: return "";
            case MLCameraError::MLCameraError_Invalid: return "Invalid/Unknown error";
            case MLCameraError::MLCameraError_Disabled: return "Camera disabled";
            case MLCameraError::MLCameraError_DeviceFailed: return "Camera device failed";
            case MLCameraError::MLCameraError_ServiceFailed: return "Camera service failed";
            case MLCameraError::MLCameraError_CaptureFailed: return "Capture failed";
            default: return "Invalid MLCameraError value!";
        }
    }

    const char *GetMLCameraDisconnectReasonString(const MLCameraDisconnectReason &reason) {
        switch (reason) {
            case MLCameraDisconnectReason::MLCameraDisconnect_DeviceLost: return "Device lost";
            case MLCameraDisconnectReason::MLCameraDisconnect_PriorityLost: return "Priority lost";
            default: return "Invalid MLCameraDisconnectReason value!";
        }
    }
}  // namespace EnumHelpers

using namespace ml::app_framework;
using namespace std::chrono_literals;

class CameraMixedRealityApp : public Application {
public:
    std::string onnx_status_message_;   // To store ONNX init result for GUI
    bool onnx_initialized_ = false;     // Flag to ensure one-time init
    void InitializeONNX();             // Declaration of the new method

    bool send_to_vlm_after_capture_ = false;  // Flag to indicate VLM sending
    CameraMixedRealityApp(struct android_app *state)
            : Application(state, std::vector<std::string>{"android.permission.CAMERA", "android.permission.RECORD_AUDIO"},
                          USE_GUI),
              recorder_camera_device_available_(false),
              capture_width_(0),
              capture_height_(0),
              recorder_camera_context_(ML_INVALID_HANDLE),
              default_output_filepath_(GetExternalFilesDir() + "/captures/"),
              default_output_filename_photo_("mr_dk_camera_photo_output"),
              entered_standby_(false) {}

    void OnStart() override {
        mkdir(default_output_filepath_.c_str(), 0755);
    }

    void OnResume() override {
        if (ArePermissionsGranted()) {
            GetGui().Show();
            SetupRestrictedResources();
        }
    }

    void OnStop() override {
        UNWRAP_MLRESULT(DestroyCamera());
    }

    void OnDestroy() override {
        for (auto &t : standby_helper_threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
        standby_helper_threads_.clear();
        UNWRAP_MLRESULT(DestroyCamera());
    }

    void OnUpdate(float delta_time_sec) override {
        UpdateGui();
    }

private:
    void SendImageToVLM(const std::string& imagePath) {
        // Implement your VLM sending logic here
        ALOGI("Sending image to VLM: %s", imagePath.c_str());
        InitializeONNX();
        // (If you plan to actually use ONNX for inference, you would load the model
    }

    void SetupRestrictedResources() {
        if (entered_standby_) {
            UNWRAP_MLRESULT(DestroyCamera());
            entered_standby_ = false;
        }
        ASSERT_MLRESULT(SetupCamera());
        ASSERT_MLRESULT(SetupCaptureSize());
    }

    void UpdateGui() {
        auto &gui = GetGui();
        gui.BeginUpdate();
        bool is_running = true;

        if (gui.BeginDialog("Camera Capture", &is_running,
                            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize |
                            ImGuiWindowFlags_NoCollapse)) {
            ImGui::Text("Capture Options:");

            if (ImGui::Button("Capture and Send to VLM")) {
                send_to_vlm_after_capture_ = true;
                UNWRAP_MLRESULT(CaptureImage());
                InitializeONNX();  // Call ONNX init when button is pressed
            }

            if (ImGui::Button("Capture Photo")) {
                send_to_vlm_after_capture_ = false;
                UNWRAP_MLRESULT(CaptureImage());
            }

            ImGui::NewLine();
            ImGui::Separator();
            ImGui::NewLine();
            ImGui::Text("Last photo info:");
            ImGui::Text("\tFilename: \"%s\"", current_filename_photo_.c_str());
            if (!onnx_status_message_.empty()) {
                ImGui::Text("ONNX status:");
                ImGui::Text("\t%s", onnx_status_message_.c_str());
            }
        }
        gui.EndDialog();
        gui.EndUpdate();

        if (!is_running) {
            FinishActivity();
        }
    }

    static void OnImageAvailable(const MLCameraOutput *output, const MLHandle metadata_handle,
                                 const MLCameraResultExtras *extra, void *data) {
        CameraMixedRealityApp *this_app = reinterpret_cast<CameraMixedRealityApp *>(data);
        if (this_app) {
            const std::string k_file_ext = ".jpg";
            this_app->current_filename_photo_ =
                    this_app->default_output_filename_photo_ + std::to_string(extra->vcam_timestamp) + k_file_ext;
            const std::string output_filename = this_app->default_output_filepath_ + this_app->current_filename_photo_;

            ALOGI("Image output filename: %s", output_filename.c_str());
            auto opened_output_file = fopen(output_filename.c_str(), "wb");
            if (opened_output_file) {
                fwrite(output->planes[0].data, output->planes[0].size, 1, opened_output_file);
                fclose(opened_output_file);

                // If the flag is set, send the image to VLM
                if (this_app->send_to_vlm_after_capture_) {
                    this_app->SendImageToVLM(output_filename);
                }
            } else {
                ALOGE("Failed to open %s, with error: %s!", output_filename.c_str(), strerror(errno));
            }
        }
    }

    MLResult CaptureImage() {
        MLHandle metadata_handle = ML_INVALID_HANDLE;
        MLCameraCaptureConfig config = {};
        MLCameraCaptureConfigInit(&config);
        config.stream_config[0].capture_type = MLCameraCaptureType_Image;
        config.stream_config[0].width = capture_width_;
        config.stream_config[0].height = capture_height_;
        config.stream_config[0].output_format = MLCameraOutputFormat_JPEG;
        config.stream_config[0].native_surface_handle = ML_INVALID_HANDLE;
        config.capture_frame_rate = MLCameraCaptureFrameRate_None;
        config.num_streams = 1;
        UNWRAP_RET_MEDIARESULT(MLCameraPrepareCapture(recorder_camera_context_, &config, &metadata_handle));
        UNWRAP_MLMEDIA_RESULT(MLCameraPreCaptureAEAWB(recorder_camera_context_));
        UNWRAP_RET_MEDIARESULT(MLCameraCaptureImage(recorder_camera_context_, 1));
        return MLResult_Ok;
    }

    MLResult DestroyCamera() {
        if (MLHandleIsValid(recorder_camera_context_)) {
            UNWRAP_RET_MEDIARESULT(MLCameraDisconnect(recorder_camera_context_));
            recorder_camera_context_ = ML_INVALID_HANDLE;
            recorder_camera_device_available_ = false;
        }
        UNWRAP_RET_MEDIARESULT(MLCameraDeInit());
        return MLResult_Ok;
    }

    MLResult SetupCamera() {
        if (MLHandleIsValid(recorder_camera_context_)) {
            return MLResult_Ok;
        }
        MLCameraDeviceAvailabilityStatusCallbacks device_availability_status_callbacks = {};
        MLCameraDeviceAvailabilityStatusCallbacksInit(&device_availability_status_callbacks);
        device_availability_status_callbacks.on_device_available = [](const MLCameraDeviceAvailabilityInfo *avail_info) {
            CheckDeviceAvailability(avail_info, true);
        };
        device_availability_status_callbacks.on_device_unavailable = [](const MLCameraDeviceAvailabilityInfo *avail_info) {
            CheckDeviceAvailability(avail_info, false);
        };

        UNWRAP_RET_MEDIARESULT(MLCameraInit(&device_availability_status_callbacks, this));
        {  // wait for maximum 2 seconds until camera becomes available
            std::unique_lock<std::mutex> lock(camera_device_available_lock_);
            camera_device_available_condition_.wait_for(lock, 2000ms, [&]() { return recorder_camera_device_available_; });
        }

        if (!recorder_camera_device_available_) {
            ALOGE("Timed out waiting for Main camera!");
            return MLResult_Timeout;
        } else {
            ALOGI("Main camera is available!");
        }

        MLCameraConnectContext camera_connect_context = {};
        MLCameraConnectContextInit(&camera_connect_context);
        camera_connect_context.cam_id = MLCameraIdentifier_MAIN;
        camera_connect_context.flags = MLCameraConnectFlag_MR;
        camera_connect_context.enable_video_stab = false;
        camera_connect_context.mr_info.blend_type = MLCameraMRBlendType_Additive;
        camera_connect_context.mr_info.frame_rate = MLCameraCaptureFrameRate_30FPS;
        camera_connect_context.mr_info.quality = MLCameraMRQuality_2880x2160;
        UNWRAP_RET_MEDIARESULT(MLCameraConnect(&camera_connect_context, &recorder_camera_context_));
        UNWRAP_RET_MEDIARESULT(SetCameraRecorderCallbacks());

        return MLResult_Ok;
    }

    static void CheckDeviceAvailability(const MLCameraDeviceAvailabilityInfo *device_availability_info,
                                        bool is_available) {
        if (device_availability_info == nullptr) {
            return;
        }
        CameraMixedRealityApp *this_app = static_cast<CameraMixedRealityApp *>(device_availability_info->user_data);
        if (this_app && device_availability_info->cam_id == MLCameraIdentifier_MAIN) {
            this_app->recorder_camera_device_available_ = is_available;
            this_app->camera_device_available_condition_.notify_one();
        }
    }

    MLResult SetCameraRecorderCallbacks() {
        MLCameraDeviceStatusCallbacks camera_device_status_callbacks = {};
        MLCameraDeviceStatusCallbacksInit(&camera_device_status_callbacks);

        camera_device_status_callbacks.on_device_error = [](MLCameraError err, void *) {
            ALOGE("on_device_error(%s) callback called for recorder camera", EnumHelpers::GetMLCameraErrorString(err));
        };

        camera_device_status_callbacks.on_device_disconnected = [](MLCameraDisconnectReason reason, void *data_ptr) {
            ALOGE("on_device_disconnected(%s) callback called for recorder camera",
                  EnumHelpers::GetMLCameraDisconnectReasonString(reason));

            if (data_ptr) {
                auto app_ptr = reinterpret_cast<CameraMixedRealityApp *>(data_ptr);
                if (!app_ptr->IsInteractive()) {
                    app_ptr->entered_standby_ = true;
                    app_ptr->standby_helper_threads_.emplace_back(&CameraMixedRealityApp::DestroyCamera, app_ptr);
                }
            }
        };
        UNWRAP_RET_MEDIARESULT(
                MLCameraSetDeviceStatusCallbacks(recorder_camera_context_, &camera_device_status_callbacks, this));

        MLCameraCaptureCallbacks camera_capture_callbacks = {};
        MLCameraCaptureCallbacksInit(&camera_capture_callbacks);

        camera_capture_callbacks.on_capture_failed = [](const MLCameraResultExtras *, void *) {
            ALOGI("on_capture_failed callback called for recorder camera");
        };

        camera_capture_callbacks.on_capture_aborted = [](void *) {
            ALOGI("on_capture_aborted callback called for recorder camera");
        };

        camera_capture_callbacks.on_image_buffer_available = OnImageAvailable;
        UNWRAP_RET_MEDIARESULT(MLCameraSetCaptureCallbacks(recorder_camera_context_, &camera_capture_callbacks, this));
        return MLResult_Ok;
    }

    MLResult SetupCaptureSize() {
        int32_t width = 0, height = 0;
        uint32_t streams_max = 0;
        UNWRAP_RET_MLRESULT(MLCameraGetNumSupportedStreams(recorder_camera_context_, &streams_max));

        typedef struct StreamCapsInfo {
            uint32_t stream_caps_max;
            MLCameraCaptureStreamCaps *stream_caps;
        } StreamCapsInfo;

        StreamCapsInfo *stream_caps_info = nullptr;
        stream_caps_info = (StreamCapsInfo *)malloc(streams_max * sizeof(StreamCapsInfo));
        if (stream_caps_info == nullptr) {
            ALOGE("Memory Allocation for StreamCapsInfo failed");
            return MLResult_UnspecifiedFailure;
        }

        for (uint32_t i = 0; i < streams_max; i++) {
            stream_caps_info[i].stream_caps_max = 0;
            stream_caps_info[i].stream_caps = nullptr;
            UNWRAP_RET_MLRESULT(
                    MLCameraGetStreamCaps(recorder_camera_context_, i, &stream_caps_info[i].stream_caps_max, nullptr));
            stream_caps_info[i].stream_caps =
                    (MLCameraCaptureStreamCaps *)malloc(stream_caps_info[i].stream_caps_max * sizeof(MLCameraCaptureStreamCaps));
            UNWRAP_RET_MLRESULT(MLCameraGetStreamCaps(recorder_camera_context_, i, &stream_caps_info[i].stream_caps_max,
                                                      stream_caps_info[i].stream_caps));

            for (uint32_t j = 0; j < stream_caps_info[i].stream_caps_max; j++) {
                const MLCameraCaptureStreamCaps capture_stream_caps = stream_caps_info[i].stream_caps[j];
                if (capture_stream_caps.capture_type == MLCameraCaptureType_Video) {
                    if (capture_stream_caps.width > width) {
                        width = capture_stream_caps.width;
                        height = capture_stream_caps.height;
                    }
                }
            }
        }

        for (uint32_t i = 0; i < streams_max; i++) {
            if (stream_caps_info[i].stream_caps != nullptr) {
                free(stream_caps_info[i].stream_caps);
            }
        }
        free(stream_caps_info);

        if (width > 0 && height > 0) {
            capture_width_ = width;
            capture_height_ = height;
        }

        return MLResult_Ok;
    }

    bool recorder_camera_device_available_;
    std::mutex camera_device_available_lock_;
    std::condition_variable camera_device_available_condition_;
    int32_t capture_width_, capture_height_;
    MLCameraContext recorder_camera_context_;
    const std::string default_output_filepath_;
    const std::string default_output_filename_photo_;
    std::string current_filename_photo_;
    bool entered_standby_;
    std::vector<std::thread> standby_helper_threads_;
};
void CameraMixedRealityApp::InitializeONNX() {
    if (onnx_initialized_) {
        // Already initialized; you can set a message or simply return.
        onnx_status_message_ = "ONNX Runtime already initialized.";
        return;
    }
    // Get ORT API handle
    const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort) {
        ALOGE("ONNX: Failed to get ORT API interface");
        onnx_status_message_ = "Failed to get ONNX Runtime API interface.";
        return;
    }
    // Create ONNX Runtime environment
    OrtEnv* env = nullptr;
    OrtStatus* status = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ML2App", &env);
    if (status != nullptr) {
        const char* errMsg = ort->GetErrorMessage(status);
        ALOGE("ONNX: OrtCreateEnv failed: %s", errMsg ? errMsg : "unknown error");
        onnx_status_message_ = std::string("ONNX init failed: ") + (errMsg ? errMsg : "unknown error");
        ort->ReleaseStatus(status);
    } else {
        ALOGI("ONNX: ONNX Runtime environment created successfully!");
        onnx_status_message_ = "ONNX Runtime environment created successfully!";
        onnx_initialized_ = true;
        // (Optional: keep `env` for reuse if doing inference later.
        // If not needed beyond this check, you can release it to free resources.)
        ort->ReleaseEnv(env);
    }
}


void android_main(struct android_app *state) {
#ifndef ML_LUMIN
    ALOGE("This app is not supported on App Sim!");
#else
    CameraMixedRealityApp app(state);
    app.RunApp();
#endif
}