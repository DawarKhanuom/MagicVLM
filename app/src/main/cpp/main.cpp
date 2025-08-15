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

#include <filesystem>  // For path validation
#include <chrono>      // For timing


#include <vector>
#include <string>
#include <fstream>


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
    ~CameraMixedRealityApp();  // Declare destructor
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
    const OrtApi* ort_ = nullptr;
    OrtSession* encoder_session_ = nullptr;
    OrtSession* decoder_session_ = nullptr;
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
CameraMixedRealityApp::~CameraMixedRealityApp() {
    if (ort_) {
        if (encoder_session_) {
            ort_->ReleaseSession(encoder_session_);
            encoder_session_ = nullptr;  // Avoid dangling pointers
        }
        if (decoder_session_) {
            ort_->ReleaseSession(decoder_session_);
            decoder_session_ = nullptr;  // Avoid dangling pointers
        }
    }
}
//void CameraMixedRealityApp::InitializeONNX() {
//    ort_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
//    if (!ort_) {
//        ALOGE("ONNX: Failed to get ORT API");
//        onnx_status_message_ = "ONNX init failed: API not available.";
//        return;
//    }
//
//    const char* ort_version = OrtGetApiBase()->GetVersionString();
//    ALOGI("ONNX Runtime Version: %s", ort_version ? ort_version : "unknown");
//    onnx_status_message_ = "ONNX Runtime Version: " + std::string(ort_version ? ort_version : "unknown");
//
//    OrtEnv* env = nullptr;
//    OrtStatus* status = ort_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ML2App", &env);
//    if (status) {
//        onnx_status_message_ += "\nFailed to create ONNX environment.";
//        ort_->ReleaseStatus(status);
//        return;
//    }
//
//    OrtSessionOptions* session_options;
//    ort_->CreateSessionOptions(&session_options);
//    ort_->SetIntraOpNumThreads(session_options, 1);
//
//    // Load encoder model
//    std::string encoder_path = "/storage/emulated/0/Android/data/com.magicleap.capi.sample.camera_mixed_reality/files/models/encoder_model.onnx";
//    status = ort_->CreateSession(env, encoder_path.c_str(), session_options, &encoder_session_);
//    if (status) {
//        onnx_status_message_ += "\nFailed to load encoder.";
//        ort_->ReleaseStatus(status);
//    } else {
//        onnx_status_message_ += "\nEncoder loaded.";
//    }
//
//    // Load decoder model
//    std::string decoder_path = "/storage/emulated/0/Android/data/com.magicleap.capi.sample.camera_mixed_reality/files/models/decoder_model.onnx";
//    status = ort_->CreateSession(env, decoder_path.c_str(), session_options, &decoder_session_);
//    if (status) {
//        onnx_status_message_ += "\nFailed to load decoder.";
//        ort_->ReleaseStatus(status);
//    } else {
//        onnx_status_message_ += "\nDecoder loaded.";
//    }
//
//    ort_->ReleaseSessionOptions(session_options);
//    ort_->ReleaseEnv(env);
//
//    // Dummy inference test
//    if (decoder_session_) {
//        std::vector<int64_t> input_ids = {101, 7592, 102};  // e.g., [CLS] hello [SEP]
//        std::vector<int64_t> shape = {1, static_cast<int64_t>(input_ids.size())};
//
//        OrtMemoryInfo* mem_info;
//        ort_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
//
//        OrtValue* input_tensor = nullptr;
//        ort_->CreateTensorWithDataAsOrtValue(mem_info, input_ids.data(),
//                                             input_ids.size() * sizeof(int64_t), shape.data(), shape.size(),
//                                             ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensor);
//
//        const char* input_names[] = {"input"};
//        const char* output_names[] = {"output"};
//        OrtValue* output_tensor = nullptr;
//
//        status = ort_->Run(decoder_session_, nullptr, input_names, &input_tensor, 1,
//                           output_names, 1, &output_tensor);
//
//        if (!status) {
//            // Placeholder response (real decoding requires tokenizer/vocab)
//            onnx_status_message_ += "\nVLM response: [Hello captioned]";
//            ort_->ReleaseValue(output_tensor);
//        } else {
//            onnx_status_message_ += "\nInference failed.";
//            ort_->ReleaseStatus(status);
//        }
//
//        ort_->ReleaseValue(input_tensor);
//        ort_->ReleaseMemoryInfo(mem_info);
//    }
//
//    onnx_initialized_ = true;
//}
void CameraMixedRealityApp::InitializeONNX() {
    onnx_status_message_.clear();
    onnx_initialized_ = false;

    // ---------- tiny helpers ----------
    auto read_file = [](const std::string& path) -> std::string {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return {};
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::string buf;
        buf.resize(sz);
        if (sz > 0) fread(&buf[0], 1, sz, f);
        fclose(f);
        return buf;
    };
    auto load_id_to_token = [&](const std::string& vocab_path) -> std::unordered_map<int, std::string> {
        std::unordered_map<int, std::string> id2tok;
        std::string s = read_file(vocab_path);
        if (s.empty()) return id2tok;
        size_t i = 0, n = s.size();
        while (i < n) {
            size_t q1 = s.find('\"', i); if (q1 == std::string::npos) break;
            size_t q2 = s.find('\"', q1 + 1); if (q2 == std::string::npos) break;
            std::string tok = s.substr(q1 + 1, q2 - q1 - 1);
            size_t colon = s.find(':', q2 + 1); if (colon == std::string::npos) break;
            size_t j = colon + 1; while (j < n && (s[j] == ' ' || s[j] == '\t')) j++;
            bool neg = false; if (j < n && s[j] == '-') { neg = true; j++; }
            long val = 0; bool ok = false;
            while (j < n && s[j] >= '0' && s[j] <= '9') { val = val*10 + (s[j]-'0'); j++; ok = true; }
            if (ok) id2tok[(int)(neg ? -val : val)] = tok;
            i = j + 1;
        }
        return id2tok;
    };
    auto resize_nn = [](const unsigned char* src, int sw, int sh, int sc, unsigned char* dst, int dw, int dh) {
        for (int y = 0; y < dh; ++y) {
            int sy = (int)((float)y * sh / dh);
            for (int x = 0; x < dw; ++x) {
                int sx = (int)((float)x * sw / dw);
                const unsigned char* sp = src + (sy * sw + sx) * sc;
                unsigned char* dp = dst + (y * dw + x) * sc;
                for (int c = 0; c < sc; ++c) dp[c] = sp[c];
            }
        }
    };
    auto hwc_to_nchw_norm = [](const unsigned char* src, int w, int h, int c, std::vector<float>& out) {
        out.assign(1 * 3 * 224 * 224, 0.f);
        const int W = 224, H = 224;
        for (int yy = 0; yy < H; ++yy) {
            for (int xx = 0; xx < W; ++xx) {
                const unsigned char* p = src + (yy * W + xx) * c;
                float r = (float)p[0] / 255.0f;
                float g = (float)p[1] / 255.0f;
                float b = (float)p[2] / 255.0f;
                r = (r - 0.5f) / 0.5f;
                g = (g - 0.5f) / 0.5f;
                b = (b - 0.5f) / 0.5f;
                out[0 * 224 * 224 + yy * 224 + xx] = r;
                out[1 * 224 * 224 + yy * 224 + xx] = g;
                out[2 * 224 * 224 + yy * 224 + xx] = b;
            }
        }
    };
    auto vec_i64 = [](int64_t n, int64_t val) { std::vector<int64_t> v(n, val); return v; };

    // ---------- ORT init ----------
    ort_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort_) { onnx_status_message_ = "ONNX init failed: API not available."; ALOGE("%s", onnx_status_message_.c_str()); return; }
    const char* ortv = OrtGetApiBase()->GetVersionString();
    onnx_status_message_ += "ONNX Runtime Version: " + std::string(ortv ? ortv : "unknown");

    OrtEnv* env = nullptr;
    if (OrtStatus* s = ort_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ML2App", &env)) {
        onnx_status_message_ += std::string("\nEnv failed: ") + ort_->GetErrorMessage(s);
        ALOGE("%s", onnx_status_message_.c_str()); ort_->ReleaseStatus(s); return;
    }

    OrtSessionOptions* so = nullptr; ort_->CreateSessionOptions(&so); ort_->SetIntraOpNumThreads(so, 1);
    // Later: try NNAPI if your ORT build supports it
    // OrtSessionOptionsAppendExecutionProvider_Nnapi(so, 0);

    std::string base = "/storage/emulated/0/Android/data/com.magicleap.capi.sample.camera_mixed_reality/files/models/";
    std::string encoder_path = base + "encoder_model.onnx";
    std::string decoder_path = base + "decoder_model.onnx";
    std::string image_path   = base + "dk.jpg";
    std::string vocab_path   = base + "vocab.json";

    // ---------- create sessions ----------
    if (OrtStatus* s = ort_->CreateSession(env, encoder_path.c_str(), so, &encoder_session_)) {
        onnx_status_message_ += std::string("\nEncoder load failed: ") + ort_->GetErrorMessage(s);
        ALOGE("%s", onnx_status_message_.c_str()); ort_->ReleaseStatus(s);
    } else { onnx_status_message_ += "\nEncoder loaded."; }
    if (OrtStatus* s = ort_->CreateSession(env, decoder_path.c_str(), so, &decoder_session_)) {
        onnx_status_message_ += std::string("\nDecoder load failed: ") + ort_->GetErrorMessage(s);
        ALOGE("%s", onnx_status_message_.c_str()); ort_->ReleaseStatus(s);
    } else { onnx_status_message_ += "\nDecoder loaded."; }

    ort_->ReleaseSessionOptions(so);
    ort_->ReleaseEnv(env);
    if (!encoder_session_ || !decoder_session_) { onnx_status_message_ += "\nAborting: sessions not ready."; return; }

    // ---------- enumerate I/O names to avoid mismatches ----------
    OrtAllocator* alloc = nullptr; ort_->GetAllocatorWithDefaultOptions(&alloc);

    // encoder inputs
    size_t enc_num_in = 0; ort_->SessionGetInputCount(encoder_session_, &enc_num_in);
    std::vector<char*> enc_in_names(enc_num_in, nullptr);
    for (size_t i = 0; i < enc_num_in; ++i) ort_->SessionGetInputName(encoder_session_, i, alloc, &enc_in_names[i]);
    // encoder outputs
    size_t enc_num_out = 0; ort_->SessionGetOutputCount(encoder_session_, &enc_num_out);
    std::vector<char*> enc_out_names(enc_num_out, nullptr);
    for (size_t i = 0; i < enc_num_out; ++i) ort_->SessionGetOutputName(encoder_session_, i, alloc, &enc_out_names[i]);

    // decoder inputs
    size_t dec_num_in = 0; ort_->SessionGetInputCount(decoder_session_, &dec_num_in);
    std::vector<char*> dec_in_names(dec_num_in, nullptr);
    for (size_t i = 0; i < dec_num_in; ++i) ort_->SessionGetInputName(decoder_session_, i, alloc, &dec_in_names[i]);
    // decoder outputs
    size_t dec_num_out = 0; ort_->SessionGetOutputCount(decoder_session_, &dec_num_out);
    std::vector<char*> dec_out_names(dec_num_out, nullptr);
    for (size_t i = 0; i < dec_num_out; ++i) ort_->SessionGetOutputName(decoder_session_, i, alloc, &dec_out_names[i]);

    // Find the expected names
    const char* ENC_IN_PIXEL = nullptr;
    for (auto n : enc_in_names) if (n && std::string(n) == "pixel_values") ENC_IN_PIXEL = n;
    if (!ENC_IN_PIXEL && !enc_in_names.empty()) ENC_IN_PIXEL = enc_in_names[0]; // fallback

    const char* ENC_OUT_FEATS = enc_out_names.empty() ? nullptr : enc_out_names[0]; // usually "last_hidden_state"

    int idx_input_ids = -1, idx_enc_hidden = -1, idx_attn = -1, idx_enc_attn = -1;
    for (int i = 0; i < (int)dec_in_names.size(); ++i) {
        std::string n = dec_in_names[i] ? dec_in_names[i] : "";
        if (n == "input_ids") idx_input_ids = i;
        else if (n == "encoder_hidden_states") idx_enc_hidden = i;
        else if (n == "attention_mask") idx_attn = i;
        else if (n == "encoder_attention_mask") idx_enc_attn = i;
    }
    if (idx_input_ids < 0) { onnx_status_message_ += "\nDecoder expects 'input_ids' but not found."; }
    if (idx_enc_hidden < 0) { onnx_status_message_ += "\nDecoder expects 'encoder_hidden_states' but not found."; }

    // ---------- load + preprocess dk.jpg ----------
    std::vector<float> nchw;
#ifdef STB_IMAGE_H
    int iw=0, ih=0, comp=0;
    unsigned char* img = stbi_load(image_path.c_str(), &iw, &ih, &comp, 3);
    if (!img) { onnx_status_message_ += "\nFailed to load image: " + image_path; ALOGE("%s", onnx_status_message_.c_str()); goto CLEANUP_NAMES; }
    const int W=224, H=224, C=3;
    std::vector<unsigned char> resized(W*H*C);
    resize_nn(img, iw, ih, 3, resized.data(), W, H);
    stbi_image_free(img);
    hwc_to_nchw_norm(resized.data(), W, H, C, nchw);
#else
    onnx_status_message_ += "\nError: stb_image.h not included.";
    ALOGE("%s", onnx_status_message_.c_str()); goto CLEANUP_NAMES;
#endif

    // ---------- run encoder ----------
    {
        OrtMemoryInfo* mi=nullptr; ort_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi);
        std::array<int64_t,4> enc_shape{1,3,224,224};
        OrtValue* enc_input = nullptr;
        if (OrtStatus* s = ort_->CreateTensorWithDataAsOrtValue(mi, (void*)nchw.data(),
                                                                sizeof(float)*nchw.size(), enc_shape.data(), enc_shape.size(),
                                                                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &enc_input)) {
            onnx_status_message_ += std::string("\nEncoder tensor failed: ") + ort_->GetErrorMessage(s);
            ALOGE("%s", onnx_status_message_.c_str()); ort_->ReleaseStatus(s); ort_->ReleaseMemoryInfo(mi); goto CLEANUP_NAMES;
        }

        const char* in_names_arr[1]  = { ENC_IN_PIXEL ? ENC_IN_PIXEL : "pixel_values" };
        const char* out_names_arr[1] = { ENC_OUT_FEATS ? ENC_OUT_FEATS : "last_hidden_state" };
        OrtValue* enc_output = nullptr;
        if (OrtStatus* s = ort_->Run(encoder_session_, nullptr, in_names_arr, &enc_input, 1,
                                     out_names_arr, 1, &enc_output)) {
            onnx_status_message_ += std::string("\nEncoder run failed: ") + ort_->GetErrorMessage(s);
            ALOGE("%s", onnx_status_message_.c_str());
            ort_->ReleaseStatus(s); ort_->ReleaseValue(enc_input); ort_->ReleaseMemoryInfo(mi); goto CLEANUP_NAMES;
        }
        ort_->ReleaseValue(enc_input);
        ort_->ReleaseMemoryInfo(mi);

        // Inspect encoder output shape for masks & sanity
        OrtTensorTypeAndShapeInfo* eti = nullptr;
        ort_->GetTensorTypeAndShape(enc_output, &eti);
        size_t endims = 0; ort_->GetDimensionsCount(eti, &endims);
        std::vector<int64_t> edims(endims,0); ort_->GetDimensions(eti, edims.data(), endims);
        ort_->ReleaseTensorTypeAndShapeInfo(eti);
        // Expected BLIP2 style: [1, S, D] or [1, 257, 1024], etc.

        onnx_status_message_ += "\nEncoder OK. Feat shape:";
        for (auto d : edims) onnx_status_message_ += " " + std::to_string(d);

        // ---------- decoder: greedy for few steps ----------
        std::string caption;
        const int BOS = 50256, EOS = 50256;
        std::vector<int64_t> seq; seq.push_back(BOS);

        // prepare encoder_attention_mask if required
        OrtValue* enc_attn_mask = nullptr;
        OrtMemoryInfo* mi2 = nullptr; ort_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi2);
        std::vector<int64_t> enc_mask;
        if (idx_enc_attn >= 0 && edims.size() >= 2) {
            int64_t enc_len = edims[1];
            enc_mask = vec_i64(enc_len, 1);
            std::array<int64_t,2> ms{1, enc_len};
            if (OrtStatus* s = ort_->CreateTensorWithDataAsOrtValue(mi2, enc_mask.data(),
                                                                    sizeof(int64_t)*enc_mask.size(), ms.data(), ms.size(),
                                                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &enc_attn_mask)) {
                onnx_status_message_ += std::string("\nencoder_attention_mask tensor failed: ") + ort_->GetErrorMessage(s);
                ALOGE("%s", onnx_status_message_.c_str()); ort_->ReleaseStatus(s);
                // Not fatal; proceed without mask
                enc_attn_mask = nullptr;
            }
        }

        for (int step = 0; step < 16; ++step) {
            // input_ids
            std::array<int64_t,2> ids_shape{1, (int64_t)seq.size()};
            OrtValue* ids_tensor = nullptr;
            if (OrtStatus* s = ort_->CreateTensorWithDataAsOrtValue(mi2, (void*)seq.data(),
                                                                    sizeof(int64_t)*seq.size(), ids_shape.data(), ids_shape.size(),
                                                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &ids_tensor)) {
                onnx_status_message_ += std::string("\nids tensor failed: ") + ort_->GetErrorMessage(s);
                ALOGE("%s", onnx_status_message_.c_str()); ort_->ReleaseStatus(s); break;
            }

            // attention_mask if needed
            OrtValue* attn_mask = nullptr;
            std::vector<int64_t> mask;
            if (idx_attn >= 0) {
                mask = vec_i64((int64_t)seq.size(), 1);
                if (OrtStatus* s = ort_->CreateTensorWithDataAsOrtValue(mi2, mask.data(),
                                                                        sizeof(int64_t)*mask.size(), ids_shape.data(), ids_shape.size(),
                                                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &attn_mask)) {
                    onnx_status_message_ += std::string("\nattention_mask tensor failed: ") + ort_->GetErrorMessage(s);
                    ALOGE("%s", onnx_status_message_.c_str()); ort_->ReleaseStatus(s);
                    attn_mask = nullptr; // continue without it
                }
            }

            // Build input array exactly in the decoder's input order
            std::vector<const char*> run_in_names; run_in_names.reserve(dec_num_in);
            std::vector<OrtValue*>   run_in_vals;  run_in_vals.reserve(dec_num_in);
            for (int i = 0; i < (int)dec_in_names.size(); ++i) {
                std::string nm = dec_in_names[i] ? dec_in_names[i] : "";
                if (nm == "input_ids")              { run_in_names.push_back(dec_in_names[i]); run_in_vals.push_back(ids_tensor); }
                else if (nm == "encoder_hidden_states") { run_in_names.push_back(dec_in_names[i]); run_in_vals.push_back(enc_output); }
                else if (nm == "attention_mask" && attn_mask) { run_in_names.push_back(dec_in_names[i]); run_in_vals.push_back(attn_mask); }
                else if (nm == "encoder_attention_mask" && enc_attn_mask) { run_in_names.push_back(dec_in_names[i]); run_in_vals.push_back(enc_attn_mask); }
                else {
                    // unhandled optional inputs: skip (ORT allows missing optional inputs by passing nullptr)
                    run_in_names.push_back(dec_in_names[i]);
                    run_in_vals.push_back(nullptr);
                }
            }

            // choose first output (usually logits)
            const char* out_name = dec_out_names.empty() ? "logits" : dec_out_names[0];
            OrtValue* dec_output = nullptr;
            if (OrtStatus* s = ort_->Run(decoder_session_, nullptr,
                                         run_in_names.data(), run_in_vals.data(), (size_t)run_in_names.size(),
                                         &out_name, 1, &dec_output)) {
                onnx_status_message_ += std::string("\nDecoder run failed: ") + ort_->GetErrorMessage(s);
                ALOGE("%s", onnx_status_message_.c_str()); ort_->ReleaseStatus(s);
                if (ids_tensor) ort_->ReleaseValue(ids_tensor);
                if (attn_mask)  ort_->ReleaseValue(attn_mask);
                break;
            }

            // shape & argmax
            OrtTensorTypeAndShapeInfo* ti = nullptr; ort_->GetTensorTypeAndShape(dec_output, &ti);
            size_t nd = 0; ort_->GetDimensionsCount(ti, &nd);
            std::vector<int64_t> dims(nd,0); ort_->GetDimensions(ti, dims.data(), nd);
            ort_->ReleaseTensorTypeAndShapeInfo(ti);

            float* logits = nullptr; ort_->GetTensorMutableData(dec_output, (void**)&logits);

            int next = 0;
            if (dims.size() == 3) {
                // [1, seq_len, vocab]
                int64_t vocab = dims[2];
                int64_t last_index = (int64_t)seq.size() - 1;
                float* last_logits = logits + last_index * vocab;
                float best = last_logits[0]; next = 0;
                for (int i = 1; i < vocab; ++i) { if (last_logits[i] > best) { best = last_logits[i]; next = i; } }
            } else if (dims.size() == 2) {
                // [1, vocab]
                int64_t vocab = dims[1];
                float best = logits[0]; next = 0;
                for (int i = 1; i < vocab; ++i) { if (logits[i] > best) { best = logits[i]; next = i; } }
            } else {
                onnx_status_message_ += "\nUnexpected decoder output shape.";
                ort_->ReleaseValue(dec_output);
                if (ids_tensor) ort_->ReleaseValue(ids_tensor);
                if (attn_mask)  ort_->ReleaseValue(attn_mask);
                break;
            }

            // append token
            seq.push_back(next);
            ort_->ReleaseValue(dec_output);
            if (ids_tensor) ort_->ReleaseValue(ids_tensor);
            if (attn_mask)  ort_->ReleaseValue(attn_mask);

            if (next == EOS) break;
        }

        // decode ids to string (very naive)
        auto id2tok = load_id_to_token(vocab_path);
        std::string caption;
        for (size_t k = 1; k < seq.size(); ++k) { // skip BOS
            if ((int)seq[k] == EOS) break;
            auto it = id2tok.find((int)seq[k]);
            if (it != id2tok.end()) caption += (caption.empty() ? "" : " ") + it->second;
            else caption += (caption.empty() ? "" : " ") + std::string("<") + std::to_string(seq[k]) + ">";
        }
        if (!caption.empty()) onnx_status_message_ += "\nCaption: " + caption;
        else                  onnx_status_message_ += "\nCaption: <empty>";

        if (enc_attn_mask) ort_->ReleaseValue(enc_attn_mask);
        if (mi2) ort_->ReleaseMemoryInfo(mi2);
        ort_->ReleaseValue(enc_output);
    }

    CLEANUP_NAMES:
    if (alloc) {
        for (auto p : enc_in_names)  if (p) alloc->Free(alloc, p);
        for (auto p : enc_out_names) if (p) alloc->Free(alloc, p);
        for (auto p : dec_in_names)  if (p) alloc->Free(alloc, p);
        for (auto p : dec_out_names) if (p) alloc->Free(alloc, p);
    }

    onnx_initialized_ = true;
}

void android_main(struct android_app *state) {
#ifndef ML_LUMIN
    ALOGE("This app is not supported on App Sim!");
#else
    CameraMixedRealityApp app(state);
    app.RunApp();
#endif
}