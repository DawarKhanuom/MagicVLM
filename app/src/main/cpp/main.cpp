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
#include <regex>


#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include <iostream>
#ifdef ML_LUMIN
#include <EGL/egl.h>
#define EGL_EGLEXT_PROTOTYPES
#include <EGL/eglext.h>
#endif

// --- stb_image (one-time implementation) ---
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>      // or: #include <stb/stb_image.h>

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





#include <regex>

// Read the whole file into a std::string
static bool ReadFileToString(const std::string& path, std::string& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return false; }
    fseek(f, 0, SEEK_SET);
    out.resize((size_t)sz);
    if (sz > 0) fread(&out[0], 1, (size_t)sz, f);
    fclose(f);
    return true;
}

// Build id->token list from vocab.json (string->int JSON map). Very lightweight parser.
static bool LoadGpt2VocabIdToToken(const std::string& vocab_path, std::vector<std::string>& id_to_token) {
    std::string json;
    if (!ReadFileToString(vocab_path, json)) return false;

    // Regex over entries:  "token": number
    // This is a simple approach; GPT-2 vocab keys are safe (no nested quotes).
    std::regex entry(R"xxx("([^"\\]|\\.)*"\s*:\s*\d+)xxx"); // find entries first
    std::regex pair (R"xxx("((?:[^"\\]|\\.)*)"\s*:\s*(\d+))xxx");

    size_t max_id = 0;
    std::vector<std::pair<size_t,std::string>> items;

    auto it = std::sregex_iterator(json.begin(), json.end(), entry);
    auto end = std::sregex_iterator();
    for (; it != end; ++it) {
        std::smatch m;
        std::string e = it->str();
        if (std::regex_search(e, m, pair)) {
            std::string tok = m[1].str();
            size_t id = (size_t)strtoull(m[2].str().c_str(), nullptr, 10);
            // Unescape minimal \" and \\ (good enough for GPT-2 vocab)
            std::string clean; clean.reserve(tok.size());
            for (size_t i=0;i<tok.size();++i) {
                if (tok[i]=='\\' && i+1<tok.size()) {
                    char c = tok[i+1];
                    if (c=='"' || c=='\\' || c=='/') { clean.push_back(c); ++i; }
                    else if (c=='n') { clean.push_back('\n'); ++i; }
                    else if (c=='t') { clean.push_back('\t'); ++i; }
                    else { clean.push_back(tok[i]); }
                } else {
                    clean.push_back(tok[i]);
                }
            }
            items.emplace_back(id, clean);
            if (id > max_id) max_id = id;
        }
    }

    if (items.empty()) return false;
    id_to_token.assign(max_id+1, std::string());
    for (auto& kv : items) {
        if (kv.first < id_to_token.size()) id_to_token[kv.first] = kv.second;
    }
    return true;
}

// Very simple GPT-2-ish “pretty” decode: join tokens and clean common markers.
// (Not a full byte-level decoder, but good enough to read in GUI.)
static std::string SimpleDecodeGpt2(const std::vector<int64_t>& ids,
                                    const std::vector<std::string>& id_to_token) {
    std::string s;
    s.reserve(ids.size()*3);
    for (auto id : ids) {
        if (id >= 0 && (size_t)id < id_to_token.size()) {
            const std::string& t = id_to_token[(size_t)id];
            if (t == "<|endoftext|>" || t == "" || t == " ") continue;
            s += t;
        } else {
            // unknown id → just skip or add placeholder
        }
    }
    // Clean common byte-level artifacts for readability
    auto replace_all = [&](const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    replace_all("Ġ", " ");   // space marker
    replace_all("Ċ", "\n");  // newline marker
    return s;
}

// UTF-8 safe substring replace
static void ReplaceAll(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}




void CameraMixedRealityApp::InitializeONNX() {
    // ----------- Guard re-entry (button double-hit etc.) -----------
    static std::atomic_flag busy = ATOMIC_FLAG_INIT;
    if (busy.test_and_set()) {
        onnx_status_message_ = "ONNX is already running…";
        return;
    }
    auto clear_busy = [&]() { busy.clear(); };

    onnx_status_message_.clear();
    onnx_initialized_ = false;

    // ----------- Small helpers (local lambdas) -----------
    auto FileExists = [](const std::string& p)->bool {
        FILE* f = fopen(p.c_str(), "rb");
        if (!f) return false; fclose(f); return true;
    };

    auto LoadImageCHW224F32 = [&](const std::string& path,
                                  std::vector<float>& out, int& W, int& H)->bool {
        // Uses stb_image to load and CPU-resize to 224x224 (nearest), then normalize.
        int w=0,h=0,c=0;
        unsigned char* img = stbi_load(path.c_str(), &w, &h, &c, 3); // force 3ch
        if (!img) return false;
        const int outW = 224, outH = 224, ch = 3;
        out.resize(ch * outH * outW);

        auto samp = [&](float x, float y, int chn) -> float {
            // nearest
            int ix = (int)std::roundf(x), iy=(int)std::roundf(y);
            if (ix < 0) ix = 0; if (ix >= w) ix = w-1;
            if (iy < 0) iy = 0; if (iy >= h) iy = h-1;
            int idx = (iy*w + ix)*3 + chn;
            return (float)img[idx] / 255.0f;
        };

        for (int oy=0; oy<outH; ++oy) {
            for (int ox=0; ox<outW; ++ox) {
                float sx = ( (ox + 0.5f) * w / (float)outW ) - 0.5f;
                float sy = ( (oy + 0.5f) * h / (float)outH ) - 0.5f;
                float r = samp(sx,sy,0);
                float g = samp(sx,sy,1);
                float b = samp(sx,sy,2);
                // normalize to [-1,1] via (x-0.5)/0.5
                r = (r - 0.5f) / 0.5f;
                g = (g - 0.5f) / 0.5f;
                b = (b - 0.5f) / 0.5f;
                // CHW layout
                int idx = oy*outW + ox;
                out[0*outH*outW + idx] = r;
                out[1*outH*outW + idx] = g;
                out[2*outH*outW + idx] = b;
            }
        }
        stbi_image_free(img);
        W = outW; H = outH;
        return true;
    };

    // Super-simple vocab loader (vocab.json: {"token": id, ...}) → id->token
    auto LoadVocabIdToToken = [&](const std::string& path, std::vector<std::string>& id2tok)->bool {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        if (sz <= 0) { fclose(f); return false; }
        std::string s; s.resize((size_t)sz);
        fread(&s[0], 1, (size_t)sz, f); fclose(f);

        size_t i=0, n=s.size();
        auto skip_ws=[&]{while(i<n && (unsigned char)s[i]<=32) ++i;};
        auto parse_string=[&]()->std::string{
            std::string out; if (i>=n || s[i]!='"') return out; ++i;
            while (i<n) {
                char c = s[i++];
                if (c=='\\' && i<n) {
                    char e = s[i++];
                    if (e=='"'||e=='\\'||e=='/') out.push_back(e);
                    else if (e=='b') out.push_back('\b');
                    else if (e=='f') out.push_back('\f');
                    else if (e=='n') out.push_back('\n');
                    else if (e=='r') out.push_back('\r');
                    else if (e=='t') out.push_back('\t');
                    else out.push_back(e); // very naive for \uXXXX
                } else if (c=='"') break;
                else out.push_back(c);
            }
            return out;
        };
        auto parse_int=[&]()->long{
            skip_ws(); bool neg=false; if (i<n && (s[i]=='-'||s[i]=='+')) { neg=(s[i]=='-'); ++i; }
            long v=0; while(i<n && s[i]>='0' && s[i]<='9'){ v = v*10 + (s[i]-'0'); ++i; }
            return neg? -v : v;
        };

        id2tok.clear(); id2tok.reserve(50000);
        // crude scan: "token" : number,
        while (i<n) {
            // find next key
            while (i<n && s[i]!='"') ++i;
            if (i>=n) break;
            std::string key = parse_string();
            if (key.empty()) break;
            // move to ':'
            while (i<n && s[i] != ':') ++i;
            if (i<n) ++i;
            long id = parse_int();
            if (id >= 0) {
                if ((size_t)id >= id2tok.size()) id2tok.resize((size_t)id+1);
                id2tok[(size_t)id] = key;
            }
            // move forward to next pair
            while (i<n && s[i]!=',' && s[i]!='}') ++i;
            if (i<n && s[i]==',') ++i;
            if (i<n && s[i]=='}') { ++i; break; }
        }
        return !id2tok.empty();
    };

    // Best-effort decode: join GPT-2 tokens with simple fixes
    auto SimpleDecode = [&](const std::vector<int64_t>& ids,
                            const std::vector<std::string>& id2tok)->std::string {
        std::string out;
        for (auto id : ids) {
            if (id < 0 || (size_t)id >= id2tok.size()) continue;
            const std::string& t = id2tok[(size_t)id];
            if (t.empty()) continue;
            // crude cleanups common with GPT-2/byte BPE vocab
            if (t == "Ċ") { out.push_back('\n'); continue; }
            if (!out.empty() && (t=="," || t=="." || t=="!" || t=="?" || t==";" || t==":" )) {
                // attach punctuation without space
                out += t; continue;
            }
            // common spacer marker in some BPEs (not strictly GPT-2)
            std::string cleaned = t;

            ReplaceAll(cleaned, u8"Ġ", " ");   // GPT-2 "space" marker
// optional extras:
            ReplaceAll(cleaned, u8"Ċ", "\n");  // GPT-2 "newline" marker
            ReplaceAll(cleaned, "<|endoftext|>", "");


            // heuristic: add space if needed
            if (!out.empty() && out.back()!=' ' && cleaned.size() && cleaned[0] != '\n' && cleaned[0] != ' ')
                out.push_back(' ');
            out += cleaned;
        }
        // trim
        while(!out.empty() && (out.front()==' ' || out.front()=='\n')) out.erase(out.begin());
        return out;
    };

    // ----------- 1) ORT API / Env / SessionOptions -----------
    ort_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort_) { onnx_status_message_ = "ONNX: API not available"; clear_busy(); return; }

    OrtEnv* env = nullptr;
    if (OrtStatus* st = ort_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ML2App_Caption", &env)) {
        onnx_status_message_ = std::string("CreateEnv failed: ") + (ort_->GetErrorMessage(st) ? ort_->GetErrorMessage(st) : "unknown");
        ort_->ReleaseStatus(st); clear_busy(); return;
    }

    OrtSessionOptions* so = nullptr;
    if (OrtStatus* st = ort_->CreateSessionOptions(&so)) {
        onnx_status_message_ = std::string("CreateSessionOptions failed: ") + (ort_->GetErrorMessage(st) ? ort_->GetErrorMessage(st) : "unknown");
        ort_->ReleaseStatus(st); ort_->ReleaseEnv(env); clear_busy(); return;
    }
    ort_->SetIntraOpNumThreads(so, 1);
    ort_->SetInterOpNumThreads(so, 1);
    ort_->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_BASIC);

    // ----------- 2) Load models -----------
    const std::string base = "/storage/emulated/0/Android/data/com.magicleap.capi.sample.camera_mixed_reality/files/models/";
    const std::string enc_path = base + "encoder_model.onnx";
    const std::string dec_path = base + "decoder_model.onnx";
    const std::string img_path = base + "dk.jpg";

    // clean old sessions
    if (encoder_session_) { ort_->ReleaseSession(encoder_session_); encoder_session_ = nullptr; }
    if (decoder_session_) { ort_->ReleaseSession(decoder_session_); decoder_session_ = nullptr; }

    OrtStatus* stA = ort_->CreateSession(env, enc_path.c_str(), so, &encoder_session_);
    OrtStatus* stB = ort_->CreateSession(env, dec_path.c_str(), so, &decoder_session_);
    if (stA) { onnx_status_message_ += "Encoder load failed.\n"; ort_->ReleaseStatus(stA); }
    else     { onnx_status_message_ += "Encoder loaded.\n"; }
    if (stB) { onnx_status_message_ += "Decoder load failed.\n"; ort_->ReleaseStatus(stB); }
    else     { onnx_status_message_ += "Decoder loaded.\n"; }

    if (!encoder_session_ || !decoder_session_) {
        ort_->ReleaseSessionOptions(so); ort_->ReleaseEnv(env); clear_busy(); return;
    }

    // ----------- 3) Resolve names -----------
    OrtAllocator* alloc = nullptr; ort_->GetAllocatorWithDefaultOptions(&alloc);
    auto get_names = [&](OrtSession* s, bool input, std::vector<char*>& names) {
        size_t n=0; (input ? ort_->SessionGetInputCount(s,&n) : ort_->SessionGetOutputCount(s,&n));
        names.resize(n,nullptr);
        for (size_t i=0;i<n;++i) {
            char* nm=nullptr;
            (input ? ort_->SessionGetInputName(s,i,alloc,&nm) : ort_->SessionGetOutputName(s,i,alloc,&nm));
            names[i] = nm;
        }
    };
    std::vector<char*> enc_in_names, enc_out_names, dec_in_names, dec_out_names;
    get_names(encoder_session_, true,  enc_in_names);
    get_names(encoder_session_, false, enc_out_names);
    get_names(decoder_session_, true,  dec_in_names);
    get_names(decoder_session_, false, dec_out_names);

    const char* enc_in  = enc_in_names.empty()?  "pixel_values" : enc_in_names[0];
    const char* enc_out = enc_out_names.empty()? nullptr         : enc_out_names[0];
    if (!enc_out) { onnx_status_message_ += "No encoder output name.\n"; }

    const char* dec_in_ids = nullptr;
    const char* dec_in_hs  = nullptr;
    for (auto* n: dec_in_names) if (n) {
            std::string s(n);
            if (!dec_in_ids && s.find("input_ids") != std::string::npos) dec_in_ids = n;
            if (!dec_in_hs  && (s.find("encoder_hidden_states")!=std::string::npos ||
                                s.find("encoder_outputs")!=std::string::npos ||
                                s.find("encoder_out")!=std::string::npos)) dec_in_hs = n;
        }
    const char* dec_out_logits = dec_out_names.empty()? "logits" : dec_out_names[0];

    if (!enc_out || !dec_in_ids || !dec_in_hs) {
        onnx_status_message_ += "Abort: required I/O names not found.\n";
        if (alloc){ for (auto*p:enc_in_names) if(p) alloc->Free(alloc,p);
            for (auto*p:enc_out_names)if(p) alloc->Free(alloc,p);
            for (auto*p:dec_in_names) if(p) alloc->Free(alloc,p);
            for (auto*p:dec_out_names)if(p) alloc->Free(alloc,p); }
        ort_->ReleaseSessionOptions(so); ort_->ReleaseEnv(env); clear_busy(); return;
    }

    // ----------- 4) Load & preprocess image dk.jpg -----------
    if (!FileExists(img_path)) {
        onnx_status_message_ += "Image dk.jpg not found in models folder.\n";
        if (alloc){ for (auto*p:enc_in_names) if(p) alloc->Free(alloc,p);
            for (auto*p:enc_out_names)if(p) alloc->Free(alloc,p);
            for (auto*p:dec_in_names) if(p) alloc->Free(alloc,p);
            for (auto*p:dec_out_names)if(p) alloc->Free(alloc,p); }
        ort_->ReleaseSessionOptions(so); ort_->ReleaseEnv(env); clear_busy(); return;
    }
    std::vector<float> pix; int W=0,H=0;
    if (!LoadImageCHW224F32(img_path, pix, W, H)) {
        onnx_status_message_ += "Failed to load/resize image.\n";
        if (alloc){ for (auto*p:enc_in_names) if(p) alloc->Free(alloc,p);
            for (auto*p:enc_out_names)if(p) alloc->Free(alloc,p);
            for (auto*p:dec_in_names) if(p) alloc->Free(alloc,p);
            for (auto*p:dec_out_names)if(p) alloc->Free(alloc,p); }
        ort_->ReleaseSessionOptions(so); ort_->ReleaseEnv(env); clear_busy(); return;
    }

    // ----------- 5) Run encoder -----------
    OrtMemoryInfo* mi = nullptr; ort_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi);
    int64_t enc_shape[4] = {1,3, (int64_t)H, (int64_t)W};
    OrtValue* enc_input = nullptr;
    OrtStatus* stC = ort_->CreateTensorWithDataAsOrtValue(
            mi, pix.data(), pix.size()*sizeof(float),
            enc_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &enc_input);
    if (stC) {
        onnx_status_message_ += "CreateTensor(enc_input) failed.\n";
        ort_->ReleaseStatus(stC);
    }

    OrtValue* enc_output = nullptr;
    if (enc_input) {
        const char* in_names[]  = { enc_in };
        const char* out_names[] = { enc_out };
        if (OrtStatus* st = ort_->Run(encoder_session_, nullptr, in_names, &enc_input, 1, out_names, 1, &enc_output)) {
            onnx_status_message_ += "Encoder run failed.\n";
            ort_->ReleaseStatus(st);
        } else {
            onnx_status_message_ += "Encoder run OK.\n";
        }
    }

    // read encoder output dims and data
    std::vector<int64_t> enc_out_dims;
    std::vector<float>   enc_feat;
    if (enc_output) {
        OrtTensorTypeAndShapeInfo* ti = nullptr; ort_->GetTensorTypeAndShape(enc_output, &ti);
        size_t nd=0; if (ti) ort_->GetDimensionsCount(ti,&nd);
        enc_out_dims.resize(nd,0);
        if (ti && nd) ort_->GetDimensions(ti, enc_out_dims.data(), nd);
        float* ptr=nullptr; ort_->GetTensorMutableData(enc_output,(void**)&ptr);
        size_t cnt=1; for (auto d: enc_out_dims) { if (d<1) d=1; cnt *= (size_t)d; }
        if (ptr && cnt) enc_feat.assign(ptr, ptr+cnt);
        if (ti) ort_->ReleaseTensorTypeAndShapeInfo(ti);
    }
    if (enc_input)  ort_->ReleaseValue(enc_input);
    if (enc_output) ort_->ReleaseValue(enc_output);

    if (enc_feat.empty()) {
        onnx_status_message_ += "Encoder produced empty features.\n";
        if (mi) ort_->ReleaseMemoryInfo(mi);
        if (alloc){ for (auto*p:enc_in_names) if(p) alloc->Free(alloc,p);
            for (auto*p:enc_out_names)if(p) alloc->Free(alloc,p);
            for (auto*p:dec_in_names) if(p) alloc->Free(alloc,p);
            for (auto*p:dec_out_names)if(p) alloc->Free(alloc,p); }
        ort_->ReleaseSessionOptions(so); ort_->ReleaseEnv(env); clear_busy(); return;
    }

    // ----------- 6) Decoder greedy generation -----------
    // Find element type for input_ids (INT64 vs INT32)
    ONNXTensorElementDataType ids_elem = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    {
        OrtTypeInfo* ti=nullptr;
        if (!ort_->SessionGetInputTypeInfo(decoder_session_, 0, &ti) && ti) {
            const OrtTensorTypeAndShapeInfo* tti=nullptr;
            if (!ort_->CastTypeInfoToTensorInfo(ti,&tti) && tti) {
                ONNXTensorElementDataType e=ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
                ort_->GetTensorElementType(tti,&e);
                if (e == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) ids_elem = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
            }
            ort_->ReleaseTypeInfo(ti);
        }
    }

    // BOS/EOS for GPT-2 style
    const int64_t BOS = 50256, EOS = 50256;
    std::vector<int64_t> input_ids; input_ids.push_back(BOS);
    int64_t ids_shape[2] = {1, 1};
    OrtValue* t_ids = nullptr;

    auto rebuild_ids = [&]() -> bool {
        if (t_ids) { ort_->ReleaseValue(t_ids); t_ids = nullptr; }
        ids_shape[1] = (int64_t)input_ids.size();
        if (ids_elem == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            std::vector<int32_t> ids32(input_ids.begin(), input_ids.end());
            if (OrtStatus* st = ort_->CreateTensorWithDataAsOrtValue(
                    mi, ids32.data(), ids32.size()*sizeof(int32_t),
                    ids_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &t_ids)) {
                ort_->ReleaseStatus(st); return false;
            }
        } else {
            if (OrtStatus* st = ort_->CreateTensorWithDataAsOrtValue(
                    mi, input_ids.data(), input_ids.size()*sizeof(int64_t),
                    ids_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_ids)) {
                ort_->ReleaseStatus(st); return false;
            }
        }
        return true;
    };

    if (!rebuild_ids()) {
        onnx_status_message_ += "Failed to create input_ids tensor.\n";
        if (mi) ort_->ReleaseMemoryInfo(mi);
        if (alloc){ for (auto*p:enc_in_names) if(p) alloc->Free(alloc,p);
            for (auto*p:enc_out_names)if(p) alloc->Free(alloc,p);
            for (auto*p:dec_in_names) if(p) alloc->Free(alloc,p);
            for (auto*p:dec_out_names)if(p) alloc->Free(alloc,p); }
        ort_->ReleaseSessionOptions(so); ort_->ReleaseEnv(env); clear_busy(); return;
    }

    // encoder_hidden_states tensor from enc_feat
    OrtValue* t_hs = nullptr;
    if (OrtStatus* st = ort_->CreateTensorWithDataAsOrtValue(
            mi, enc_feat.data(), enc_feat.size()*sizeof(float),
            enc_out_dims.data(), enc_out_dims.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_hs)) {
        onnx_status_message_ += "CreateTensor(encoder_hidden_states) failed.\n";
        ort_->ReleaseStatus(st);
    }

    std::vector<int64_t> gen_ids; // newly generated (for display)
    int max_steps = 30;
    for (int step=0; step<max_steps; ++step) {
        if (!t_ids || !t_hs) break;
        const char* feed_names[2] = { dec_in_ids, dec_in_hs };
        OrtValue*   feed_vals [2] = { t_ids,      t_hs      };
        const char* fetch_names[1] = { dec_out_logits };
        OrtValue*   fetches    [1] = { nullptr };

        OrtStatus* st = ort_->Run(decoder_session_, nullptr,
                                  feed_names, feed_vals, 2,
                                  fetch_names, 1, fetches);
        if (st) {
            onnx_status_message_ += "Decoder run failed.\n";
            ort_->ReleaseStatus(st);
            if (fetches[0]) ort_->ReleaseValue(fetches[0]);
            break;
        }

        // logits shape [1, seq_len, vocab]
        float* logits = nullptr; ort_->GetTensorMutableData(fetches[0], (void**)&logits);
        OrtTensorTypeAndShapeInfo* ti=nullptr; ort_->GetTensorTypeAndShape(fetches[0], &ti);
        size_t nd=0; std::vector<int64_t> lshape;
        if (ti) { ort_->GetDimensionsCount(ti,&nd); lshape.resize(nd,0); if (nd) ort_->GetDimensions(ti,lshape.data(),nd); }
        int64_t seq_len = (lshape.size()>=2)? lshape[1] : 0;
        int64_t vocab   = (lshape.size()>=3)? lshape[2] : 0;
        int64_t best_id = 0;
        if (logits && seq_len>0 && vocab>0) {
            float* last = logits + (seq_len-1)*vocab;
            float best_val = last[0];
            for (int64_t i=1;i<vocab;++i) if (last[i] > best_val) { best_val = last[i]; best_id = i; }
        }
        if (ti) ort_->ReleaseTensorTypeAndShapeInfo(ti);
        ort_->ReleaseValue(fetches[0]);

        input_ids.push_back(best_id);
        gen_ids.push_back(best_id);
        if (best_id == EOS) break;
        if (!rebuild_ids()) { onnx_status_message_ += "Rebuild input_ids failed.\n"; break; }
    }

    // ----------- 7) Display results -----------
    onnx_status_message_ += "Caption token IDs: ";
    for (size_t i=1; i<input_ids.size(); ++i) { // skip BOS at index 0
        onnx_status_message_ += std::to_string(input_ids[i]);
        if (i+1 < input_ids.size()) onnx_status_message_ += ",";
    }
    onnx_status_message_ += "\n";

    // best-effort text decode using vocab.json if present
    const std::string vocab_path = base + "vocab.json";
    std::vector<std::string> id2tok;
    if (FileExists(vocab_path) && LoadVocabIdToToken(vocab_path, id2tok)) {
        std::string decoded = SimpleDecode(std::vector<int64_t>(input_ids.begin()+1, input_ids.end()), id2tok);
        if (!decoded.empty()) {
            onnx_status_message_ += "Caption (decoded): ";
            onnx_status_message_ += decoded;
            onnx_status_message_ += "\n";
        } else {
            onnx_status_message_ += "Decoded caption empty (vocab present).\n";
        }
    } else {
        onnx_status_message_ += "vocab.json not found → showing IDs only.\n";
    }

    // ----------- Cleanup -----------
    if (t_ids)  ort_->ReleaseValue(t_ids);
    if (t_hs)   ort_->ReleaseValue(t_hs);
    if (mi)     ort_->ReleaseMemoryInfo(mi);

    if (alloc){
        for (auto*p:enc_in_names)  if(p) alloc->Free(alloc,p);
        for (auto*p:enc_out_names) if(p) alloc->Free(alloc,p);
        for (auto*p:dec_in_names)  if(p) alloc->Free(alloc,p);
        for (auto*p:dec_out_names) if(p) alloc->Free(alloc,p);
    }
    ort_->ReleaseSessionOptions(so);
    ort_->ReleaseEnv(env);

    onnx_initialized_ = true;
    onnx_status_message_ += "ONNX captioning complete.\n";
    clear_busy();
}



void android_main(struct android_app *state) {
#ifndef ML_LUMIN
    ALOGE("This app is not supported on App Sim!");
#else
    CameraMixedRealityApp app(state);
    app.RunApp();
#endif
}