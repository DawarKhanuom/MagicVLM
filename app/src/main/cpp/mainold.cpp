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
#include <cstring>
#include <chrono>
#include <condition_variable>
#include <sstream>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cmath>
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

#include <vector>
#include <string>
#include <fstream>
#include <regex>
#include <iostream>
#include <mutex>
#include <unordered_map>

#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#ifdef ML_LUMIN
#include <EGL/egl.h>
#define EGL_EGLEXT_PROTOTYPES
#include <EGL/eglext.h>
#endif

// --- stb_image (one-time implementation) ---
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define UNWRAP_RET_MEDIARESULT(res) UNWRAP_RET_MLRESULT_GENERIC(res, UNWRAP_MLMEDIA_RESULT)

#ifndef ML_WINDOWS
#include <unistd.h>   // fsync
#endif


// ===== ANDROID / JNI (for STT/TTS) =====
#include <jni.h>
#include <android_native_app_glue.h>
#include <android/native_activity.h>

// Mailbox for ASR results
static std::mutex g_asrMutex;
static std::string g_lastASR;
static std::string g_lastASRError;

// Java class/object handles
static jclass  g_VoiceBridgeCls = nullptr;
static jobject g_VoiceBridgeObj = nullptr;

// Load Java class via Activity's ClassLoader (works on ML2)
static jclass LoadClassFromActivity(JNIEnv* env, jobject activity, const char* fqcn) {
    jclass activityCls = env->GetObjectClass(activity);
    jmethodID getCL = env->GetMethodID(activityCls, "getClassLoader", "()Ljava/lang/ClassLoader;");
    jobject clObj = env->CallObjectMethod(activity, getCL);
    jclass clCls = env->FindClass("java/lang/ClassLoader");
    jmethodID loadClass = env->GetMethodID(clCls, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;");
    jstring name = env->NewStringUTF(fqcn);
    jclass result = (jclass)env->CallObjectMethod(clObj, loadClass, name);
    env->DeleteLocalRef(name);
    return result;
}


//............................................
static std::string TrimCollapse(const std::string& in) {
    std::string out; out.reserve(in.size());
    bool space=false;
    for (char c: in) {
        if (c==' '||c=='\t'||c=='\n'||c=='\r') { if (!space) { out.push_back(' '); space=true; } }
        else { out.push_back(c); space=false; }
    }
    while (!out.empty() && out.front()==' ') out.erase(out.begin());
    while (!out.empty() && out.back()==' ') out.pop_back();
    return out;
}
//.................................

// JNI callbacks from VoiceBridge.java → C++
extern "C" {

JNIEXPORT void JNICALL
Java_com_magicleap_capi_sample_camera_1mixed_1reality_VoiceBridge_nativeOnASRResult(
        JNIEnv* env, jclass /*cls*/, jstring jtext) {
    const char* t = env->GetStringUTFChars(jtext, nullptr);
    {
        std::lock_guard<std::mutex> lk(g_asrMutex);
        g_lastASR = t ? t : "";
        g_lastASRError.clear();
    }
    env->ReleaseStringUTFChars(jtext, t);
}

JNIEXPORT void JNICALL
Java_com_magicleap_capi_sample_camera_1mixed_1reality_VoiceBridge_nativeOnASRError(
        JNIEnv* env, jclass /*cls*/, jstring jerr) {
    const char* e = env->GetStringUTFChars(jerr, nullptr);
    {
        std::lock_guard<std::mutex> lk(g_asrMutex);
        g_lastASRError = e ? e : "ASR error";
    }
    env->ReleaseStringUTFChars(jerr, e);
}

} // extern "C"

// ===== Helpers for camera error strings =====
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
}

using namespace ml::app_framework;
using namespace std::chrono_literals;

//--------------------------
static bool FileExistsAbs(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb"); if (!f) return false; fclose(f); return true;
}

static bool CopyFileAbs(const std::string& src, const std::string& dst) {
    FILE* in = fopen(src.c_str(), "rb"); if (!in) return false;
    FILE* out = fopen(dst.c_str(), "wb"); if (!out) { fclose(in); return false; }
    char buf[1<<16];
    size_t n = 0;
    while ((n = fread(buf,1,sizeof(buf),in)) > 0) fwrite(buf,1,n,out);
    fflush(out);
    int fd = fileno(out);
    if (fd >= 0) fsync(fd);
    fclose(in); fclose(out);
    return true;
}

// =================== APP ===================
class CameraMixedRealityApp : public Application {
public:
    CameraMixedRealityApp(struct android_app* state);
    ~CameraMixedRealityApp() override;

    // ONNX status for GUI
    std::string onnx_status_message_;
    bool onnx_initialized_ = false;

    // Single-flow UI state
    bool send_to_vlm_after_capture_ = false;
    std::string last_captured_image_;
    bool pending_voice_question_ = false;
    std::string typed_question_;

    void InitializeONNX(const std::string& image_path, const std::string& question_text);

    void OnStart() override { mkdir(default_output_filepath_.c_str(), 0755);
        RestoreLastImageAlias();
    }

    void OnResume() override {
        if (ArePermissionsGranted()) {
            GetGui().Show();
            SetupRestrictedResources();
            InitVoice();   // start TTS/ASR bridge
        }
    }

    void OnStop() override {
        UNWRAP_MLRESULT(DestroyCamera());
        DestroyVoice(); // shutdown TTS/ASR bridge
    }

    void OnDestroy() override {
        for (auto &t : standby_helper_threads_) if (t.joinable()) t.join();
        standby_helper_threads_.clear();
        UNWRAP_MLRESULT(DestroyCamera());
    }

    void OnUpdate(float) override { UpdateGui(); }

    // Voice bridge API (C++ → Java)
    void InitVoice();
    void DestroyVoice();
    void StartListening();
    void StopListening();
    void Speak(const std::string& text);

private:
    // --- TTS / answer playback control ---
    bool speak_auto_ = false;                 // OFF by default (text-only until you opt in)
    std::string last_answer_text_;            // last decoded answer text
    std::atomic<uint64_t> answer_seq_{0};     // increments each time we compute a new answer
//............................................................


    //...................
    std::string AliasPath() const { return default_output_filepath_ + "dk.jpg"; }

    void PersistLastImageAlias(const std::string& real_path) {
        // Make a stable alias the app can always find
        if (CopyFileAbs(real_path, AliasPath())) {
            last_captured_image_ = AliasPath(); // always point UI & VLM to the alias
        }
    }

    bool RestoreLastImageAlias() {
        if (FileExistsAbs(AliasPath())) { last_captured_image_ = AliasPath(); return true; }
        return false;
    }



    void ShowKeyboard();
    void OpenEditDialog(const std::string& initial);
    char question_buf_[512];        // <-- persistent GUI text buffer
    void UpdateGui();

    android_app* app_state_ = nullptr;

    // ONNX
    const OrtApi* ort_ = nullptr;
    OrtSession* encoder_session_ = nullptr;
    OrtSession* decoder_session_ = nullptr;
    //---------]
    OrtEnv* env_ = nullptr;
    OrtSessionOptions* so_ = nullptr;
    bool vlm_ready_ = false;
    bool EnsureVLMLoaded();
    void UnloadVLM();


    void SetupRestrictedResources() {
        if (entered_standby_) {
            UNWRAP_MLRESULT(DestroyCamera());
            entered_standby_ = false;
        }
        ASSERT_MLRESULT(SetupCamera());
        ASSERT_MLRESULT(SetupCaptureSize());
    }

    static void OnImageAvailable(const MLCameraOutput *output, const MLHandle /*metadata_handle*/,
                                 const MLCameraResultExtras *extra, void *data);

    MLResult CaptureImage();
    MLResult DestroyCamera();
    MLResult SetupCamera();
    static void CheckDeviceAvailability(const MLCameraDeviceAvailabilityInfo *device_availability_info, bool is_available);
    MLResult SetCameraRecorderCallbacks();
    MLResult SetupCaptureSize();

    // ---- camera state ----
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
//----------------------------------------------endof  class--------------------------------------------------
//================================================================================
CameraMixedRealityApp::CameraMixedRealityApp(struct android_app* state)
        : Application(state, {"android.permission.CAMERA","android.permission.RECORD_AUDIO"}, USE_GUI),
          app_state_(state),
          recorder_camera_device_available_(false),
          capture_width_(0),
          capture_height_(0),
          recorder_camera_context_(ML_INVALID_HANDLE),
          default_output_filepath_(GetExternalFilesDir() + "/captures/"),
          default_output_filename_photo_("mr_dk_camera_photo_output"),
          entered_standby_(false)
{
    std::strncpy(question_buf_, "What is in the image?", sizeof(question_buf_));
    question_buf_[sizeof(question_buf_)-1] = '\0';
}

// ===== Destructor (cleanup ONNX sessions) =====
CameraMixedRealityApp::~CameraMixedRealityApp() {
    if (ort_) {
        if (encoder_session_) { ort_->ReleaseSession(encoder_session_); encoder_session_ = nullptr; }
        if (decoder_session_) { ort_->ReleaseSession(decoder_session_); decoder_session_ = nullptr; }
    }
}
//.............................................vlm loading once .............start............................
bool CameraMixedRealityApp::EnsureVLMLoaded() {
    if (vlm_ready_ && encoder_session_ && decoder_session_) return true;

    ort_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort_) { onnx_status_message_ = "ONNX API not available"; return false; }

    if (!env_) {
        if (OrtStatus* st = ort_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "BLIP2App", &env_)) {
            onnx_status_message_ = std::string("CreateEnv failed: ") + (st? ort_->GetErrorMessage(st) : "unknown");
            if (st) ort_->ReleaseStatus(st);
            return false;
        }
    }
    if (!so_) {
        if (OrtStatus* st = ort_->CreateSessionOptions(&so_)) {
            onnx_status_message_ = "CreateSessionOptions failed";
            if (st) ort_->ReleaseStatus(st);
            return false;
        }
        ort_->SetIntraOpNumThreads(so_, 1);
        ort_->SetInterOpNumThreads(so_, 1);
        ort_->SetSessionGraphOptimizationLevel(so_, ORT_ENABLE_BASIC);
    }

    // Load sessions only if missing
    if (!encoder_session_) {
        const std::string base = "/storage/emulated/0/Android/data/com.magicleap.capi.sample.camera_mixed_reality/files/models/";
        const std::string enc_path = base + "encoder_model.onnx";
        if (OrtStatus* st = ort_->CreateSession(env_, enc_path.c_str(), so_, &encoder_session_)) {
            onnx_status_message_ += "Encoder load failed.\n"; ort_->ReleaseStatus(st); return false;
        } else onnx_status_message_ += "Encoder loaded.\n";
    }
    if (!decoder_session_) {
        const std::string base = "/storage/emulated/0/Android/data/com.magicleap.capi.sample.camera_mixed_reality/files/models/";
        const std::string dec_path = base + "decoder_model.onnx";
        if (OrtStatus* st = ort_->CreateSession(env_, dec_path.c_str(), so_, &decoder_session_)) {
            onnx_status_message_ += "Decoder load failed.\n"; ort_->ReleaseStatus(st); return false;
        } else onnx_status_message_ += "Decoder loaded.\n";
    }

    vlm_ready_ = (encoder_session_ && decoder_session_);
    return vlm_ready_;
}

void CameraMixedRealityApp::UnloadVLM() {
    if (decoder_session_) { ort_->ReleaseSession(decoder_session_); decoder_session_ = nullptr; }
    if (encoder_session_) { ort_->ReleaseSession(encoder_session_); encoder_session_ = nullptr; }
    if (so_) { ort_->ReleaseSessionOptions(so_); so_ = nullptr; }
    if (env_) { ort_->ReleaseEnv(env_); env_ = nullptr; }
    vlm_ready_ = false;
}

//...............................vlm loading once end .................................

// ===== GUI ========================================================================
// ===== GUI =====
void CameraMixedRealityApp::UpdateGui() {
    auto &gui = GetGui();
    gui.BeginUpdate();

    bool app_running = true;

    if (gui.BeginDialog("Image + Voice Q&A", &app_running,
                        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize)) {

        ImGui::Text("1) Capture an image");

        // Capture + send using the current text in the box after the photo is saved
        if (ImGui::Button("Capture & Ask in text")) {
            send_to_vlm_after_capture_ = true;
            pending_voice_question_    = false;
            typed_question_            = question_buf_;   // snapshot
            UNWRAP_MLRESULT(CaptureImage());
        }
        ImGui::SameLine();
        if (ImGui::Button("Capture & ask by voice")) {
            send_to_vlm_after_capture_ = false;
            pending_voice_question_    = true;           // start listening after capture
            UNWRAP_MLRESULT(CaptureImage());
        }
        ImGui::SameLine();
        if (ImGui::Button("Capture only")) {
            send_to_vlm_after_capture_ = false;
            pending_voice_question_    = false;
            UNWRAP_MLRESULT(CaptureImage());
        }

        ImGui::Separator();
        ImGui::Text("Last photo:");
        ImGui::TextWrapped("%s", last_captured_image_.empty() ? "(none yet)" : last_captured_image_.c_str());

        ImGui::Separator();
        ImGui::Text("2) Ask a question about the last photo");
        //.............................
        ImGui::Separator();
        if (ImGui::Button(vlm_ready_ ? "Reload VLM" : "Load VLM")) {
            if (vlm_ready_) UnloadVLM();
            onnx_status_message_.clear();
            EnsureVLMLoaded();
        }
        ImGui::SameLine();
        if (ImGui::Button("Unload VLM")) {
            UnloadVLM();
        }
//.............................................

        if (!last_captured_image_.empty()) {
            if (ImGui::Button("Ask by voice")) {
                pending_voice_question_ = false; // listen immediately
                StartListening();
            }
            ImGui::SameLine();
            if (ImGui::Button("Stop listening")) {
                StopListening();
            }
        } else {
            ImGui::TextDisabled("Capture an image first to enable voice/text question.");
        }

        // Text box bound to the persistent member buffer
        ImGui::InputText("or type question", question_buf_, IM_ARRAYSIZE(question_buf_));

        ImGui::SameLine();
        if (ImGui::Button("Edit")) {
            OpenEditDialog(std::string(question_buf_));
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            question_buf_[0] = '\0';
        }

        ImGui::SameLine();
        if (ImGui::Button("Ask (text)")) {
            if (!last_captured_image_.empty() && question_buf_[0] != '\0') {
                std::string img = last_captured_image_;
                std::string q   = std::string(question_buf_);
                std::thread([this, img, q]() {
                    InitializeONNX(img, q);  // runs on worker thread
                }).detach();
            } else {
                onnx_status_message_ = "Type a question and capture an image first.";
            }
        }


        // ---- Poll ASR mailbox (JNI -> C++) ----
        // ---- Poll ASR mailbox (JNI -> C++) ----
        {
            std::lock_guard<std::mutex> lk(g_asrMutex);
            if (!g_lastASRError.empty()) {
                ImGui::TextColored(ImVec4(1,0.4f,0.4f,1), "ASR error: %s", g_lastASRError.c_str());
                g_lastASRError.clear();
            }
            if (!g_lastASR.empty()) {
                std::string transcript = g_lastASR;
                g_lastASR.clear();

                // Just update the GUI text box — do NOT call VLM
                std::strncpy(question_buf_, transcript.c_str(), sizeof(question_buf_) - 1);
                question_buf_[sizeof(question_buf_) - 1] = '\0';
            }
        }

        //...................................

        // ---- Status / Answer + auto TTS ----
        ImGui::Separator();
        ImGui::Text("VLM status:");
        ImGui::BeginChild("onnx_scroll", ImVec2(520, 200), true);
        ImGui::TextWrapped("%s", onnx_status_message_.c_str());
        ImGui::EndChild();


        //..........................
        // After your status window (onnx_status_message_)
        ImGui::Separator();
        ImGui::Text("Answer audio:");

        ImGui::Checkbox("Speak answers automatically", &speak_auto_);
        ImGui::SameLine();

        bool canSpeak = !last_answer_text_.empty() && last_answer_text_ != "(empty)";
        if (!canSpeak) ImGui::BeginDisabled();
        if (ImGui::Button("▶Listen to last answer")) {
            StopListening();           // ensure mic off
            Speak(last_answer_text_);  // Java will request audio focus
        }
        if (!canSpeak) ImGui::EndDisabled();

        ImGui::SameLine();
        if (ImGui::Button("Test TTS")) {
            StopListening();
            Speak("Text to speech is working.");
            onnx_status_message_ = "TTs testing done... .";
        }

// Auto-speak once per new answer (guarded by sequence)
        static uint64_t last_spoken_seq = 0;
        const uint64_t seq = answer_seq_.load(std::memory_order_relaxed);
        if (speak_auto_ && seq != 0 && seq != last_spoken_seq) {
            const std::string ans = last_answer_text_;
            if (!ans.empty() && ans != "(empty)") {
                StopListening();
                Speak(ans);
                last_spoken_seq = seq;
            }
        }
//........................

        // Speak the last "Answer:" once
        static std::string last_spoken;
        if (!onnx_status_message_.empty()) {
            size_t p = onnx_status_message_.rfind("Answer:");
            if (p != std::string::npos) {
                size_t nl = onnx_status_message_.find('\n', p);
                std::string answer = onnx_status_message_.substr(
                        p + 7, (nl==std::string::npos? std::string::npos : nl - (p+7)));
                if (!answer.empty() && answer != "(empty)" && answer != last_spoken) {
                    Speak(answer);
                    last_spoken = answer;
                }
            }
        }

        gui.EndDialog();   // keep EndDialog inside the if (BeginDialog) block
    }

    gui.EndUpdate();

    if (!app_running) {
        FinishActivity();
    }
}

//===========================endof update gui===========================================

// ===== Camera callbacks & helpers =====
void CameraMixedRealityApp::OnImageAvailable(const MLCameraOutput *output,
                                             const MLHandle /*metadata_handle*/,
                                             const MLCameraResultExtras *extra,
                                             void *data) {
    auto *this_app = reinterpret_cast<CameraMixedRealityApp *>(data);
    if (!this_app) return;

    const std::string k_file_ext = ".jpg";
    this_app->current_filename_photo_ =
            this_app->default_output_filename_photo_ + std::to_string(extra->vcam_timestamp) + k_file_ext;
    const std::string output_filename = this_app->default_output_filepath_ + this_app->current_filename_photo_;

    ALOGI("Image output filename: %s", output_filename.c_str());
    FILE* f = fopen(output_filename.c_str(), "wb");
    if (f) {
        fwrite(output->planes[0].data, output->planes[0].size, 1, f);
        fflush(f);
        int fd = fileno(f);
        if (fd >= 0) fsync(fd);
        fclose(f);

        // Publish via stable alias so the app always has a valid path
        this_app->PersistLastImageAlias(output_filename);


        if (this_app->pending_voice_question_) { /* ... unchanged ... */ }
        else if (this_app->send_to_vlm_after_capture_) { /* ... unchanged ... */ }
    } else {
        ALOGE("Failed to open %s, with error: %s!", output_filename.c_str(), strerror(errno));
    }

}

 //............................................................

MLResult CameraMixedRealityApp::CaptureImage() {
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

MLResult CameraMixedRealityApp::DestroyCamera() {
    if (MLHandleIsValid(recorder_camera_context_)) {
        UNWRAP_RET_MEDIARESULT(MLCameraDisconnect(recorder_camera_context_));
        recorder_camera_context_ = ML_INVALID_HANDLE;
        recorder_camera_device_available_ = false;
    }
    UNWRAP_RET_MEDIARESULT(MLCameraDeInit());
    return MLResult_Ok;
}

MLResult CameraMixedRealityApp::SetupCamera() {
    if (MLHandleIsValid(recorder_camera_context_)) return MLResult_Ok;

    MLCameraDeviceAvailabilityStatusCallbacks device_availability_status_callbacks = {};
    MLCameraDeviceAvailabilityStatusCallbacksInit(&device_availability_status_callbacks);
    device_availability_status_callbacks.on_device_available = [](const MLCameraDeviceAvailabilityInfo *avail_info) {
        CheckDeviceAvailability(avail_info, true);
    };
    device_availability_status_callbacks.on_device_unavailable = [](const MLCameraDeviceAvailabilityInfo *avail_info) {
        CheckDeviceAvailability(avail_info, false);
    };

    UNWRAP_RET_MEDIARESULT(MLCameraInit(&device_availability_status_callbacks, this));

    { // wait up to 2 seconds until camera becomes available
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

void CameraMixedRealityApp::CheckDeviceAvailability(const MLCameraDeviceAvailabilityInfo *device_availability_info,
                                                    bool is_available) {
    if (!device_availability_info) return;
    auto *this_app = static_cast<CameraMixedRealityApp *>(device_availability_info->user_data);
    if (this_app && device_availability_info->cam_id == MLCameraIdentifier_MAIN) {
        this_app->recorder_camera_device_available_ = is_available;
        this_app->camera_device_available_condition_.notify_one();
    }
}

MLResult CameraMixedRealityApp::SetCameraRecorderCallbacks() {
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
    UNWRAP_RET_MEDIARESULT(MLCameraSetDeviceStatusCallbacks(recorder_camera_context_, &camera_device_status_callbacks, this));

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

MLResult CameraMixedRealityApp::SetupCaptureSize() {
    int32_t width = 0, height = 0;
    uint32_t streams_max = 0;
    UNWRAP_RET_MLRESULT(MLCameraGetNumSupportedStreams(recorder_camera_context_, &streams_max));

    struct StreamCapsInfo {
        uint32_t stream_caps_max;
        MLCameraCaptureStreamCaps *stream_caps;
    };

    StreamCapsInfo *stream_caps_info = (StreamCapsInfo *)malloc(streams_max * sizeof(StreamCapsInfo));
    if (!stream_caps_info) {
        ALOGE("Memory Allocation for StreamCapsInfo failed");
        return MLResult_UnspecifiedFailure;
    }

    for (uint32_t i = 0; i < streams_max; i++) {
        stream_caps_info[i].stream_caps_max = 0;
        stream_caps_info[i].stream_caps = nullptr;
        UNWRAP_RET_MLRESULT(MLCameraGetStreamCaps(recorder_camera_context_, i, &stream_caps_info[i].stream_caps_max, nullptr));
        stream_caps_info[i].stream_caps = (MLCameraCaptureStreamCaps *)malloc(stream_caps_info[i].stream_caps_max * sizeof(MLCameraCaptureStreamCaps));
        UNWRAP_RET_MLRESULT(MLCameraGetStreamCaps(recorder_camera_context_, i, &stream_caps_info[i].stream_caps_max, stream_caps_info[i].stream_caps));

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
        if (stream_caps_info[i].stream_caps != nullptr) free(stream_caps_info[i].stream_caps);
    }
    free(stream_caps_info);

    if (width > 0 && height > 0) {
        capture_width_ = width;
        capture_height_ = height;
    }
    return MLResult_Ok;
}

// ===== Small file utils used by ONNX path =====
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

static bool LoadGpt2VocabIdToToken(const std::string& vocab_path, std::vector<std::string>& id_to_token) {
    std::string json;
    if (!ReadFileToString(vocab_path, json)) return false;

    std::regex entry(R"xxx("([^"\\]|\\.)*"\s*:\s*\d+)xxx");
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
            std::string clean; clean.reserve(tok.size());
            for (size_t i=0;i<tok.size();++i) {
                if (tok[i]=='\\' && i+1<tok.size()) {
                    char c = tok[i+1];
                    if (c=='"' || c=='\\' || c=='/') { clean.push_back(c); ++i; }
                    else if (c=='n') { clean.push_back('\n'); ++i; }
                    else if (c=='t') { clean.push_back('\t'); ++i; }
                    else { clean.push_back(tok[i]); }
                } else clean.push_back(tok[i]);
            }
            items.emplace_back(id, clean);
            if (id > max_id) max_id = id;
        }
    }

    if (items.empty()) return false;
    id_to_token.assign(max_id+1, std::string());
    for (auto& kv : items) if (kv.first < id_to_token.size()) id_to_token[kv.first] = kv.second;
    return true;
}

static std::string SimpleDecodeGpt2(const std::vector<int64_t>& ids,
                                    const std::vector<std::string>& id_to_token) {
    std::string s; s.reserve(ids.size()*3);
    for (auto id : ids) {
        if (id >= 0 && (size_t)id < id_to_token.size()) {
            const std::string& t = id_to_token[(size_t)id];
            if (t == "<|endoftext|>" || t == "" || t == " ") continue;
            s += t;
        }
    }
    auto replace_all = [&](const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) { s.replace(pos, from.size(), to); pos += to.size(); }
    };
    replace_all("Ġ", " ");
    replace_all("Ċ", "\n");
    return s;
}

static void ReplaceAll(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) { s.replace(pos, from.size(), to); pos += to.size(); }
}
//----------------------------keyboard stufss start-----------------------------------
void CameraMixedRealityApp::ShowKeyboard() {
    if (!app_state_ || !app_state_->activity || !g_VoiceBridgeObj || !g_VoiceBridgeCls) return;
    JNIEnv* env = nullptr; app_state_->activity->vm->AttachCurrentThread(&env, nullptr);
    if (!env) return;
    jmethodID m = env->GetMethodID(g_VoiceBridgeCls, "showKeyboard", "()V");
    if (!m) { ALOGE("showKeyboard not found"); return; }
    env->CallVoidMethod(g_VoiceBridgeObj, m);
    if (env->ExceptionCheck()) { env->ExceptionDescribe(); env->ExceptionClear(); }
}

void CameraMixedRealityApp::OpenEditDialog(const std::string& initial) {
    if (!app_state_ || !app_state_->activity || !g_VoiceBridgeObj || !g_VoiceBridgeCls) return;
    JNIEnv* env = nullptr; app_state_->activity->vm->AttachCurrentThread(&env, nullptr);
    if (!env) return;
    jmethodID m = env->GetMethodID(g_VoiceBridgeCls, "openEditDialog", "(Ljava/lang/String;)V");
    if (!m) { ALOGE("openEditDialog(String) not found"); return; }
    jstring jinitial = env->NewStringUTF(initial.c_str());
    env->CallVoidMethod(g_VoiceBridgeObj, m, jinitial);
    env->DeleteLocalRef(jinitial);
    if (env->ExceptionCheck()) { env->ExceptionDescribe(); env->ExceptionClear(); }
}

//----------------------------------keyboardstuffs end-------------------------------------

// =================== ONNX / VLM ===================start..........
void CameraMixedRealityApp::InitializeONNX(const std::string& image_path,
                                           const std::string& question_text) {
    // ----- load sessions once -----
    if (!EnsureVLMLoaded()) {
        onnx_status_message_ = "VLM not ready (load failed).";
        return;
    }
    if (!ort_) ort_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    // Prevent overlapping runs
    static std::atomic_flag busy = ATOMIC_FLAG_INIT;
    if (busy.test_and_set()) {
        onnx_status_message_ = "ONNX is already running…";
        return;
    }
    auto clear_busy = [&]() { busy.clear(); };

    // We'll rebuild the whole panel in one shot to avoid flicker/duplication
    std::string panel;

    // ------- tiny helpers -------
    auto FileExists = [](const std::string &p) -> bool {
        FILE *f = fopen(p.c_str(), "rb");
        if (!f) return false;
        fclose(f);
        return true;
    };
    auto ReadFile = [](const std::string &p, std::string &out) -> bool {
        FILE *f = fopen(p.c_str(), "rb");
        if (!f) return false;
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        if (n < 0) {
            fclose(f);
            return false;
        }
        fseek(f, 0, SEEK_SET);
        out.resize((size_t) n);
        size_t r = fread(out.data(), 1, (size_t) n, f);
        fclose(f);
        return r == (size_t) n;
    };
    auto ReplaceAllLocal = [](std::string &s, const std::string &from, const std::string &to) {
        if (from.empty()) return;
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    auto LoadImageCHW224F32 = [&](const std::string &path, std::vector<float> &out, int &W,
                                  int &H) -> bool {
        int w = 0, h = 0, c = 0;
        unsigned char *img = stbi_load(path.c_str(), &w, &h, &c, 3);
        if (!img) return false;
        const int outW = 224, outH = 224, ch = 3;
        out.assign((size_t) ch * outH * outW, 0.f);
        auto samp = [&](float x, float y, int chn) -> float {
            int ix = (int) std::roundf(x), iy = (int) std::roundf(y);
            if (ix < 0) ix = 0;
            if (ix >= w) ix = w - 1;
            if (iy < 0) iy = 0;
            if (iy >= h) iy = h - 1;
            int idx = (iy * w + ix) * 3 + chn;
            return (float) img[idx] / 255.0f;
        };
        for (int oy = 0; oy < outH; ++oy)
            for (int ox = 0; ox < outW; ++ox) {
                float sx = ((ox + 0.5f) * w / (float) outW) - 0.5f;
                float sy = ((oy + 0.5f) * h / (float) outH) - 0.5f;
                float r = samp(sx, sy, 0), g = samp(sx, sy, 1), b = samp(sx, sy, 2);
                r = (r - 0.5f) / 0.5f;
                g = (g - 0.5f) / 0.5f;
                b = (b - 0.5f) / 0.5f;
                int idx = oy * outW + ox;
                out[0 * outH * outW + idx] = r;
                out[1 * outH * outW + idx] = g;
                out[2 * outH * outW + idx] = b;
            }
        stbi_image_free(img);
        W = outW;
        H = outH;
        return true;
    };

    // ------- cached vocab + special tokens (load once, process-wide) -------
    using json = nlohmann::json;
    const std::string base = "/storage/emulated/0/Android/data/com.magicleap.capi.sample.camera_mixed_reality/files/models/";
    const std::string vocab_path = base + "vocab.json";
    const std::string genconf_path = base + "generation_config.json";

    static bool vocab_ok = false;
    static std::vector<std::string> id2tok;
    static int64_t start_id = 50256, eos_id = 50256;

    auto EnsureVocab = [&]() -> bool {
        if (vocab_ok) return true;
        std::string s;
        if (!ReadFile(vocab_path, s)) {
            panel += "vocab.json missing.\n";
            return false;
        }
        json j = json::parse(s, nullptr, false);
        if (j.is_discarded() || !j.is_object()) {
            panel += "vocab.json invalid.\n";
            return false;
        }
        int64_t max_id = -1;
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (!it.value().is_number_integer()) continue;
            int64_t id = it.value().get<int64_t>();
            if (id > max_id) max_id = id;
        }
        if (max_id < 0) {
            panel += "vocab.json empty.\n";
            return false;
        }
        id2tok.clear();
        id2tok.resize((size_t) max_id + 1);
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (!it.value().is_number_integer()) continue;
            int64_t id = it.value().get<int64_t>();
            if (id >= 0 && (size_t) id < id2tok.size()) id2tok[(size_t) id] = it.key();
        }
        // special tokens from generation_config.json if present
        std::string g;
        if (ReadFile(genconf_path, g)) {
            auto pull_int = [&](const char *key, int64_t &out) {
                size_t k = g.find(key);
                if (k == std::string::npos) return;
                k = g.find(':', k);
                if (k == std::string::npos) return;
                ++k;
                while (k < g.size() && (g[k] == ' ' || g[k] == '\t')) ++k;
                bool neg = false;
                if (k < g.size() && (g[k] == '-' || g[k] == '+')) {
                    neg = (g[k] == '-');
                    ++k;
                }
                long long v = 0;
                while (k < g.size() && g[k] >= '0' && g[k] <= '9') {
                    v = v * 10 + (g[k] - '0');
                    ++k;
                }
                out = neg ? -v : v;
            };
            pull_int("decoder_start_token_id", start_id);
            pull_int("bos_token_id", start_id);
            pull_int("eos_token_id", eos_id);
        }
        for (size_t i = 0; i < id2tok.size(); ++i)
            if (id2tok[i] == "<|endoftext|>") {
                eos_id = (int64_t) i;
                break;
            }
        vocab_ok = true;
        return true;
    };
    if (!EnsureVocab()) {
        onnx_status_message_ = panel;
        clear_busy();
        return;
    }

    // ------- Tokenizer helpers (GPT2-like) -------
    auto TokenizeQuestion = [&](const std::string &question) -> std::vector<int64_t> {
        std::unordered_map<std::string, int64_t> tok2id;
        tok2id.reserve(id2tok.size());
        for (int64_t i = 0; i < (int64_t) id2tok.size(); ++i)
            if (!id2tok[(size_t) i].empty()) tok2id[id2tok[(size_t) i]] = i;
        auto find_tok = [&](const std::string &t) -> int64_t {
            auto it = tok2id.find(t);
            return (it == tok2id.end() ? -1 : it->second);
        };

        std::vector<int64_t> ids;
        auto push_if = [&](const std::string &t) {
            int64_t id = find_tok(t);
            if (id >= 0) ids.push_back(id);
        };
        push_if("ĠQuestion");
        push_if(":");

        std::istringstream iss(question);
        std::string w;
        while (iss >> w) {
            int64_t id = -1;
            if ((id = find_tok("Ġ" + w)) < 0) id = find_tok(w);
            if (id < 0) {
                std::string wl = w;
                std::transform(wl.begin(), wl.end(), wl.begin(), ::tolower);
                if ((id = find_tok("Ġ" + wl)) < 0) id = find_tok(wl);
            }
            if (id >= 0) ids.push_back(id);
        }
        if (question.find('?') != std::string::npos) {
            int64_t qid = find_tok("?");
            if (qid >= 0) ids.push_back(qid);
        }
        push_if("ĠAnswer");
        push_if(":");
        return ids;
    };
    auto SimpleDecodeGpt2Local = [&](const std::vector<int64_t> &ids) -> std::string {
        std::string out;
        out.reserve(ids.size() * 3);
        for (auto id: ids) {
            if (id < 0 || (size_t) id >= id2tok.size()) continue;
            std::string t = id2tok[(size_t) id];
            ReplaceAllLocal(t, u8"Ġ", " ");
            ReplaceAllLocal(t, u8"Ċ", "\n");
            ReplaceAllLocal(t, "<|endoftext|>", "");
            out += t;
        }
        // collapse double spaces
        for (size_t i = 1; i < out.size();) {
            if (out[i] == ' ' && out[i - 1] == ' ')
                out.erase(i, 1);
            else ++i;
        }
        // trim
        while (!out.empty() && (out.back() == ' ' || out.back() == '\n')) out.pop_back();
        while (!out.empty() && (out.front() == ' ' || out.front() == '\n')) out.erase(out.begin());
        return out;
    };

    // ------- validate image -------
    std::string use_path = image_path;
    if (use_path.empty() || !FileExistsAbs(use_path)) {
        if (!FileExistsAbs(AliasPath())) {
            onnx_status_message_ = "Image not found; please capture one.";
            clear_busy();
            return;
        }
        use_path = AliasPath();
    }


    // ------- prompt engineering: center focus + optional detail -------
    std::string q = question_text;
    auto tolower_copy = [](std::string s) {
        for (char &c: s)
            c = (char) std::tolower((unsigned char) c);
        return s;
    };
    std::string ql = tolower_copy(q);

    const bool wants_detail =
            (ql.find("detail") != std::string::npos) ||
            (ql.find("explain") != std::string::npos) ||
            (ql.find("describe") != std::string::npos) ||
            q.empty();

    // Inject center-focus + style hint while preserving user intent
//    if (q.empty()) {
//        q = "Describe the main object in the center of the image. Provide 2–4 clear sentences.";
//    } else {
//        q += wants_detail
//             ? " Focus on the main object in the center of the image and provide 2–4 clear sentences."
//             : " Focus on the main object in the center of the image and answer concisely (5–6 sentences).";
//    }

    // ------- image preprocess -------
    std::vector<float> pixel_values;
    int W = 0, H = 0;
    if (!LoadImageCHW224F32(image_path, pixel_values, W, H)) {
        onnx_status_message_ = "Failed to load image.";
        clear_busy();
        return;
    }

    // ------- encoder I/O names -------
    std::string enc_in_name = "pixel_values";
    std::string enc_out_name = "last_hidden_state";
    {
        size_t nout = 0;
        ort_->SessionGetOutputCount(encoder_session_, &nout);
        if (nout > 0) {
            OrtAllocator *alloc = nullptr;
            ort_->GetAllocatorWithDefaultOptions(&alloc);
            char *nm = nullptr;
            if (!ort_->SessionGetOutputName(encoder_session_, 0, alloc, &nm) && nm) {
                enc_out_name = nm;
                alloc->Free(alloc, nm);
            }
        }
    }

    // ------- run encoder -------
    OrtMemoryInfo *mi = nullptr;
    ort_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi);
    int64_t enc_shape[4] = {1, 3, (int64_t) H, (int64_t) W};
    OrtValue *enc_input = nullptr;
    ort_->CreateTensorWithDataAsOrtValue(mi, pixel_values.data(),
                                         pixel_values.size() * sizeof(float),
                                         enc_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                         &enc_input);
    const char *enc_in_names[1] = {enc_in_name.c_str()};
    const char *enc_out_names[1] = {enc_out_name.c_str()};
    OrtValue *enc_output = nullptr;
    if (OrtStatus *st = ort_->Run(encoder_session_, nullptr, enc_in_names, &enc_input, 1,
                                  enc_out_names, 1, &enc_output)) {
        ort_->ReleaseStatus(st);
        if (enc_input) ort_->ReleaseValue(enc_input);
        if (mi) ort_->ReleaseMemoryInfo(mi);
        onnx_status_message_ = "Encoder run failed.";
        clear_busy();
        return;
    }

    std::vector<int64_t> enc_out_dims;
    std::vector<float> enc_feat;
    if (enc_output) {
        OrtTensorTypeAndShapeInfo *ti = nullptr;
        ort_->GetTensorTypeAndShape(enc_output, &ti);
        size_t nd = 0;
        if (ti) ort_->GetDimensionsCount(ti, &nd);
        enc_out_dims.resize(nd, 0);
        if (ti && nd) ort_->GetDimensions(ti, enc_out_dims.data(), nd);
        float *p = nullptr;
        ort_->GetTensorMutableData(enc_output, (void **) &p);
        size_t cnt = 1;
        for (auto d: enc_out_dims) {
            if (d < 1) d = 1;
            cnt *= (size_t) d;
        }
        if (p && cnt) enc_feat.assign(p, p + cnt);
        if (ti) ort_->ReleaseTensorTypeAndShapeInfo(ti);
    }
    if (enc_input) ort_->ReleaseValue(enc_input);
    if (enc_output) ort_->ReleaseValue(enc_output);
    if (enc_feat.empty()) {
        if (mi) ort_->ReleaseMemoryInfo(mi);
        onnx_status_message_ = "Empty encoder features.";
        clear_busy();
        return;
    }

    // ------- decoder names -------
    std::string name_input_ids = "input_ids";
    std::string name_enc_hs = "encoder_hidden_states";
    std::string name_logits = "logits";
    std::string name_attn;
    std::string name_enc_attn;
    { // inputs
        size_t nin = 0;
        ort_->SessionGetInputCount(decoder_session_, &nin);
        OrtAllocator *alloc = nullptr;
        ort_->GetAllocatorWithDefaultOptions(&alloc);
        for (size_t i = 0; i < nin; ++i) {
            char *nm = nullptr;
            if (!ort_->SessionGetInputName(decoder_session_, i, alloc, &nm) && nm) {
                std::string s(nm);
                alloc->Free(alloc, nm);
                if (s.find("input_ids") != std::string::npos) name_input_ids = s;
                else if (s.find("encoder_hidden_states") != std::string::npos ||
                         s.find("encoder_outputs") != std::string::npos ||
                         s.find("encoder_out") != std::string::npos)
                    name_enc_hs = s;
                else if (s.find("encoder_attention_mask") != std::string::npos) name_enc_attn = s;
                else if (s.find("attention_mask") != std::string::npos) name_attn = s;
            }
        }
    }
    { // outputs
        size_t nout = 0;
        ort_->SessionGetOutputCount(decoder_session_, &nout);
        OrtAllocator *alloc = nullptr;
        ort_->GetAllocatorWithDefaultOptions(&alloc);
        if (nout > 0) {
            char *nm = nullptr;
            if (!ort_->SessionGetOutputName(decoder_session_, 0, alloc, &nm) && nm) {
                name_logits = nm;
                alloc->Free(alloc, nm);
            }
        }
    }

    ONNXTensorElementDataType ids_elem = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    auto GetInputDTypeByName = [&](OrtSession *s, const std::string &name,
                                   ONNXTensorElementDataType &out) -> bool {
        size_t nin = 0;
        ort_->SessionGetInputCount(s, &nin);
        OrtAllocator *alloc = nullptr;
        ort_->GetAllocatorWithDefaultOptions(&alloc);
        for (size_t i = 0; i < nin; ++i) {
            char *nm = nullptr;
            if (!ort_->SessionGetInputName(s, i, alloc, &nm) && nm) {
                std::string got(nm);
                alloc->Free(alloc, nm);
                if (got == name) {
                    OrtTypeInfo *ti = nullptr;
                    if (!ort_->SessionGetInputTypeInfo(s, i, &ti) && ti) {
                        const OrtTensorTypeAndShapeInfo *tti = nullptr;
                        if (!ort_->CastTypeInfoToTensorInfo(ti, &tti) && tti) {
                            ort_->GetTensorElementType(tti, &out);
                            ort_->ReleaseTypeInfo(ti);
                            return true;
                        }
                        ort_->ReleaseTypeInfo(ti);
                    }
                    return false;
                }
            }
        }
        return false;
    };
    (void) GetInputDTypeByName(decoder_session_, name_input_ids, ids_elem);

    // ------- build decoder feeds -------
    std::vector<int64_t> input_ids;
    input_ids.push_back(start_id);
    {
        auto q_ids = TokenizeQuestion(q);
        input_ids.insert(input_ids.end(), q_ids.begin(), q_ids.end());
    }

    OrtValue *t_ids = nullptr;
    int64_t ids_shape[2] = {1, (int64_t) input_ids.size()};
    if (ids_elem == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        std::vector<int32_t> ids32(input_ids.begin(), input_ids.end());
        ort_->CreateTensorWithDataAsOrtValue(mi, ids32.data(), ids32.size() * sizeof(int32_t),
                                             ids_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                                             &t_ids);
    } else {
        ort_->CreateTensorWithDataAsOrtValue(mi, input_ids.data(),
                                             input_ids.size() * sizeof(int64_t),
                                             ids_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                                             &t_ids);
    }

    OrtValue *t_hs = nullptr;
    ort_->CreateTensorWithDataAsOrtValue(mi, enc_feat.data(), enc_feat.size() * sizeof(float),
                                         enc_out_dims.data(), enc_out_dims.size(),
                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_hs);

    // attention masks (optional depending on model)
    OrtValue *t_attn = nullptr, *t_enc_attn = nullptr;
    auto build_attn_1xL = [&](const std::string &name, OrtValue **out, int64_t L) {
        if (name.empty()) return;
        ONNXTensorElementDataType elem = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        GetInputDTypeByName(decoder_session_, name, elem);
        int64_t shape[2] = {1, L};
        if (elem == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            std::vector<int32_t> ones((size_t) L, 1);
            ort_->CreateTensorWithDataAsOrtValue(mi, ones.data(), ones.size() * sizeof(int32_t),
                                                 shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                                                 out);
        } else {
            std::vector<int64_t> ones((size_t) L, 1);
            ort_->CreateTensorWithDataAsOrtValue(mi, ones.data(), ones.size() * sizeof(int64_t),
                                                 shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                                                 out);
        }
    };
    build_attn_1xL(name_attn, &t_attn, ids_shape[1]);
    {
        int64_t enc_L = (enc_out_dims.size() >= 2) ? enc_out_dims[1] : (int64_t) enc_feat.size();
        if (enc_L <= 0) enc_L = 1;
        build_attn_1xL(name_enc_attn, &t_enc_attn, enc_L);
    }

    // ------- decode loop (greedy) -------
    std::vector<int64_t> generated;
    const int max_steps = wants_detail ? 96 : 64;
    const int min_new_tokens = wants_detail ? 24 : 14;

    for (int step = 0; step < max_steps; ++step) {
        const char *fetch_names[1] = {name_logits.c_str()};
        OrtValue *fetches[1] = {nullptr};

        const char *feed_names[4];
        OrtValue *feed_vals[4];
        size_t feed_count = 0;
        feed_names[feed_count] = name_input_ids.c_str();
        feed_vals[feed_count] = t_ids;
        ++feed_count;
        feed_names[feed_count] = name_enc_hs.c_str();
        feed_vals[feed_count] = t_hs;
        ++feed_count;
        if (t_attn) {
            feed_names[feed_count] = name_attn.c_str();
            feed_vals[feed_count] = t_attn;
            ++feed_count;
        }
        if (t_enc_attn) {
            feed_names[feed_count] = name_enc_attn.c_str();
            feed_vals[feed_count] = t_enc_attn;
            ++feed_count;
        }

        OrtStatus *st = ort_->Run(decoder_session_, nullptr,
                                  feed_names, feed_vals, feed_count,
                                  fetch_names, 1, fetches);
        if (st) {
            ort_->ReleaseStatus(st);
            panel += "Decoder run failed.\n";
            break;
        }
        OrtTensorTypeAndShapeInfo *ti = nullptr;
        ort_->GetTensorTypeAndShape(fetches[0], &ti);
        std::vector<int64_t> lshape;
        if (ti) {
            size_t nd = 0;
            ort_->GetDimensionsCount(ti, &nd);
            lshape.resize(nd, 0);
            if (nd) ort_->GetDimensions(ti, lshape.data(), nd);
        }
        float *logits = nullptr;
        ort_->GetTensorMutableData(fetches[0], (void **) &logits);

        int64_t next_id = eos_id;
        if (logits && lshape.size() == 3 && lshape[1] > 0 && lshape[2] > 0) {
            const int64_t seq_len = lshape[1], vocab = lshape[2];
            float *last = logits + (seq_len - 1) * vocab;

            int64_t best_i = 0;
            float best_v = last[0];
            for (int64_t i = 1; i < vocab; ++i)
                if (last[i] > best_v) {
                    best_v = last[i];
                    best_i = i;
                }
            // prevent too-early EOS
            if (best_i == eos_id && step < min_new_tokens) {
                int64_t second = best_i;
                float second_v = -1e30f;
                for (int64_t i = 0; i < vocab; ++i) {
                    if (i == eos_id) continue;
                    if (last[i] > second_v) {
                        second_v = last[i];
                        second = i;
                    }
                }
                next_id = second;
            } else next_id = best_i;
        }
        if (ti) ort_->ReleaseTensorTypeAndShapeInfo(ti);
        ort_->ReleaseValue(fetches[0]);

        generated.push_back(next_id);
        input_ids.push_back(next_id);
        if (next_id == eos_id && (int) generated.size() >= min_new_tokens) break;

        // grow input_ids tensor (+ attention)
        int64_t new_len = (int64_t) input_ids.size();
        int64_t ids_shape2[2] = {1, new_len};
        if (t_ids) {
            ort_->ReleaseValue(t_ids);
            t_ids = nullptr;
        }
        if (ids_elem == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            std::vector<int32_t> ids32(input_ids.begin(), input_ids.end());
            ort_->CreateTensorWithDataAsOrtValue(mi, ids32.data(), ids32.size() * sizeof(int32_t),
                                                 ids_shape2, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                                                 &t_ids);
        } else {
            ort_->CreateTensorWithDataAsOrtValue(mi, input_ids.data(),
                                                 input_ids.size() * sizeof(int64_t),
                                                 ids_shape2, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                                                 &t_ids);
        }
        if (t_attn) {
            ONNXTensorElementDataType elem = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
            GetInputDTypeByName(decoder_session_, name_attn, elem);
            if (elem == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
                std::vector<int32_t> ones((size_t) new_len, 1);
                ort_->ReleaseValue(t_attn);
                t_attn = nullptr;
                ort_->CreateTensorWithDataAsOrtValue(mi, ones.data(), ones.size() * sizeof(int32_t),
                                                     ids_shape2, 2,
                                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &t_attn);
            } else {
                std::vector<int64_t> ones((size_t) new_len, 1);
                ort_->ReleaseValue(t_attn);
                t_attn = nullptr;
                ort_->CreateTensorWithDataAsOrtValue(mi, ones.data(), ones.size() * sizeof(int64_t),
                                                     ids_shape2, 2,
                                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_attn);
            }
        }
    }

    // ------- decode text & build panel once (no IDs printed) -------
    // ------- decode text & update state once -------
    std::string decoded = SimpleDecodeGpt2Local(generated);
    if (decoded.empty()) decoded = "(empty)";

// Normalize & store once per VLM run (used for TTS/UI)
    std::string decoded_norm = TrimCollapse(decoded);   // ensure TrimCollapse helper exists
    last_answer_text_ = decoded_norm;
    const uint64_t seq_now = answer_seq_.fetch_add(1, std::memory_order_relaxed) + 1;

// Build the visible status text exactly once
// (Use the SAME 'panel' string you created earlier at the top of InitializeONNX)
    panel.clear();
    panel.reserve(q.size() + decoded_norm.size() + 32);
    panel += "Q: " + q + "\n";
    panel += "Answer: " + decoded_norm + "\n";

// ------- cleanup (no session/env teardown) -------
    if (t_ids) ort_->ReleaseValue(t_ids);
    if (t_hs) ort_->ReleaseValue(t_hs);
    if (t_attn) ort_->ReleaseValue(t_attn);
    if (t_enc_attn) ort_->ReleaseValue(t_enc_attn);
    if (mi) ort_->ReleaseMemoryInfo(mi);

    onnx_initialized_ = true;
    onnx_status_message_ = panel;   // single write to avoid dup prints
    clear_busy();
}
//..............................onnxend ....................................

// =================== Voice bridge (C++ → Java) ===================
void CameraMixedRealityApp::InitVoice() {
    if (!app_state_ || !app_state_->activity) { ALOGE("InitVoice: app_state/activity null"); return; }

    JNIEnv* env = nullptr;
    if (app_state_->activity->vm->AttachCurrentThread(&env, nullptr) != JNI_OK || !env) {
        ALOGE("InitVoice: AttachCurrentThread failed");
        return;
    }
    jobject activityObj = app_state_->activity->clazz;

    jclass voiceCls = LoadClassFromActivity(env, activityObj,
                                            "com.magicleap.capi.sample.camera_mixed_reality.VoiceBridge");
    if (!voiceCls) {
        ALOGE("InitVoice: VoiceBridge class NOT found (check package/class name)");
        if (env->ExceptionCheck()) { env->ExceptionDescribe(); env->ExceptionClear(); }
        return;
    }
    g_VoiceBridgeCls = (jclass)env->NewGlobalRef(voiceCls);

    jmethodID ctor = env->GetMethodID(g_VoiceBridgeCls, "<init>", "(Landroid/app/Activity;)V");
    if (!ctor) { ALOGE("InitVoice: constructor not found"); return; }

    jobject objLocal = env->NewObject(g_VoiceBridgeCls, ctor, activityObj);
    if (!objLocal) { ALOGE("InitVoice: NewObject failed"); return; }
    g_VoiceBridgeObj = env->NewGlobalRef(objLocal);

    jmethodID init = env->GetMethodID(g_VoiceBridgeCls, "init", "()V");
    if (!init) { ALOGE("InitVoice: init() not found"); return; }

    env->CallVoidMethod(g_VoiceBridgeObj, init);
    if (env->ExceptionCheck()) { env->ExceptionDescribe(); env->ExceptionClear(); ALOGE("InitVoice: exception from init()"); }
}

void CameraMixedRealityApp::DestroyVoice() {
    if (!app_state_ || !app_state_->activity) return;
    JNIEnv* env = nullptr; app_state_->activity->vm->AttachCurrentThread(&env, nullptr);
    if (!env) return;

    if (g_VoiceBridgeObj && g_VoiceBridgeCls) {
        jmethodID destroy = env->GetMethodID(g_VoiceBridgeCls, "destroy", "()V");
        if (destroy) env->CallVoidMethod(g_VoiceBridgeObj, destroy);
        if (env->ExceptionCheck()) { env->ExceptionDescribe(); env->ExceptionClear(); }
        env->DeleteGlobalRef(g_VoiceBridgeObj);
        g_VoiceBridgeObj = nullptr;
    }
    if (g_VoiceBridgeCls) { env->DeleteGlobalRef(g_VoiceBridgeCls); g_VoiceBridgeCls = nullptr; }
}

void CameraMixedRealityApp::StartListening() {
    if (!app_state_ || !app_state_->activity || !g_VoiceBridgeObj || !g_VoiceBridgeCls) return;
    JNIEnv* env = nullptr; app_state_->activity->vm->AttachCurrentThread(&env, nullptr);
    if (!env) return;
    jmethodID m = env->GetMethodID(g_VoiceBridgeCls, "startListening", "()V");
    if (!m) { ALOGE("startListening not found"); return; }
    env->CallVoidMethod(g_VoiceBridgeObj, m);
    if (env->ExceptionCheck()) { env->ExceptionDescribe(); env->ExceptionClear(); }
}

void CameraMixedRealityApp::StopListening() {
    if (!app_state_ || !app_state_->activity || !g_VoiceBridgeObj || !g_VoiceBridgeCls) return;
    JNIEnv* env = nullptr; app_state_->activity->vm->AttachCurrentThread(&env, nullptr);
    if (!env) return;
    jmethodID m = env->GetMethodID(g_VoiceBridgeCls, "stopListening", "()V");
    if (!m) { ALOGE("stopListening not found"); return; }
    env->CallVoidMethod(g_VoiceBridgeObj, m);
    if (env->ExceptionCheck()) { env->ExceptionDescribe(); env->ExceptionClear(); }
}

void CameraMixedRealityApp::Speak(const std::string& text) {
    ALOGI("Speak(): about to call Java speak(), text='%s'", text.c_str());

    if (!app_state_ || !app_state_->activity || !g_VoiceBridgeObj) {
        ALOGE("Speak(): app_state/activity/VoiceBridgeObj is null");
        return;
    }

    JNIEnv* env = nullptr;
    bool did_attach = false;
    JavaVM* jvm = app_state_->activity->vm;

    // Attach only if needed
    if (jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        if (jvm->AttachCurrentThread(&env, nullptr) != JNI_OK || !env) {
            ALOGE("Speak(): AttachCurrentThread failed");
            return;
        }
        did_attach = true;
    }

    // Always resolve method on the runtime class of the object (safer than cached class)
    jclass cls = env->GetObjectClass(g_VoiceBridgeObj);
    if (!cls) {
        ALOGE("Speak(): GetObjectClass returned null");
        if (did_attach) jvm->DetachCurrentThread();
        return;
    }

    jmethodID mid = env->GetMethodID(cls, "speak", "(Ljava/lang/String;)V");
    if (!mid) {
        ALOGE("Speak(): method 'speak(String)' not found on VoiceBridge");
        if (env->ExceptionCheck()) { env->ExceptionDescribe(); env->ExceptionClear(); }
        if (did_attach) jvm->DetachCurrentThread();
        return;
    }

    jstring jtext = env->NewStringUTF(text.c_str());
    if (!jtext) {
        ALOGE("Speak(): NewStringUTF failed");
        if (did_attach) jvm->DetachCurrentThread();
        return;
    }

    env->CallVoidMethod(g_VoiceBridgeObj, mid, jtext);
    env->DeleteLocalRef(jtext);

    if (env->ExceptionCheck()) {
        ALOGE("Speak(): exception thrown by Java speak()");
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    if (did_attach) jvm->DetachCurrentThread();
}


// =================== ANDROID ENTRY ===================
void android_main(struct android_app *state) {
#ifndef ML_LUMIN
    ALOGE("This app is not supported on App Sim!");
#else
    CameraMixedRealityApp app(state);
    app.RunApp();
#endif
}