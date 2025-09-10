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
//.....................
#include <ml_eye_tracking.h>
#include <ml_head_tracking.h>
#include <ml_perception.h>
#include <ml_input.h>

#include <app_framework/node.h>
#include <app_framework/material/flat_material.h>
#include <app_framework/components/renderable_component.h>

#include <app_framework/material/flat_material.h>
#include <app_framework/components/renderable_component.h>


#ifdef ML_LUMIN
#include <GLES3/gl3.h>
#endif


//......................
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

#include <unordered_set>  // for DedupSentences() 'seen' set
#include <cctype>         // for std::isspace / std::ispunct / std::tolower




// ===== ANDROID / JNI (for STT/TTS) =====
#include <jni.h>
#include <android_native_app_glue.h>
#include <android/native_activity.h>


#include <app_framework/node.h>
#include <app_framework/geometry/quad_mesh.h>
#include <app_framework/material/flat_material.h>
#include <app_framework/components/renderable_component.h>
#include <app_framework/convert.h>
#include <app_framework/material/flat_material.h>



// Quick write to JPEG using stb_image_write
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

// Mailbox for ASR results
static std::mutex g_asrMutex;
static std::string g_lastASR;
static std::string g_lastASRError;

// Java class/object handles
static jclass  g_VoiceBridgeCls = nullptr;
static jobject g_VoiceBridgeObj = nullptr;


static inline void ClearForMR() {
//#ifdef ML_LUMIN
//    glDisable(GL_SCISSOR_TEST);
//    glDisable(GL_STENCIL_TEST);
//
//    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
//    glDepthMask(GL_TRUE);
//
//    // Black, transparent: adds nothing to the camera feed
//    glClearColor(0.f, 0.f, 0.f, 0.f);
//    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//    // IMPORTANT: Re-enable blending so translucent things (ImGui, thin lines) don’t write opaque RGB
//    glEnable(GL_BLEND);
//    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//#endif
}

static inline void ImageRectOnPad(float hx, float hy, float aspect, float& sx, float& sy) {
    // fits a width:height = aspect:1 image inside a pad with half sizes (hx,hy)
    // returns half sizes (sx, sy) of the image rect inside the pad
    aspect = std::max(1e-6f, aspect);
    float s = std::min(hy, hx / aspect); // uniform scale to fit
    sx = s * aspect;
    sy = s;
}




#ifdef ML_LUMIN
static bool LoadTextureFromFileRGBA8888(const std::string& path,
                                        GLuint& out_tex, int& out_w, int& out_h) {
    int w=0,h=0,c=0;
    stbi_uc* data = stbi_load(path.c_str(), &w, &h, &c, 4);
    if (!data) return false;

    if (out_tex == 0) glGenTextures(1, &out_tex);
    glBindTexture(GL_TEXTURE_2D, out_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data);
    glBindTexture(GL_TEXTURE_2D, 0);

    stbi_image_free(data);
    out_w = w; out_h = h;
    return true;
}
#endif




static inline ImVec2 Clamp01(ImVec2 p) {
    return ImVec2(std::min(1.f, std::max(0.f, p.x)),
                  std::min(1.f, std::max(0.f, p.y)));
}
//...................................



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


static bool CropAndSaveSquareJPEG(
        const std::string& src_path,
        const std::string& dst_path,
        int cx, int cy, int crop_size_px)
{
    int w=0,h=0,c=0;
    unsigned char* img = stbi_load(src_path.c_str(), &w, &h, &c, 3);
    if (!img) return false;
    // clamp crop box
    int half = crop_size_px/2;
    int x0 = std::max(0, cx - half);
    int y0 = std::max(0, cy - half);
    int x1 = std::min(w, x0 + crop_size_px);
    int y1 = std::min(h, y0 + crop_size_px);
    int cw = x1 - x0, ch = y1 - y0;
    if (cw <= 0 || ch <= 0) { stbi_image_free(img); return false; }

    std::vector<unsigned char> crop((size_t)cw*ch*3);
    for (int y=0; y<ch; ++y) {
        memcpy(&crop[(size_t)y*cw*3],
               &img[((y0+y)*w + x0)*3],
               (size_t)cw*3);
    }
    stbi_image_free(img);

    int ok = stbi_write_jpg(dst_path.c_str(), cw, ch, 3, crop.data(), 90);
    return ok != 0;
}


//............................................
//static std::string TrimCollapse(const std::string& in) {
//    std::string out; out.reserve(in.size());
//    bool space=false;
//    for (char c: in) {
//        if (c==' '||c=='\t'||c=='\n'||c=='\r') { if (!space) { out.push_back(' '); space=true; } }
//        else { out.push_back(c); space=false; }
//    }
//    while (!out.empty() && out.front()==' ') out.erase(out.begin());
//    while (!out.empty() && out.back()==' ') out.pop_back();
//    return out;
//}
//.................................

//=================new of duplicate answer issue ===========
static std::string Trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && isspace((unsigned char)s[a])) ++a;
    while (b > a && isspace((unsigned char)s[b-1])) --b;
    return s.substr(a, b - a);
}

static std::string ToLower(std::string s) {
    for (char &c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

static std::string TrimCollapse(const std::string& in) {
    std::string out; out.reserve(in.size());
    bool space=false;
    for (char c: in) {
        if (c==' '||c=='\t'||c=='\n'||c=='\r') { if (!space) { out.push_back(' '); space=true; } }
        else { out.push_back(c); space=false; }
    }
    return Trim(out);
}

// Keep only text after the last "Answer:" (case-insensitive). If none, strip any "Question: ...".
static std::string ExtractAnswer(const std::string& decoded) {
    std::string s = decoded;
    std::string low = ToLower(s);
    const char* tags[] = { "answer:", "a:" };
    size_t pos = std::string::npos;
    for (auto tag : tags) {
        size_t p = low.rfind(tag);
        if (p != std::string::npos) { pos = std::max(pos, p + std::strlen(tag)); }
    }
    if (pos != std::string::npos) {
        return TrimCollapse(s.substr(pos));
    }
    // No explicit "Answer:" — try to remove "Question:" if present
    const char* qtags[] = { "question:", "q:" };
    size_t qpos = std::string::npos;
    for (auto tag : qtags) {
        size_t p = low.find(tag);
        if (p != std::string::npos) { qpos = p; break; }
    }
    if (qpos != std::string::npos) {
        // remove up to the end of that line
        size_t nl = s.find('\n', qpos);
        if (nl != std::string::npos) return TrimCollapse(s.substr(nl+1));
    }
    return TrimCollapse(s);
}

// Remove an echoed question if it appears inside the candidate answer
static std::string RemoveQuestionEcho(const std::string& answer, const std::string& question) {
    std::string a = answer, q = TrimCollapse(question);
    if (q.size() >= 12) {
        // case-insensitive erase
        std::string al = ToLower(a), ql = ToLower(q);
        size_t p = al.find(ql);
        if (p != std::string::npos) {
            a.erase(p, q.size());
        }
    }
    return TrimCollapse(a);
}
// Put near your other helpers (file scope)
static glm::quat LookRotationY(const glm::vec3& dir) {
    glm::vec3 y = glm::normalize(dir);
    glm::vec3 a = (fabsf(y.y) > 0.99f) ? glm::vec3(0,0,1) : glm::vec3(0,1,0);
    glm::vec3 x = glm::normalize(glm::cross(a, y));
    glm::vec3 z = glm::normalize(glm::cross(x, y));
    glm::mat3 m(x, y, z); // right, up(along ray), forward
    return glm::quat_cast(m);
}

// Deduplicate sentences while preserving order (very simple splitter)
static std::string DedupSentences(const std::string& text) {
    std::vector<std::string> parts;
    std::string cur;
    auto flush = [&](){
        std::string t = TrimCollapse(cur);
        if (!t.empty()) parts.push_back(t);
        cur.clear();
    };
    for (char c: text) {
        cur.push_back(c);
        if (c=='.' || c=='!' || c=='?' || c=='\n') flush();
    }
    flush();

    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    for (auto &p : parts) {
        std::string key = ToLower(p);
        if (seen.insert(key).second) out.push_back(p);
    }
    // Rebuild
    std::string res;
    for (size_t i=0;i<out.size();++i) {
        if (i) res += " ";
        res += out[i];
        if (!res.empty() && res.back()!='.' && res.back()!='!' && res.back()!='?') res += '.';
    }
    return TrimCollapse(res);
}

// Are two strings basically the same (used to avoid speaking the question)?
static bool IsBasicallySame(const std::string& a, const std::string& b) {
    auto norm = [](std::string s){
        s = ToLower(TrimCollapse(s));
        // strip common labels
        auto strip = [&](const char* t){ std::string sl=ToLower(s); size_t p; while ((p=sl.find(t))!=std::string::npos){ s.erase(p, std::strlen(t)); sl=ToLower(s);} };
        strip("question:"); strip("answer:"); strip("q:"); strip("a:");
        // strip punctuation
        std::string t; t.reserve(s.size());
        for (char c : s) if (!ispunct((unsigned char)c)) t.push_back(c);
        return TrimCollapse(t);
    };
    std::string na = norm(a), nb = norm(b);
    if (na.empty() || nb.empty()) return false;
    if (na == nb) return true;
    // very simple containment heuristic
    if (na.size() > 12 && nb.find(na) != std::string::npos) return true;
    if (nb.size() > 12 && na.find(nb) != std::string::npos) return true;
    return false;
}

//=================endof newfor duplcate answer =================

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
            SetupGazeResources();
            SetupSceneNodes();
            UpdateWorldCropFrame();  // <-- show the red frame right away
        }
    }



    void OnStop() override {
        UNWRAP_MLRESULT(DestroyCamera());
        DestroyVoice(); // shutdown TTS/ASR bridge
        DestroyGazeResources();

    }

    void OnDestroy() override {
        for (auto &t : standby_helper_threads_) if (t.joinable()) t.join();
        standby_helper_threads_.clear();
        UNWRAP_MLRESULT(DestroyCamera());
    }

    void OnUpdate(float dt) override {

        ClearForMR();
        // Move the gaze button & handle trigger click first
        UpdateGazeButtonAndTrigger(dt);





        if (awaiting_auto_question_) {

            awaiting_timer_s_ += dt;
            since_last_asr_s_ += dt;

            std::string transcript;
            {
                std::lock_guard<std::mutex> lk(g_asrMutex);
                if (!g_lastASR.empty()) {
                    transcript.swap(g_lastASR); // take and clear
                }
            }

            if (!transcript.empty()) {
                // Show the latest transcript in the text box
                heard_any_asr_ = true;  // NEW
                std::strncpy(question_buf_, transcript.c_str(), sizeof(question_buf_) - 1);
                question_buf_[sizeof(question_buf_) - 1] = '\0';
                since_last_asr_s_ = 0.0f;

                // If we are *not* in "listen until silence" mode, send immediately
                if (!listen_until_silence_) {
                    if (!last_captured_image_.empty()) {
                        std::string img = last_captured_image_;
                        std::thread([this, img, transcript]() { InitializeONNX(img, transcript); }).detach();
                    }
                    awaiting_auto_question_ = false;
                    pending_voice_question_ = false;
                    StopListening();
                }
            }

            const bool timed_out        = (awaiting_timer_s_ >= auto_wait_timeout_s_);
            const bool silence_finished = (listen_until_silence_ && heard_any_asr_ && since_last_asr_s_ >= silence_grace_s_); // NEW


            if (timed_out || silence_finished) {
                std::string q = question_buf_;
                if (q.empty()) q = "What is in the image?";
                if (!last_captured_image_.empty()) {
                    std::string img = last_captured_image_;
                    std::thread([this, img, q]() { InitializeONNX(img, q); }).detach();
                }
                awaiting_auto_question_ = false;
                pending_voice_question_ = false;
                StopListening();
            }
        }










        // Then draw your regular ImGui window
        UpdateGui();
    }


    // Voice bridge API (C++ → Java)
    void InitVoice();
    void DestroyVoice();
    void StartListening();
    void StopListening();
    void Speak(const std::string& text);

private:
    bool heard_any_asr_ = false; // reset per capture


    // in class CameraMixedRealityApp (private:)
    struct NormCrop {
        ImVec2 ctr{0.5f, 0.5f};
        ImVec2 size{0.35f, 0.35f};
        bool   locked = false;
    } crop_at_shutter_;



    // class member:
    float latest_image_aspect_ = 1.0f;
    // Add to class (private):
    void SyncCropSizeFromHUD(bool force=false) {
        static glm::vec2 prev = {-1.f, -1.f};
        const float base_w = std::max(aim_pad_base_size_m_.x, 1e-6f);
        const float base_h = std::max(aim_pad_base_size_m_.y, 1e-6f);
        auto clamp01 = [](float v){ return std::min(0.98f, std::max(0.05f, v)); };

        if (force || fabsf(aim_pad_size_m_.x - prev.x) > 1e-4f) {
            crop_norm_size_.x = clamp01(0.35f * (aim_pad_size_m_.x / base_w));
            prev.x = aim_pad_size_m_.x;
        }
        if (force || fabsf(aim_pad_size_m_.y - prev.y) > 1e-4f) {
            crop_norm_size_.y = clamp01(0.35f * (aim_pad_size_m_.y / base_h));
            prev.y = aim_pad_size_m_.y;
        }

        // keep center inside after any size change
        ImVec2 halfN(0.5f * crop_norm_size_.x, 0.5f * crop_norm_size_.y);
        crop_norm_center_.x = std::min(1.f - halfN.x, std::max(halfN.x, crop_norm_center_.x));
        crop_norm_center_.y = std::min(1.f - halfN.y, std::max(halfN.y, crop_norm_center_.y));
        UpdateWorldCropFrame();
    }

    // ----- Simple UI / visuals -----
    bool simple_ui_            = true;   // show a minimal GUI
    bool show_capture_button_  = false;  // HIDE the floating gaze button quad
    bool show_gaze_ray_        = false;  // HIDE the thin "laser" strip
    bool trigger_to_capture_   = true;   // click with controller trigger anywhere (no button)

// ----- World-space crop frame (red rectangle on the head-locked pad) -----
    std::shared_ptr<ml::app_framework::Node> crop_frame_node_;
    std::array<std::shared_ptr<ml::app_framework::Node>, 4> crop_edges_; // L,T,R,B
    std::shared_ptr<ml::app_framework::FlatMaterial> crop_edge_mat_;

    void UpdateWorldCropFrame(); // updates the 4 edges from crop_norm_center_/size_





    // --- UI sizing / second window (HUD) ---
    float ui_scale_ = 1.15f;                // global UI scale (fonts + paddings)
    bool  show_mini_hud_ = true;            // enable the small gaze HUD
    bool  mini_hud_follow_gaze_ = true;     // make HUD follow gaze on the head-locked pad
    ImVec2 mini_hud_offset_px_ = ImVec2(12, -12); // nudge HUD relative to gaze point



    bool   crop_with_gaze_    = true;              // NEW: use gaze to control crop rect
    ImVec2 last_pad_uv01_     = ImVec2(0.5f,0.5f); // NEW: last gaze on pad mapped to [0..1]


    // --- Gaze reticle (world, not GUI) ---
    std::shared_ptr<ml::app_framework::Node> aim_pad_node_;        // invisible head-locked plane
    std::shared_ptr<ml::app_framework::FlatMaterial> aim_pad_mat_;

    std::shared_ptr<ml::app_framework::Node> aim_reticle_node_;    // small red dot
    std::shared_ptr<ml::app_framework::FlatMaterial> aim_reticle_mat_;

    bool  use_gaze_reticle_   = true;   // toggle in UI if you want
    float aim_pad_distance_m_ = 0.95f;  // distance from head
    glm::vec2 aim_pad_size_m_ = {0.64f, 0.64f}; // W x H (meters)

    const glm::vec2 aim_pad_base_size_m_ = {0.64f, 0.64f}; // NEW: reference for mapping HUD→crop


    bool       gaze_on_pad_   = false;
    glm::vec2  cur_pad_local_ = {0.0f, 0.0f};   // local coords of current gaze hit (meters in pad space)
    glm::vec2  last_pad_uv_   = {0.0f, 0.0f};   // normalized [-1..+1] saved at click for cropping
    bool       have_last_pad_uv_ = false;


    //....................................
#ifdef ML_LUMIN
    GLuint last_photo_tex_ = 0;
#endif
    int     last_photo_w_ = 0;
    int     last_photo_h_ = 0;
    std::string last_photo_loaded_path_;   // to know when to reload the texture

// Crop GUI state (normalized to the image shown in the GUI)
    bool   use_crop_gui_     = true;             // toggle to enable the movable red rectangle
    ImVec2 crop_norm_center_ = ImVec2(0.5f, 0.5f); // 0..1 in image space
    ImVec2 crop_norm_size_   = ImVec2(0.35f, 0.35f); // width/height in 0..1 (not forced square)
    bool   crop_dragging_     = false;
    int    crop_drag_corner_  = -1;              // -1 move, 0..3 resize handles (TL, TR, BR, BL)
    ImVec2 crop_drag_start_mouse_;
    ImVec2 crop_drag_start_center_;
    ImVec2 crop_drag_start_size_;
    //....................................................................................
    // --- Visual "laser" for the gaze ray ---
    std::shared_ptr<ml::app_framework::Node> gaze_ray_node_;
    std::shared_ptr<ml::app_framework::FlatMaterial> gaze_ray_mat_;

// --- Fixation gate (makes clicking feel human/sensible) ---
    float fixate_ms_needed_ = 150.f;   // how long gaze must be steady
    float fixate_angle_deg_ = 1.5f;    // how steady (degrees)
    float fixate_timer_ms_  = 0.f;
    glm::vec3 fixate_ref_dir_{0,0,-1};





    bool listening_ = false;
    // Stores where on the button (in local quad coords) the last click occurred
    bool   have_last_click_local_ = false;
    glm::vec2 last_click_local_{0.0f, 0.0f}; // range ≈ [-half.x..+half.x], [-half.y..+half.y]
    //........................
    // --- Auto-capture behavior ---
    // --- Auto-capture behavior ---
    bool  wait_after_auto_capture_ = true;   // capture → listen for question
    float auto_wait_timeout_s_     = 5.0f;   // max overall wait before fallback
    bool  awaiting_auto_question_  = false;  // currently listening/waiting
    float awaiting_timer_s_        = 0.0f;   // counts while waiting

// NEW: listen-until-silence behavior
    bool  listen_until_silence_    = true;   // if true: keep listening while the user speaks
    float silence_grace_s_         = 1.2f;   // consider it "done" after this much silence
    float since_last_asr_s_        = 0.0f;   // time since the last ASR text arrived


    // Material so we can highlight on hover
    std::shared_ptr<ml::app_framework::FlatMaterial> capture_button_mat_;

// Dwell-to-click (eyes-only) hover state
    bool   gaze_over_button_ = false;

    // ---------- Scene nodes for the gaze button ----------
    std::shared_ptr<ml::app_framework::Node> capture_button_node_;

// Distance of the floating button from the eye midpoint (meters)
    float gaze_button_distance_m_ = 0.6f;

// ---------- Eye/Head tracking & input ----------
    MLHandle eye_tracker_  = ML_INVALID_HANDLE;
    MLHandle head_tracker_ = ML_INVALID_HANDLE;
    MLEyeTrackingStaticData eye_static_data_ {};
    MLHeadTrackingStaticData head_static_data_ {};
    MLHandle input_handle_  = ML_INVALID_HANDLE;

// Last resolved gaze ray (world space)
    glm::vec3 last_gaze_origin_{0,0,0};
    glm::vec3 last_gaze_dir_{0,0,-1};

// Debounce state for trigger-as-click
    struct { bool was_down = false; float cooldown_ms = 0.f; } trigger_click_;
    void SetupGazeResources() {
        if (eye_tracker_ == ML_INVALID_HANDLE) {
            MLResult r = MLEyeTrackingCreate(&eye_tracker_);
            if (r == MLResult_Ok) {
                UNWRAP_MLRESULT_FATAL(MLEyeTrackingGetStaticData(eye_tracker_, &eye_static_data_));
            } else {
                ALOGW("Eye tracking unavailable (r=%d). Disabling gaze capture.", (int)r);
                use_gaze_capture_ = false;
            }
        }

        if (head_tracker_ == ML_INVALID_HANDLE) {
            UNWRAP_MLRESULT(MLHeadTrackingCreate(&head_tracker_));
            UNWRAP_MLRESULT_FATAL(MLHeadTrackingGetStaticData(head_tracker_, &head_static_data_));
        }
        if (input_handle_ == ML_INVALID_HANDLE) {
            UNWRAP_MLRESULT(MLInputCreate(&input_handle_));
        }
    }

    void DestroyGazeResources() {
        if (eye_tracker_ != ML_INVALID_HANDLE) {
            UNWRAP_MLRESULT(MLEyeTrackingDestroy(eye_tracker_));
            eye_tracker_ = ML_INVALID_HANDLE;
        }
        if (head_tracker_ != ML_INVALID_HANDLE) {
            UNWRAP_MLRESULT(MLHeadTrackingDestroy(head_tracker_));
            head_tracker_ = ML_INVALID_HANDLE;
        }
        if (input_handle_ != ML_INVALID_HANDLE) {
            UNWRAP_MLRESULT(MLInputDestroy(input_handle_));
            input_handle_ = ML_INVALID_HANDLE;
        }
    }

    void SetupSceneNodes();
//..........
// Make a quaternion that looks along 'dir' with world up (0,1,0)
    static glm::quat LookRotation(const glm::vec3& dir) {
        glm::vec3 f = glm::normalize(dir);
        glm::vec3 up(0,1,0);
        // avoid degeneracy
        if (fabsf(glm::dot(f, up)) > 0.99f) up = glm::vec3(0,0,1);
        glm::vec3 r = glm::normalize(glm::cross(up, f));
        glm::vec3 u = glm::normalize(glm::cross(f, r));
        glm::mat3 m(r, u, f); // columns
        return glm::quat_cast(m);
    }

// Ray vs. oriented quad (world-space) for trigger click
    bool RayHitsQuad(const glm::vec3& ray_o,
                     const glm::vec3& ray_d_norm,
                     const glm::vec3& quad_pos,
                     const glm::quat& quad_rot,
                     const glm::vec2& half)
    {
        const glm::vec3 n = quad_rot * glm::vec3(0,0,1);
        const float denom = glm::dot(n, ray_d_norm);
        if (fabsf(denom) < 1e-4f) return false;

        const float t = glm::dot(quad_pos - ray_o, n) / denom;
        if (t <= 0.0f) return false;

        const glm::vec3 hit = ray_o + t * ray_d_norm;
        const glm::vec3 local = glm::inverse(quad_rot) * (hit - quad_pos);
        return (fabsf(local.x) <= half.x && fabsf(local.y) <= half.y);
    }

    //...................
    void UpdateGazeButtonAndTrigger(float dt_seconds);


    //..........................................................
    // --- Gaze capture controls ---
    bool  use_gaze_capture_  = true;     // toggle from GUI
    int   gaze_crop_px_      = 384;      // square crop size (px)
    bool  auto_dwell_snap_   = false;    // optional: auto-snap when gaze steady
    float dwell_ms_needed_   = 500.f;
    float dwell_timer_ms_    = 0.f;

    // Last known mid-gaze in view space (debug/reticle). Replace with real eye tracking later.
    glm::vec3 last_mid_gaze_dir_{0,0,-1};

    // Returns true if we have a valid mid-gaze direction in HEAD space.
    bool GetMidGazeRay(glm::vec3& out_origin, glm::vec3& out_dir_norm);

    //.....................................................................
    std::string last_question_text_;
    std::atomic<uint64_t> answer_seq_{0};


    //................................
    // --- TTS / answer playback control ---
    bool speak_auto_ = true;                 // OFF by default (text-only until you opt in)
    std::string last_answer_text_;            // last decoded answer text
//    std::atomic<uint64_t> answer_seq_{0};     // increments each time we compute a new answer
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

// Compute local coords at gaze-ray / quad hit, returns true if it hits and fills local
// Return true and fill out_local with the button's local (x,y) hit if the gaze ray hits the quad
static bool HitQuadLocal(
        const glm::vec3& ray_o, const glm::vec3& ray_d_norm,
        const glm::vec3& quad_pos, const glm::quat& quad_rot,
        const glm::vec2& half, glm::vec2& out_local)
{
    // IMPORTANT: the app_framework Quad faces -Z in local space
    const glm::vec3 n = quad_rot * glm::vec3(0,0,-1);
    const float denom = glm::dot(n, ray_d_norm);
    if (fabsf(denom) < 1e-4f) return false;

    const float t = glm::dot(quad_pos - ray_o, n) / denom;
    if (t <= 0.0f) return false;

    const glm::vec3 hit = ray_o + t * ray_d_norm;
    const glm::vec3 local3 = glm::inverse(quad_rot) * (hit - quad_pos);

    if (fabsf(local3.x) <= half.x && fabsf(local3.y) <= half.y) {
        out_local = glm::vec2(local3.x, local3.y);
        return true;
    }
    return false;
}
//----------------------------------------------endof  class--------------------------------------------------

//...................... start of button trigger................................
void CameraMixedRealityApp::UpdateGazeButtonAndTrigger(float dt_seconds) {
    using ml::app_framework::RenderableComponent;

    // --- Resolve current gaze ray (eyes preferred, head fallback) ---
    glm::vec3 o{}, d{};
    const bool have_gaze = GetMidGazeRay(o, d);

    // --- Place the floating capture button along the ray (but visibility is flag-controlled) ---
    auto rc_btn = capture_button_node_
                  ? capture_button_node_->GetComponent<RenderableComponent>()
                  : nullptr;

    if (capture_button_node_ && have_gaze) {
        const glm::vec3 pos = o + d * gaze_button_distance_m_;
        const glm::quat rot = LookRotation(-d); // face the user
        capture_button_node_->SetWorldTranslation(pos);
        capture_button_node_->SetWorldRotation(rot);
    }
    if (rc_btn) {
        rc_btn->SetVisible(show_capture_button_ && use_gaze_capture_);
    }

    // --- Head-locked aim pad + red reticle (pad stays INVISIBLE; only edges show) ---
    MLSnapshot* snap = nullptr;
    MLTransform head{};
    bool have_head = false;
    if (MLPerceptionGetSnapshot(&snap) == MLResult_Ok && snap) {
        if (head_tracker_ != ML_INVALID_HANDLE) {
            have_head = (MLSnapshotGetTransform(snap, &head_static_data_.coord_frame_head, &head) == MLResult_Ok);
        }
    }

    if (have_head && aim_pad_node_) {
        glm::vec3 hpos = ml::app_framework::to_glm(head.position);
        glm::quat hrot = ml::app_framework::to_glm(head.rotation);
        glm::vec3 hfwd = glm::normalize(hrot * glm::vec3(0,0,-1));

        // Keep pad centered in front of you
        aim_pad_node_->SetLocalScale(glm::vec3(aim_pad_size_m_.x, aim_pad_size_m_.y, 0.001f));

// Map HUD size (meters) → crop size (normalized 0..1), preserving aspect
        { SyncCropSizeFromHUD();

        }




        glm::vec3 pad_pos = hpos + hfwd * aim_pad_distance_m_;
        glm::quat pad_rot = LookRotation(-hfwd);
        aim_pad_node_->SetWorldTranslation(pad_pos);
        aim_pad_node_->SetWorldRotation(pad_rot);

        // IMPORTANT: keep aim pad QUAD invisible (no fill)
        if (auto pad_rc = aim_pad_node_->GetComponent<RenderableComponent>()) {
            pad_rc->SetVisible(false);                 // <— don’t render the pad
        }
        if (aim_pad_mat_) {
            aim_pad_mat_->SetColor(glm::vec4(1,1,1,0.0f)); // <— fully transparent, no tint
        }

        // Ray vs. pad to place the red dot (reticle)
        bool show_reticle = false;
        if (have_gaze && aim_reticle_node_) {
            glm::vec2 half = glm::vec2(fabsf(aim_pad_node_->GetLocalScale().x) * 0.5f,
                                       fabsf(aim_pad_node_->GetLocalScale().y) * 0.5f);
            glm::vec2 hit_local{};
            gaze_on_pad_ = HitQuadLocal(o, glm::normalize(d), pad_pos, pad_rot, half, hit_local);

            if (gaze_on_pad_) {
                aim_reticle_node_->SetLocalTranslation(glm::vec3(hit_local.x, hit_local.y, 0.001f));
                cur_pad_local_ = hit_local;
                show_reticle   = true;
            } else {
                show_reticle = false;
            }
        } else {
            gaze_on_pad_ = false;
        }

        if (auto rrc = aim_reticle_node_
                       ? aim_reticle_node_->GetComponent<RenderableComponent>()
                       : nullptr) {
            rrc->SetVisible(use_gaze_reticle_ && show_reticle);
        }
    } else {
        gaze_on_pad_ = false;
        if (auto rrc = aim_reticle_node_
                       ? aim_reticle_node_->GetComponent<RenderableComponent>()
                       : nullptr) {
            rrc->SetVisible(false);
        }
    }
    if (snap) MLPerceptionReleaseSnapshot(snap);

    // --- Drive crop rect from gaze on the (invisible) pad; update red border edges ---
    if (use_crop_gui_ && crop_with_gaze_ && gaze_on_pad_ && aim_pad_node_) {
        glm::vec2 halfPad = glm::vec2(fabsf(aim_pad_node_->GetLocalScale().x) * 0.5f,
                                      fabsf(aim_pad_node_->GetLocalScale().y) * 0.5f);

        const float hx = halfPad.x;
        const float hy = halfPad.y;

        float aspect = latest_image_aspect_;
        if (!(aspect > 0.f)) {
            aspect = (capture_width_ > 0 && capture_height_ > 0)
                     ? (float)capture_width_ / (float)capture_height_
                     : 1.0f;
        }

        float sx=0.f, sy=0.f;
        ImageRectOnPad(hx, hy, aspect, sx, sy);

// map gaze hit inside the *image rect* (±sx, ±sy) → [0..1]
        float u = cur_pad_local_.x / std::max(1e-6f, sx);
        float v = cur_pad_local_.y / std::max(1e-6f, sy);
        u = std::min(1.f, std::max(-1.f, u));
        v = std::min(1.f, std::max(-1.f, v));

        float u01 = 0.5f + 0.5f * u;
        float v01 = 0.5f - 0.5f * v;
        last_pad_uv01_ = ImVec2(u01, v01);

        ImVec2 halfN(0.5f * crop_norm_size_.x, 0.5f * crop_norm_size_.y);
        crop_norm_center_.x = std::min(1.f - halfN.x, std::max(halfN.x, u01));
        crop_norm_center_.y = std::min(1.f - halfN.y, std::max(halfN.y, v01));

        // This moves the 4 red edge quads (border only)
        UpdateWorldCropFrame();
    }

    // --- Optional visible gaze ray (thin quad) ---
    if (gaze_ray_node_) {
        if (auto rcRay = gaze_ray_node_->GetComponent<RenderableComponent>()) {
            rcRay->SetVisible(show_gaze_ray_ && have_gaze);
            if (show_gaze_ray_ && have_gaze) {
                const float len = std::max(0.05f, glm::length(
                        (capture_button_node_ ? capture_button_node_->GetWorldTranslation() : (o + d * 0.6f)) - o));
                gaze_ray_node_->SetWorldRotation(LookRotationY(d));
                gaze_ray_node_->SetWorldTranslation(o + d * (len * 0.5f));
                gaze_ray_node_->SetLocalScale(glm::vec3(0.003f, len, 0.001f));
                if (gaze_ray_mat_) {
                    gaze_ray_mat_->SetColor((fixate_timer_ms_ >= fixate_ms_needed_)
                                            ? glm::vec4(1,1,1,0.65f)
                                            : glm::vec4(1,1,1,0.25f));
                }
            }
        }
    }

    // --- Fixation gate ---
    bool fixation_ok = false;
    if (have_gaze) {
        if (fixate_timer_ms_ <= 0.f) fixate_ref_dir_ = d;
        const float ang = glm::degrees(acosf(glm::clamp(glm::dot(glm::normalize(d),
                                                                 glm::normalize(fixate_ref_dir_)),
                                                        -1.0f, 1.0f)));
        if (ang < fixate_angle_deg_) {
            fixate_timer_ms_ += dt_seconds * 1000.f;
        } else {
            fixate_timer_ms_ = 0.f;
            fixate_ref_dir_  = d;
        }
        fixation_ok = (fixate_timer_ms_ >= fixate_ms_needed_);
    } else {
        fixate_timer_ms_ = 0.f;
        fixation_ok = false;
    }

    // --- Dwell-to-click over the floating button (optional) ---
    bool gaze_hits_button = false;
    glm::vec2 hit_local_btn(0.0f);
    if (show_capture_button_ && use_gaze_capture_ && have_gaze && capture_button_node_) {
        const glm::vec3 bpos = capture_button_node_->GetWorldTranslation();
        const glm::quat brot = capture_button_node_->GetWorldRotation();
        const glm::vec3 s    = capture_button_node_->GetLocalScale();
        const glm::vec2 half = glm::vec2(fabsf(s.x) * 0.5f, fabsf(s.y) * 0.5f);

        const bool hits_quad = HitQuadLocal(o, glm::normalize(d), bpos, brot, half, hit_local_btn);
        gaze_hits_button = fixation_ok && hits_quad;

        if (capture_button_mat_) {
            const glm::vec4 col = gaze_hits_button
                                  ? glm::vec4(0.20f, 1.00f, 0.20f, 0.95f)
                                  : (fixation_ok ? glm::vec4(1.00f, 0.85f, 0.20f, 0.95f)
                                                 : glm::vec4(1.00f, 1.00f, 1.00f, 0.90f));
            capture_button_mat_->SetColor(col);
        }
    } else if (capture_button_mat_) {
        capture_button_mat_->SetColor(glm::vec4(1.00f, 1.00f, 1.00f, 0.90f));
    }
    gaze_over_button_ = gaze_hits_button;

    if (auto_dwell_snap_ && gaze_hits_button) {
        dwell_timer_ms_ += dt_seconds * 1000.f;
        if (dwell_timer_ms_ >= dwell_ms_needed_) {
            if (wait_after_auto_capture_) {
                send_to_vlm_after_capture_ = false;
                pending_voice_question_    = true;
                awaiting_auto_question_    = true;
                awaiting_timer_s_          = 0.0f;
            } else {
                send_to_vlm_after_capture_ = true;
                pending_voice_question_    = false;
                typed_question_            = question_buf_;
            }

            have_last_click_local_ = true;
            last_click_local_      = hit_local_btn;

            // Snapshot gaze → crop-at-gaze BEFORE capture
            if (gaze_on_pad_) {
                have_last_pad_uv_ = true;
                last_pad_uv_ = glm::vec2(
                        (last_pad_uv01_.x - 0.5f) * 2.0f,
                        (0.5f - last_pad_uv01_.y) * 2.0f
                );
            } else {
                have_last_pad_uv_ = false;
            }


            // right before UNWRAP_MLRESULT(CaptureImage());
            crop_at_shutter_.ctr  = crop_norm_center_;
            crop_at_shutter_.size = crop_norm_size_;
            crop_at_shutter_.locked = true;

            UNWRAP_MLRESULT(CaptureImage());

            dwell_timer_ms_ = 0.f;
        }
    } else {
        dwell_timer_ms_ = 0.f;
    }

    // --- Controller: resize crop when looking at the pad (bumper/trigger) + trigger capture ---
    if (input_handle_ != ML_INVALID_HANDLE) {
        MLInputControllerStateEx st[MLInput_MaxControllers];
        MLInputControllerStateExInit(st);
        if (MLInputGetControllerStateEx(input_handle_, st) == MLResult_Ok) {
            int use_i = -1;
            for (int i = 0; i < MLInput_MaxControllers; ++i)
                if (st[i].is_connected) { use_i = i; break; }

            const bool have = (use_i >= 0);
            const float trig   = have ? st[use_i].trigger_normalized : 0.0f; // 0..1
            const bool  bumper = have ? st[use_i].button_state[MLInputControllerButton_Bumper] : false;

            // Resize HUD while gazing at the pad (crop follows via HUD→crop mapping)
            if (use_crop_gui_ && crop_with_gaze_ && gaze_on_pad_) {

                // --- Controller: resize HUD only if bumper is down; trigger-only is capture ---
                const bool trigDown = (trig >= 0.85f);
                const bool bumDown  = bumper;

// Resize HUD while gazing at the pad (crop follows via HUD→crop mapping)
                if (use_crop_gui_ && crop_with_gaze_ && gaze_on_pad_ && bumDown) {
                    const float u = last_pad_uv01_.x;
                    const float v = last_pad_uv01_.y;
                    const glm::vec2 minHud(0.10f, 0.10f), maxHud(1.20f, 1.20f);
                    const float k = 0.35f;
                    const float base_w = std::max(aim_pad_base_size_m_.x, 1e-6f);
                    const float base_h = std::max(aim_pad_base_size_m_.y, 1e-6f);
                    auto clamp_norm = [](float x){ return std::min(0.98f, std::max(0.05f, x)); };

                    if (bumDown && !trigDown) {
                        // Width only (bumper)
                        float desired_norm_w = clamp_norm(2.0f * fabsf(u - crop_norm_center_.x));
                        float new_w_m = (desired_norm_w / k) * base_w;
                        aim_pad_size_m_.x = std::min(maxHud.x, std::max(minHud.x, new_w_m));
                    } else if (bumDown && trigDown) {
                        // Width + height (both)
                        float desired_norm_w = clamp_norm(2.0f * fabsf(u - crop_norm_center_.x));
                        float desired_norm_h = clamp_norm(2.0f * fabsf(v - crop_norm_center_.y));
                        float new_w_m = (desired_norm_w / k) * base_w;
                        float new_h_m = (desired_norm_h / k) * base_h;
                        aim_pad_size_m_.x = std::min(maxHud.x, std::max(minHud.x, new_w_m));
                        aim_pad_size_m_.y = std::min(maxHud.y, std::max(minHud.y, new_h_m));
                    }

                    if (aim_pad_node_) {
                        aim_pad_node_->SetLocalScale(glm::vec3(aim_pad_size_m_.x, aim_pad_size_m_.y, 0.001f));
                    }
                    SyncCropSizeFromHUD(true);
                }

            }


            // Trigger to capture anywhere
            const bool down = (trig >= 0.85f) || bumper;
            trigger_click_.cooldown_ms = std::max(0.f, trigger_click_.cooldown_ms - dt_seconds * 1000.f);

            if (trigger_to_capture_ && down && !trigger_click_.was_down && trigger_click_.cooldown_ms <= 0.f) {
                if (wait_after_auto_capture_) {
                    send_to_vlm_after_capture_ = false;
                    pending_voice_question_    = true;
                    awaiting_auto_question_    = true;
                    awaiting_timer_s_          = 0.0f;
                } else {
                    send_to_vlm_after_capture_ = true;
                    pending_voice_question_    = false;
                    typed_question_            = question_buf_;
                }

                // Snapshot gaze BEFORE capture for crop-at-gaze
                if (gaze_on_pad_) {
                    have_last_pad_uv_ = true;
                    last_pad_uv_ = glm::vec2(
                            (last_pad_uv01_.x - 0.5f) * 2.0f,
                            (0.5f - last_pad_uv01_.y) * 2.0f
                    );
                } else {
                    have_last_pad_uv_ = false;
                }


                // right before UNWRAP_MLRESULT(CaptureImage());
                crop_at_shutter_.ctr  = crop_norm_center_;
                crop_at_shutter_.size = crop_norm_size_;
                crop_at_shutter_.locked = true;

                UNWRAP_MLRESULT(CaptureImage());

                trigger_click_.cooldown_ms = 250.f;
            }
            trigger_click_.was_down = down;
        }
    }
}


//...................... end of button trigger................................

//...............................................................
void CameraMixedRealityApp::SetupSceneNodes() {
    using namespace ml::app_framework;

    // --- Capture button (create once) ---
    if (!capture_button_node_) {
        capture_button_node_ = std::make_shared<Node>();
        GetRoot()->AddChild(capture_button_node_);

        static std::shared_ptr<QuadMesh> button_mesh = std::make_shared<QuadMesh>(); // unit quad
        capture_button_mat_ = std::make_shared<FlatMaterial>(glm::vec4(1.0f, 0.0f, 0.0f, 0.9f)); // red

        auto renderable = std::make_shared<RenderableComponent>(button_mesh, capture_button_mat_);
        capture_button_node_->AddComponent(renderable);

        // ~12cm x 5cm quad (depth tiny)
        capture_button_node_->SetLocalScale(glm::vec3(0.12f, 0.05f, 0.005f));
        renderable->SetVisible(false);
    }

    // --- Gaze "laser" (create once) ---
    if (!gaze_ray_node_) {
        auto ray_mesh = std::make_shared<QuadMesh>();
        gaze_ray_mat_  = std::make_shared<FlatMaterial>(glm::vec4(1.0f, 1.0f, 1.0f, 0.55f));
        gaze_ray_node_ = std::make_shared<Node>();
        auto ray_rc    = std::make_shared<RenderableComponent>(ray_mesh, gaze_ray_mat_);
        gaze_ray_node_->AddComponent(ray_rc);
        GetRoot()->AddChild(gaze_ray_node_);

        // Tall axis = local +Y. We'll rescale/position/rotate every frame.
        gaze_ray_node_->SetLocalScale(glm::vec3(0.003f, 0.6f, 0.001f));
        ray_rc->SetVisible(false);
    }
    if (!aim_pad_node_) {
        aim_pad_node_ = std::make_shared<Node>();
        GetRoot()->AddChild(aim_pad_node_);

        auto pad_mesh = std::make_shared<QuadMesh>();                 // unit quad
        aim_pad_mat_  = std::make_shared<FlatMaterial>(glm::vec4(1,1,1,0.03f)); // nearly invisible
        auto pad_rc   = std::make_shared<RenderableComponent>(pad_mesh, aim_pad_mat_);
        aim_pad_node_->AddComponent(pad_rc);



        aim_pad_mat_->SetColor(glm::vec4(1,1,1,0.0f));
        pad_rc->SetVisible(false);



        // scale to desired size (meters)
        aim_pad_node_->SetLocalScale(glm::vec3(aim_pad_size_m_.x, aim_pad_size_m_.y, 0.001f));

        // child: red reticle (small dot)
        aim_reticle_node_ = std::make_shared<Node>();
        auto ret_mesh     = std::make_shared<QuadMesh>();
        aim_reticle_mat_  = std::make_shared<FlatMaterial>(glm::vec4(1.0f, 0.15f, 0.15f, 0.95f)); // red
        auto ret_rc       = std::make_shared<RenderableComponent>(ret_mesh, aim_reticle_mat_);
        aim_reticle_node_->AddComponent(ret_rc);
        aim_pad_node_->AddChild(aim_reticle_node_);

        // ~1.5 cm dot
        aim_reticle_node_->SetLocalScale(glm::vec3(0.015f, 0.015f, 0.001f));
        ret_rc->SetVisible(false); // hidden until gaze hits pad
    }

// --- World crop frame (child of the head-locked aim pad) ---
    if (!crop_frame_node_) {
        using namespace ml::app_framework;
        crop_frame_node_ = std::make_shared<Node>();
        if (!aim_pad_node_) {
            // create aim pad if not yet created (safety)
            aim_pad_node_ = std::make_shared<Node>();
            GetRoot()->AddChild(aim_pad_node_);
            auto pad_mesh = std::make_shared<QuadMesh>();
            aim_pad_mat_  = std::make_shared<FlatMaterial>(glm::vec4(1,0,0,1.0f)); // fully invisible pad
            auto pad_rc   = std::make_shared<RenderableComponent>(pad_mesh, aim_pad_mat_);
            aim_pad_node_->AddComponent(pad_rc);
            aim_pad_node_->SetLocalScale(glm::vec3(aim_pad_size_m_.x, aim_pad_size_m_.y, 0.001f));
        }
        aim_pad_node_->AddChild(crop_frame_node_);

        crop_edge_mat_ = std::make_shared<FlatMaterial>(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
        for (int i = 0; i < 4; ++i) {
            auto edge = std::make_shared<Node>();
            auto mesh = std::make_shared<QuadMesh>();
            auto rc   = std::make_shared<RenderableComponent>(mesh, crop_edge_mat_);
            edge->AddComponent(rc);
            crop_frame_node_->AddChild(edge);
            crop_edges_[i] = edge;
            edge->SetLocalScale(glm::vec3(0.001f, 0.001f, 0.001f));
            rc->SetVisible(true);
        }
    }

// Ensure the world crop frame is initialized to current normalized rect
    UpdateWorldCropFrame();


}
//.........................................................crop start.......................
void CameraMixedRealityApp::UpdateWorldCropFrame() {
    using namespace ml::app_framework;
    if (!aim_pad_node_ || !crop_frame_node_) return;
    for (int i=0;i<4;++i) if (!crop_edges_[i]) return;

    const float hx = fabsf(aim_pad_node_->GetLocalScale().x) * 0.5f;
    const float hy = fabsf(aim_pad_node_->GetLocalScale().y) * 0.5f;

    float aspect = latest_image_aspect_;
    if (!(aspect > 0.f)) {
        aspect = (capture_width_ > 0 && capture_height_ > 0)
                 ? (float)capture_width_ / (float)capture_height_
                 : 1.0f;
    }

    float sx=0.f, sy=0.f;
    ImageRectOnPad(hx, hy, aspect, sx, sy);   // <- image half-sizes inside the pad

    const float cx = (crop_norm_center_.x - 0.5f) * 2.0f * sx;
    const float cy = (0.5f - crop_norm_center_.y) * 2.0f * sy;

    const float w  = std::max(0.0f, crop_norm_size_.x * 2.0f * sx);
    const float h  = std::max(0.0f, crop_norm_size_.y * 2.0f * sy);

    const float t  = 0.004f;

    auto set_edge = [](std::shared_ptr<Node>& n, float x, float y, float sx, float sy) {
        n->SetLocalTranslation(glm::vec3(x, y, 0.001f));
        n->SetLocalScale(glm::vec3(std::max(0.001f, sx), std::max(0.001f, sy), 0.001f));
    };
    set_edge(crop_edges_[1], cx,          cy + h*0.5f, std::max(w, t), t); // top
    set_edge(crop_edges_[3], cx,          cy - h*0.5f, std::max(w, t), t); // bottom
    set_edge(crop_edges_[0], cx - w*0.5f, cy,          t,              std::max(h, t)); // left
    set_edge(crop_edges_[2], cx + w*0.5f, cy,          t,              std::max(h, t)); // right
}

//.........................................................crop end....................

//........................................................................................
bool CameraMixedRealityApp::GetMidGazeRay(glm::vec3& out_origin, glm::vec3& out_dir_norm) {
    // If neither tracker exists there's nothing we can do
    if (eye_tracker_ == ML_INVALID_HANDLE && head_tracker_ == ML_INVALID_HANDLE)
        return false;

    MLSnapshot* snapshot = nullptr;
    if (MLPerceptionGetSnapshot(&snapshot) != MLResult_Ok || !snapshot)
        return false;

    // Try to read eyes; also try head so we have a fallback
    MLTransform verg{}, left{}, right{}, head{};
    bool eye_ok  = false;
    bool head_ok = false;

    if (eye_tracker_ != ML_INVALID_HANDLE) {
        MLResult r1 = MLSnapshotGetTransform(snapshot, &eye_static_data_.vergence,     &verg);
        MLResult r2 = MLSnapshotGetTransform(snapshot, &eye_static_data_.left_center,  &left);
        MLResult r3 = MLSnapshotGetTransform(snapshot, &eye_static_data_.right_center, &right);
        eye_ok = (r1 == MLResult_Ok) && (r2 == MLResult_Ok) && (r3 == MLResult_Ok);
    }

    if (head_tracker_ != ML_INVALID_HANDLE) {
        head_ok = (MLSnapshotGetTransform(snapshot, &head_static_data_.coord_frame_head, &head) == MLResult_Ok);
    }

    // Prefer eye gaze if we have it
    if (eye_ok) {
        glm::vec3 l_o = ml::app_framework::to_glm(left.position);
        glm::vec3 r_o = ml::app_framework::to_glm(right.position);
        glm::vec3 mid = 0.5f * (l_o + r_o);

        glm::quat v_q = ml::app_framework::to_glm(verg.rotation);
        glm::vec3 dir = glm::normalize(v_q * glm::vec3(0,0,-1));

        out_origin        = mid;
        out_dir_norm      = dir;
        last_mid_gaze_dir_= dir;
        last_gaze_origin_ = mid;
        last_gaze_dir_    = dir;

        MLPerceptionReleaseSnapshot(snapshot);
        return true;
    }

    // Fallback: head pose forward
    if (head_ok) {
        glm::vec3 origin = ml::app_framework::to_glm(head.position);
        glm::quat h_q    = ml::app_framework::to_glm(head.rotation);
        glm::vec3 dir    = glm::normalize(h_q * glm::vec3(0,0,-1));

        out_origin        = origin;
        out_dir_norm      = dir;
        last_gaze_origin_ = origin;
        last_gaze_dir_    = dir;

        MLPerceptionReleaseSnapshot(snapshot);
        return true;
    }

    // Nothing usable
    MLPerceptionReleaseSnapshot(snapshot);
    return false;
}
//================================================================================
CameraMixedRealityApp::CameraMixedRealityApp(struct android_app* state)
        : Application(state, {
        "android.permission.CAMERA",
        "android.permission.RECORD_AUDIO",
        "com.magicleap.permission.EYE_TRACKING"
}, USE_GUI),

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
// ===== GUI =====start


void CameraMixedRealityApp::UpdateGui() {
    // ---------- Mic → textbox ingest (NO ImGui calls here) ----------
    std::string asr_err_local;
    {
        std::lock_guard<std::mutex> lk(g_asrMutex);

        if (!g_lastASR.empty()) {
            std::string transcript = std::move(g_lastASR);
            g_lastASR.clear();
            std::strncpy(question_buf_, transcript.c_str(), sizeof(question_buf_) - 1);
            question_buf_[sizeof(question_buf_) - 1] = '\0';

            // keep the "listen until silence" timers sane
            since_last_asr_s_ = 0.0f;
            heard_any_asr_    = true;
        }
        if (!g_lastASRError.empty()) {
            asr_err_local.swap(g_lastASRError); // render inside the window below
        }
    }

    // ---------- Window styling ----------
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0,0,0,0.0f));
    ImGui::PushStyleColor(ImGuiCol_ChildBg,  ImVec4(0,0,0,0.0f));

    auto& gui = GetGui();
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 screen = io.DisplaySize;

    // Global UI scale guard
    static float applied_scale = 1.0f;
    if (fabsf(applied_scale - ui_scale_) > 1e-3f) {
        float ratio = ui_scale_ / applied_scale;
        ImGui::GetStyle().ScaleAllSizes(ratio);
        io.FontGlobalScale = ui_scale_;
        applied_scale = ui_scale_;
    }

    // Window position/size
    ImVec2 mainSize(std::min(screen.x * 0.48f, 820.0f),
                    std::min(screen.y * 0.90f, 980.0f));
    ImGui::SetNextWindowPos(ImVec2(screen.x - 12.0f, screen.y * 0.06f),
                            ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowSize(mainSize, ImGuiCond_Always);

    gui.BeginUpdate();

    bool running = true;
    bool opened = gui.BeginDialog("MagicVLM - Simple", &running,
                                  ImGuiWindowFlags_NoCollapse |
                                  ImGuiWindowFlags_NoBackground |
                                  ImGuiWindowFlags_AlwaysVerticalScrollbar);

    if (opened) {
        // Mic/ASR error (rendered INSIDE the window)
        if (!asr_err_local.empty()) {
            ImGui::TextColored(ImVec4(1,0.4f,0.4f,1), "ASR error: %s", asr_err_local.c_str());
        }

        // ───────────────────────────────────────────────────────────
        // 1) CAPTURE
        // ───────────────────────────────────────────────────────────
        ImGui::Text("1) Capture");
        if (ImGui::Button("Capture")) {
            send_to_vlm_after_capture_ = false;
            pending_voice_question_    = false;

            // snapshot HUD rect used at shutter
            crop_at_shutter_.ctr    = crop_norm_center_;
            crop_at_shutter_.size   = crop_norm_size_;
            crop_at_shutter_.locked = true;

            UNWRAP_MLRESULT(CaptureImage());
        }
        ImGui::SameLine();
        if (ImGui::Button("Capture + Voice")) {
            send_to_vlm_after_capture_ = false;
            pending_voice_question_    = true;

            crop_at_shutter_.ctr    = crop_norm_center_;
            crop_at_shutter_.size   = crop_norm_size_;
            crop_at_shutter_.locked = true;

            UNWRAP_MLRESULT(CaptureImage());
        }
        ImGui::SameLine();
        if (ImGui::Button("Capture + Ask")) {
            send_to_vlm_after_capture_ = true;
            pending_voice_question_    = false;
            typed_question_            = question_buf_;

            crop_at_shutter_.ctr    = crop_norm_center_;
            crop_at_shutter_.size   = crop_norm_size_;
            crop_at_shutter_.locked = true;

            UNWRAP_MLRESULT(CaptureImage());
        }

        ImGui::Separator();

        ImGui::Checkbox("Wait after capture", &wait_after_auto_capture_);
        ImGui::SameLine();
        ImGui::Checkbox("Listen until silence", &listen_until_silence_);
        ImGui::SliderFloat("Silence timeout (s)", &silence_grace_s_, 0.5f, 5.0f, "%.1f");
        ImGui::SliderFloat("Max wait (s)", &auto_wait_timeout_s_, 2.0f, 20.0f, "%.1f");

        // ───────────────────────────────────────────────────────────
        // 2) QUESTION (TEXT + VOICE)
        // ───────────────────────────────────────────────────────────
        ImGui::Text("2) Ask a question");

        // Text entry
        ImGui::InputText("Question", question_buf_, IM_ARRAYSIZE(question_buf_));
        ImGui::SameLine();
        if (ImGui::Button("Ask (text)")) {
            if (!last_captured_image_.empty() && question_buf_[0] != '\0') {
                std::string img = last_captured_image_;
                std::string q   = std::string(question_buf_);
                std::thread([this, img, q]() { InitializeONNX(img, q); }).detach();
            } else {
                onnx_status_message_ = "Type a question and capture an image first.";
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Edit")) {
            OpenEditDialog(std::string(question_buf_));
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            question_buf_[0] = '\0';
        }

        // Voice controls (no inner ASR reads here — we did it at the top)
        {
            bool have_photo = !last_captured_image_.empty();
            if (!have_photo) ImGui::BeginDisabled();
            if (ImGui::Button("🎤 Ask by voice (listen)")) {
                StartListening();
            }
            ImGui::SameLine();
            if (ImGui::Button("⏹ Stop listening")) {
                StopListening();
            }
            if (!have_photo) ImGui::EndDisabled();
        }

        ImGui::Separator();

        // ───────────────────────────────────────────────────────────
        // 3) GAZE & CROP OPTIONS
        // ───────────────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Gaze & Crop", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Use eye gaze to move crop", &crop_with_gaze_);
            ImGui::SameLine();
            ImGui::Checkbox("Show gaze reticle", &use_gaze_reticle_);
            ImGui::SameLine();
            ImGui::Checkbox("Show gaze ray", &show_gaze_ray_);

            ImGui::Checkbox("Use world crop overlay", &use_crop_gui_);
            ImGui::SameLine();
            ImGui::Checkbox("Trigger to capture anywhere", &trigger_to_capture_);
            ImGui::SameLine();
            ImGui::Checkbox("Show floating capture button", &show_capture_button_);

            bool changed = false;
            changed |= ImGui::SliderFloat("HUD distance (m)", &aim_pad_distance_m_, 0.35f, 0.90f, "%.2f");
            changed |= ImGui::SliderFloat2("HUD size (m)", &aim_pad_size_m_.x, 0.10f, 1.20f, "%.2f");

            if (changed) {
                UpdateWorldCropFrame();
            }
        }

        ImGui::Separator();

        // ───────────────────────────────────────────────────────────
        // 4) MODEL LOAD/UNLOAD
        // ───────────────────────────────────────────────────────────
        if (ImGui::Button(vlm_ready_ ? "Reload VLM" : "Load VLM")) {
            if (vlm_ready_) UnloadVLM();
            onnx_status_message_.clear();
            EnsureVLMLoaded();
        }
        ImGui::SameLine();
        if (ImGui::Button("Unload VLM")) {
            UnloadVLM();
        }

        ImGui::Separator();

        // ───────────────────────────────────────────────────────────
        // 5) LAST PHOTO
        // ───────────────────────────────────────────────────────────
        ImGui::Text("Last photo:");
        ImGui::TextWrapped("%s", last_captured_image_.empty() ? "(none yet)" : last_captured_image_.c_str());

        ImGui::Separator();

        // ───────────────────────────────────────────────────────────
        // 6) STATUS / ANSWER + TTS
        // ───────────────────────────────────────────────────────────
        ImGui::Text("Status / Answer:");
        ImGui::BeginChild("onnx_scroll", ImVec2(520, 200), true);
        ImGui::TextWrapped("%s", onnx_status_message_.c_str());
        ImGui::EndChild();

        ImGui::Checkbox("Auto speak answers", &speak_auto_);
        ImGui::SameLine();
        if (ImGui::Button("▶ Speak last answer")) {
            StopListening();
            if (!last_answer_text_.empty() && last_answer_text_ != "(empty)") {
                Speak(last_answer_text_);
            } else {
                Speak("There is no answer yet.");
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Test TTS")) {
            StopListening();
            Speak("Text to speech is working.");
        }

        // Auto speak once per new answer (avoid echoing the question)
        static uint64_t last_spoken_seq = 0;
        const uint64_t seq = answer_seq_.load(std::memory_order_relaxed);
        if (speak_auto_ && seq != 0 && seq != last_spoken_seq) {
            const std::string ans = last_answer_text_;
            if (!ans.empty() && ans != "(empty)" &&
                !IsBasicallySame(ans, last_question_text_)) {
                StopListening();
                Speak(ans);
                last_spoken_seq = seq;
            }
        }
    }

    // ALWAYS end the dialog, even if opened == false
    gui.EndDialog();

    gui.EndUpdate();
    if (!running) FinishActivity();

    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(2);
}



// ========================= end of UpdateGui ===========================


//===========================endof update gui===========================================

// ===== Camera callbacks & helpers =====
//....................................start of onimageavailble.............................
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

    // Write full frame to disk
    {
        FILE* f = fopen(output_filename.c_str(), "wb");
        if (!f) { ALOGE("open %s failed: %s", output_filename.c_str(), strerror(errno)); return; }
        fwrite(output->planes[0].data, output->planes[0].size, 1, f);
        fflush(f);
        int fd = fileno(f);
        if (fd >= 0) fsync(fd);
        fclose(f);
    }

    // We will alias exactly once at the end. Default to the raw image.
    std::string best_alias_path = output_filename;
    bool did_crop = false;

    // === 1) WORLD-RECT CROP ===
    {
        int w=0,h=0,c=0;
        unsigned char* img = stbi_load(output_filename.c_str(), &w, &h, &c, 3);
        this_app->latest_image_aspect_ = (h > 0) ? (float)w/(float)h : 1.0f;

        if (img && w>0 && h>0) {
            // use the rect we had at shutter time (what you saw on HUD)
            ImVec2 ctr  = this_app->crop_at_shutter_.locked ? this_app->crop_at_shutter_.ctr  : this_app->crop_norm_center_;
            ImVec2 size = this_app->crop_at_shutter_.locked ? this_app->crop_at_shutter_.size : this_app->crop_norm_size_;
            this_app->crop_at_shutter_.locked = false;

            const int cx = (int)std::roundf(ctr.x  * w);
            const int cy = (int)std::roundf(ctr.y  * h);
            const int hw = (int)std::roundf(0.5f * size.x * w);
            const int hh = (int)std::roundf(0.5f * size.y * h);

            const int x0 = std::max(0, cx - hw), x1 = std::min(w, cx + hw);
            const int y0 = std::max(0, cy - hh), y1 = std::min(h, cy + hh);
            const int cw = x1 - x0, ch = y1 - y0;
            // ... copy & write dk_crop.jpg as you already do ...


            if (cw > 8 && ch > 8) {
                std::vector<unsigned char> crop((size_t)cw*ch*3);
                for (int y=0; y<ch; ++y) {
                    memcpy(&crop[(size_t)y*cw*3], &img[((y0+y)*w + x0)*3], (size_t)cw*3);
                }
                const std::string crop_path = this_app->default_output_filepath_ + "dk_crop.jpg";
                stbi_write_jpg(crop_path.c_str(), cw, ch, 3, crop.data(), 90);
                best_alias_path = crop_path;       // <-- set alias target here
                did_crop = true;
            }
            stbi_image_free(img);
        }
    }

    // === 2) GAZE-RETICLE CROP (if world-rect didn't happen) ===
    if (!did_crop && this_app->have_last_pad_uv_) {
        this_app->have_last_pad_uv_ = false;
        int w=0,h=0,c=0;
        unsigned char* img = stbi_load(output_filename.c_str(), &w, &h, &c, 3);
        if (img && w>0 && h>0) {
            const float margin = 0.45f;
            int cx = (int)std::roundf(w * (0.5f + margin * this_app->last_pad_uv_.x));
            int cy = (int)std::roundf(h * (0.5f - margin * this_app->last_pad_uv_.y));
            const int crop_sz = this_app->gaze_crop_px_;
            const int half    = crop_sz / 2;
            const int x0 = std::max(0, cx - half);
            const int y0 = std::max(0, cy - half);
            const int x1 = std::min(w, x0 + crop_sz);
            const int y1 = std::min(h, y0 + crop_sz);
            const int cw = x1 - x0, ch = y1 - y0;

            if (cw > 0 && ch > 0) {
                std::vector<unsigned char> crop((size_t)cw*ch*3);
                for (int y = 0; y < ch; ++y) {
                    memcpy(&crop[(size_t)y*cw*3], &img[((y0+y)*w + x0)*3], (size_t)cw*3);
                }
                const std::string crop_path = this_app->default_output_filepath_ + "dk_crop.jpg";
                stbi_write_jpg(crop_path.c_str(), cw, ch, 3, crop.data(), 90);
                best_alias_path = crop_path;       // <-- set alias target here
                did_crop = true;
            }
            stbi_image_free(img);
        }
    }

    // === 3) CLICK-LOCAL CROP (if neither of the above) ===
    if (!did_crop && this_app->have_last_click_local_) {
        this_app->have_last_click_local_ = false;
        int iw=0, ih=0, ic=0;
        unsigned char* tmp = stbi_load(output_filename.c_str(), &iw, &ih, &ic, 3);
        if (tmp) { stbi_image_free(tmp);
            glm::vec3 s = this_app->capture_button_node_->GetLocalScale();
            glm::vec2 half_btn = glm::vec2(fabsf(s.x) * 0.5f, fabsf(s.y) * 0.5f);
            const float nu = (half_btn.x > 1e-6f) ? (this_app->last_click_local_.x / half_btn.x) : 0.0f;
            const float nv = (half_btn.y > 1e-6f) ? (this_app->last_click_local_.y / half_btn.y) : 0.0f;

            const float off_scale_u = 0.15f, off_scale_v = 0.15f;
            int cx = (int)std::roundf(iw * (0.5f + 0.5f * nu * off_scale_u));
            int cy = (int)std::roundf(ih * (0.5f - 0.5f * nv * off_scale_v));

            const int crop_sz = this_app->gaze_crop_px_;
            int w=0, h=0, c=0;
            unsigned char* img = stbi_load(output_filename.c_str(), &w, &h, &c, 3);
            if (img) {
                int half = crop_sz/2;
                int x0 = std::max(0, cx - half);
                int y0 = std::max(0, cy - half);
                int x1 = std::min(w, x0 + crop_sz);
                int y1 = std::min(h, y0 + crop_sz);
                int cw = x1 - x0, ch = y1 - y0;

                if (cw > 0 && ch > 0) {
                    std::vector<unsigned char> crop((size_t)cw * ch * 3);
                    for (int y = 0; y < ch; ++y) {
                        memcpy(&crop[(size_t)y * cw * 3], &img[((y0 + y) * w + x0) * 3], (size_t)cw * 3);
                    }
                    const std::string crop_path = this_app->default_output_filepath_ + "dk_crop.jpg";
                    stbi_write_jpg(crop_path.c_str(), cw, ch, 3, crop.data(), 90);
                    best_alias_path = crop_path;   // <-- set alias target here
                    did_crop = true;
                }
                stbi_image_free(img);
            }
        }
    }

    // Alias exactly once (cropped if we made one, else raw)
    this_app->PersistLastImageAlias(best_alias_path);

    // Continue with your post-capture flow …
    if (this_app->pending_voice_question_) {
        this_app->pending_voice_question_ = false;
        this_app->awaiting_auto_question_ = true;   // NEW
        this_app->awaiting_timer_s_       = 0.0f;   // NEW
        this_app->since_last_asr_s_       = 0.0f;   // NEW


        this_app->heard_any_asr_          = false;   // NEW


        this_app->StartListening();
    } else if (this_app->send_to_vlm_after_capture_) {
        this_app->send_to_vlm_after_capture_ = false;
        std::string q = this_app->typed_question_;
        if (q.empty()) q = this_app->question_buf_;
        if (q.empty()) q = "What is in the image?";
        this_app->InitializeONNX(this_app->last_captured_image_, q);
        this_app->typed_question_.clear();
    }
}


//...................................end of onimageavailble ................
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

    camera_connect_context.mr_info.frame_rate = MLCameraCaptureFrameRate_30FPS;
    camera_connect_context.mr_info.quality = MLCameraMRQuality_2880x2160;

    camera_connect_context.mr_info.blend_type = MLCameraMRBlendType_Additive;



    const float t  = 0.04f; // thickness in meters

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


    latest_image_aspect_ = (capture_height_ > 0) ?
                           (float)capture_width_ / (float)capture_height_ : 1.0f;

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
    // decoded tokens -> raw text
    std::string decoded = SimpleDecodeGpt2Local(generated);

// Extract a clean answer (avoid prompt echoes), remove question echoes, dedup repeats
    std::string answer = ExtractAnswer(decoded);
    answer = RemoveQuestionEcho(answer, question_text);    // <-- use the ORIGINAL question_text
    answer = DedupSentences(answer);
    if (answer.empty()) answer = "(empty)";

// Persist Q/A for UI + TTS
    last_question_text_ = TrimCollapse(question_text);     // show what the user asked (not augmented q)
    last_answer_text_   = answer;
    const uint64_t seq_now = answer_seq_.fetch_add(1, std::memory_order_relaxed) + 1;

// Build the visible status text exactly once (use the SAME 'panel' defined earlier)
    panel.clear();
    panel.reserve(last_question_text_.size() + answer.size() + 32);
    panel += "Q: " + last_question_text_ + "\n";
    panel += "Answer: " + answer + "\n";


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
    std::lock_guard<std::mutex> lk(g_asrMutex);
    g_lastASR.clear();
    g_lastASRError.clear();


    if (listening_) return;

    if (!app_state_ || !app_state_->activity || !g_VoiceBridgeObj || !g_VoiceBridgeCls) {
        onnx_status_message_ = "Voice bridge not initialized.";
        return;
    }
    JNIEnv* env = nullptr; app_state_->activity->vm->AttachCurrentThread(&env, nullptr);
    if (!env) return;
    jmethodID m = env->GetMethodID(g_VoiceBridgeCls, "startListening", "()V");
    if (!m) { ALOGE("startListening not found"); return; }
    env->CallVoidMethod(g_VoiceBridgeObj, m);
    if (env->ExceptionCheck()) { env->ExceptionDescribe(); env->ExceptionClear(); return; }
    listening_ = true;
}
void CameraMixedRealityApp::StopListening() {
    if (!listening_) return;
    if (!app_state_ || !app_state_->activity || !g_VoiceBridgeObj || !g_VoiceBridgeCls) return;
    JNIEnv* env = nullptr; app_state_->activity->vm->AttachCurrentThread(&env, nullptr);
    if (!env) return;
    jmethodID m = env->GetMethodID(g_VoiceBridgeCls, "stopListening", "()V");
    if (!m) { ALOGE("stopListening not found"); return; }
    env->CallVoidMethod(g_VoiceBridgeObj, m);
    if (env->ExceptionCheck()) { env->ExceptionDescribe(); env->ExceptionClear(); return; }
    listening_ = false;
}


void CameraMixedRealityApp::Speak(const std::string& text) {
    StopListening(); // <— hard stop before speaking
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