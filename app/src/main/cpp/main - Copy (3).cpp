#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include <iostream>

int main() {
    const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort) {
        std::cerr << "Failed to get ONNX Runtime API interface." << std::endl;
        return -1;
    }

    OrtEnv* env = nullptr;
    OrtStatus* status = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ML2App", &env);
    if (status != nullptr) {
        const char* errMsg = ort->GetErrorMessage(status);
        std::cerr << "OrtCreateEnv failed: " << (errMsg ? errMsg : "unknown error") << std::endl;
        ort->ReleaseStatus(status);
        return -1;
    }

    std::cout << "ONNX Runtime environment created successfully!" << std::endl;
    ort->ReleaseEnv(env);
    return 0;
}
