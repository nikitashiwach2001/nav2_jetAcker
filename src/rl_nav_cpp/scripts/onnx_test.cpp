#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    Ort::Session session(env, "/home/user/Documents/ros2_ws_ict_nav2/src/model.onnx", session_options);

    std::vector<float> input = {0,0,0, 0,0,0, 0,0};  // same shape (8)

    std::array<int64_t, 2> shape{1, 8};

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input.data(), input.size(), shape.data(), shape.size());

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1);

    float* output = output_tensors[0].GetTensorMutableData<float>();

    std::cout << "Output: " << output[0] << ", " << output[1] << std::endl;

    return 0;
}