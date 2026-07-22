#include "rclcpp/rclcpp.hpp"
#include <onnxruntime_cxx_api.h>

class ONNXNode : public rclcpp::Node
{
public:
    ONNXNode() : Node("onnx_node")
    {
        RCLCPP_INFO(this->get_logger(), "ONNX Node Started");

        // Initialize ONNX
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "onnx");

        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(1);

        std::string model_path = "/home/user/Documents/ros2_ws_ict_nav2/src/rl_nav_cpp/model.onnx";

        session_ = std::make_unique<Ort::Session>(
            *env_, model_path.c_str(), *session_options_);

        // Timer (runs every second)
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&ONNXNode::runInference, this)
        );
    }

private:
    void runInference()
    {
        std::vector<float> obs = {0,0,0, 0,0,0, 0,0};

        std::array<int64_t, 2> shape{1, 8};

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, obs.data(), obs.size(), shape.data(), shape.size());

        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1);

        float* output = output_tensors[0].GetTensorMutableData<float>();

        RCLCPP_INFO(this->get_logger(),
            "Action: [%f, %f]", output[0], output[1]);
    }

    // ROS
    rclcpp::TimerBase::SharedPtr timer_;

    // ONNX
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ONNXNode>());
    rclcpp::shutdown();
    return 0;
}