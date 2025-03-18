#include <grpcpp/grpcpp.h>
#include "eye_tracker.grpc.pb.h"
#include <iostream>

class EyeTrackerServiceImpl final : public eye_tracker::EyeTrackerService::Service {
public:
    grpc::Status StreamEyeData(grpc::ServerContext* context,
        grpc::ServerReaderWriter<eye_tracker::FatigueAlert, eye_tracker::EyeData>* stream) override {
        eye_tracker::EyeData eye_data;
        while (stream->Read(&eye_data)) {
            float blinkRate = eye_data.blink_rate();
            float EAR = eye_data.ear();

            eye_tracker::FatigueAlert alert;
            if (blinkRate > 20 || EAR < 0.2) {
                alert.set_message("Fatigue detected! Consider taking a break.");
                alert.set_take_break(true);
                stream->Write(alert);
            }
        }
        return grpc::Status::OK;
    }
};

void RunServer() {
    std::string server_address("0.0.0.0:50051");
    EyeTrackerServiceImpl service;

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;
    server->Wait();
}

int main() {
    RunServer();
    return 0;
}
