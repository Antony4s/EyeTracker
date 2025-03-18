#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <Windows.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <vector>
#include <grpcpp/grpcpp.h>
#include "generated/eye_tracker.grpc.pb.h"

// gRPC Client
class EyeTrackerClient {
public:
    EyeTrackerClient(std::shared_ptr<grpc::Channel> channel)
        : stub_(eye_tracker::EyeTrackerService::NewStub(channel)) {
    }

    void StreamEyeData(float blinkRate, float ear) {
        grpc::ClientContext context;
        std::shared_ptr<grpc::ClientReaderWriter<eye_tracker::EyeData, eye_tracker::FatigueAlert>> stream(
            stub_->StreamEyeData(&context));

        eye_tracker::EyeData eyeData;
        eyeData.set_blink_rate(blinkRate);
        eyeData.set_ear(ear);

        if (!stream->Write(eyeData)) {
            std::cerr << "Failed to send eye tracking data.\n";
            return;
        }

        eye_tracker::FatigueAlert alert;
        while (stream->Read(&alert)) {
            std::cout << "ALERT: " << alert.message() << std::endl;
            if (alert.take_break()) {
                std::cout << "Suggesting a break!\n";
            }
        }

        stream->WritesDone();
        grpc::Status status = stream->Finish();
        if (!status.ok()) {
            std::cerr << "gRPC stream error: " << status.error_message() << std::endl;
        }
    }

private:
    std::unique_ptr<eye_tracker::EyeTrackerService::Stub> stub_;
};

// ----------------- Mouse Control -----------------
void moveMouse(int x, int y) {
    SetCursorPos(x, y);
}

void singleClickMouse() {
    INPUT input = { 0 };
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    SendInput(1, &input, sizeof(INPUT));
    input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
    SendInput(1, &input, sizeof(INPUT));
}

// ----------------- EAR Calculation -----------------
double calculateEAR(const dlib::full_object_detection& shape, bool leftEye) {
    int start = leftEye ? 36 : 42;
    std::vector<cv::Point> pts;
    for (int i = 0; i < 6; i++) {
        pts.emplace_back(shape.part(start + i).x(), shape.part(start + i).y());
    }
    double distV1 = cv::norm(pts[1] - pts[5]);
    double distV2 = cv::norm(pts[2] - pts[4]);
    double distH = cv::norm(pts[0] - pts[3]);
    return (distV1 + distV2) / (2.0 * distH);
}

// ----------------- Averaging Eye Landmarks -----------------
cv::Point2f averageEyeLandmarks(const dlib::full_object_detection& shape, int startIdx) {
    cv::Point2f sum(0, 0);
    for (int i = 0; i < 6; i++) {
        sum += cv::Point2f(shape.part(startIdx + i).x(), shape.part(startIdx + i).y());
    }
    sum *= (1.0f / 6.0f);
    return sum;
}

// ----------------- Helper: Median-based Outlier Rejection -----------------
std::vector<cv::Point2f> removeOutliers(const std::vector<cv::Point2f>& points, float threshold) {
    if (points.size() < 3) {
        return points;
    }
    std::vector<float> xs, ys;
    xs.reserve(points.size());
    ys.reserve(points.size());
    for (auto& p : points) {
        xs.push_back(p.x);
        ys.push_back(p.y);
    }
    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());
    float medianX = xs[xs.size() / 2];
    float medianY = ys[ys.size() / 2];

    std::vector<cv::Point2f> filtered;
    for (auto& p : points) {
        float dx = p.x - medianX;
        float dy = p.y - medianY;
        if ((dx * dx + dy * dy) < (threshold * threshold)) {
            filtered.push_back(p);
        }
    }
    return filtered;
}

// ----------------- Main Function -----------------
int main() {
    EyeTrackerClient client(grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials()));

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera!\n";
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;
    try {
        dlib::deserialize("models/shape_predictor_68_face_landmarks.dat") >> predictor;
    }
    catch (...) {
        std::cerr << "Failed to load shape predictor. Check path!\n";
        return -1;
    }

    double EAR_THRESHOLD = 0.22;
    int blinkCounter = 0;
    bool blinkTriggered = false;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        dlib::cv_image<dlib::bgr_pixel> dlibImg(frame);
        std::vector<dlib::rectangle> faces = detector(dlibImg);

        if (!faces.empty()) {
            dlib::full_object_detection shape = predictor(dlibImg, faces[0]);

            double leftEAR = calculateEAR(shape, true);
            double rightEAR = calculateEAR(shape, false);
            double ear = std::min(leftEAR, rightEAR);

            if (ear < EAR_THRESHOLD) {
                blinkCounter++;
            }
            else {
                if (blinkCounter >= 2) {
                    if (!blinkTriggered) {
                        singleClickMouse();
                        blinkTriggered = true;
                    }
                }
                blinkCounter = 0;
            }
            if (blinkCounter == 0) {
                blinkTriggered = false;
            }

            // Send data to gRPC server
            client.StreamEyeData(blinkCounter, ear);
        }

        cv::putText(frame, "Tracking Active", cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Camera", frame);

        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
