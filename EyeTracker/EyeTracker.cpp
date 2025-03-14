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
    // 68-point model: left eye = indices [36..41], right eye = [42..47]
    int start = leftEye ? 36 : 42;
    std::vector<cv::Point> pts;
    for (int i = 0; i < 6; i++) {
        pts.emplace_back(shape.part(start + i).x(), shape.part(start + i).y());
    }
    double distV1 = cv::norm(pts[1] - pts[5]);
    double distV2 = cv::norm(pts[2] - pts[4]);
    double distH = cv::norm(pts[0] - pts[3]);
    double ear = (distV1 + distV2) / (2.0 * distH);
    return ear;
}

// ----------------- Averaging Eye Landmarks -----------------
cv::Point2f averageEyeLandmarks(const dlib::full_object_detection& shape, int startIdx) {
    // sum up 6 landmarks for one eye, then /6
    cv::Point2f sum(0, 0);
    for (int i = 0; i < 6; i++) {
        sum += cv::Point2f(shape.part(startIdx + i).x(), shape.part(startIdx + i).y());
    }
    sum *= (1.0f / 6.0f);
    return sum;
}

// ----------------- Helper: Median-based Outlier Rejection -----------------
// Removes samples that are too far from the median eye point.
std::vector<cv::Point2f> removeOutliers(const std::vector<cv::Point2f>& points, float threshold) {
    if (points.size() < 3) {
        return points; // too few to do outlier rejection
    }
    // Compute median
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

    // Filter out points whose distance from median > threshold
    std::vector<cv::Point2f> filtered;
    filtered.reserve(points.size());
    for (auto& p : points) {
        float dx = p.x - medianX;
        float dy = p.y - medianY;
        float dist2 = dx * dx + dy * dy;
        if (dist2 < threshold * threshold) {
            filtered.push_back(p);
        }
    }
    return filtered;
}

int main() {
    // 1) Open camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera!\n";
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    // 2) Screen dimensions
    int screenW = GetSystemMetrics(SM_CXSCREEN);
    int screenH = GetSystemMetrics(SM_CYSCREEN);

    // 3) Dlib: face detector & predictor
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;
    try {
        dlib::deserialize("models/shape_predictor_68_face_landmarks.dat") >> predictor;
    }
    catch (...) {
        std::cerr << "Failed to load shape predictor. Check path!\n";
        return -1;
    }

    // 4) Build a 3x3 grid of calibration points (normalized screen coords)
    // e.g. (0.2,0.2), (0.5,0.2), (0.8,0.2) ... (0.8,0.8)
    // Adjust as desired. We avoid extreme edges (like 0.0 or 1.0).
    std::vector<cv::Point2f> screenPointsNorm = {
        {0.2f, 0.2f}, {0.5f, 0.2f}, {0.8f, 0.2f},
        {0.2f, 0.5f}, {0.5f, 0.5f}, {0.8f, 0.5f},
        {0.2f, 0.8f}, {0.5f, 0.8f}, {0.8f, 0.8f}
    };
    // Convert normalized coords to actual screen coords
    std::vector<cv::Point2f> screenPoints;
    screenPoints.reserve(screenPointsNorm.size());
    for (auto& norm : screenPointsNorm) {
        screenPoints.push_back(cv::Point2f(norm.x * screenW, norm.y * screenH));
    }

    // We'll store the matching eyePoints (the averaged eye center for each calibration).
    std::vector<cv::Point2f> eyePoints;
    eyePoints.reserve(screenPoints.size());

    // For each calibration point, we collect N frames
    const int CALIB_FRAMES = 60; // gather 60 frames per point
    const float OUTLIER_THRESHOLD = 30.f; // Euclidian threshold for discarding outliers

    // We'll create a full-screen window for calibration
    // so the user can see exactly where to look.
    const std::string calibWindowName = "Calibration";
    cv::namedWindow(calibWindowName, cv::WINDOW_NORMAL);
    cv::setWindowProperty(calibWindowName, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    // We also create a smaller window to show camera feed for debugging
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    // 5) Blink detection config
    double EAR_THRESHOLD = 0.22;
    int BLINK_MIN_FRAMES = 2;
    int blinkCounter = 0;
    bool blinkTriggered = false;

    // We'll track the largest face every few frames
    std::vector<dlib::rectangle> faces;
    int detectInterval = 5;
    int detectFrameCount = 0;

    // Eye-based smoothing
    cv::Point cursorPos(screenW / 2, screenH / 2);

    // 6) --- PHASE 1: CALIBRATION ---
    for (size_t i = 0; i < screenPoints.size(); i++) {
        // Show the user a dot on the calibration window
        // We'll display a black image with a single circle at the calibration point
        cv::Mat calibImg(screenH, screenW, CV_8UC3, cv::Scalar(0, 0, 0));

        // Draw a circle where we want them to look
        cv::circle(calibImg,
            cv::Point((int)screenPoints[i].x, (int)screenPoints[i].y),
            20, cv::Scalar(0, 255, 255), cv::FILLED);

        // Label
        std::string txt = "Calibration point " + std::to_string(i + 1)
            + "/" + std::to_string(screenPoints.size());
        cv::putText(calibImg, txt, cv::Point(50, 50),
            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

        // Show it full-screen
        cv::imshow(calibWindowName, calibImg);
        cv::waitKey(1); // tiny wait to refresh display

        // Collect frames for this calibration point
        std::vector<cv::Point2f> samples;
        samples.reserve(CALIB_FRAMES);

        int frameCollected = 0;
        while (frameCollected < CALIB_FRAMES) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) break;

            // show camera for debugging
            cv::imshow("Camera", frame);

            // Face detect every detectInterval frames
            if (detectFrameCount % detectInterval == 0) {
                dlib::cv_image<dlib::bgr_pixel> dlibImg(frame);
                faces = detector(dlibImg);
            }
            detectFrameCount++;

            if (!faces.empty()) {
                // largest face
                auto faceRect = *std::max_element(
                    faces.begin(), faces.end(),
                    [](auto& a, auto& b) {
                        return a.area() < b.area();
                    }
                );
                dlib::full_object_detection shape = predictor(dlib::cv_image<dlib::bgr_pixel>(frame), faceRect);

                // Blink detection
                double leftEAR = calculateEAR(shape, true);
                double rightEAR = calculateEAR(shape, false);
                double ear = std::min(leftEAR, rightEAR);

                if (ear < EAR_THRESHOLD) {
                    blinkCounter++;
                }
                else {
                    if (blinkCounter >= BLINK_MIN_FRAMES) {
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

                // Eye center
                cv::Point2f leftAvg = averageEyeLandmarks(shape, 36);
                cv::Point2f rightAvg = averageEyeLandmarks(shape, 42);
                cv::Point2f midEye = 0.5f * (leftAvg + rightAvg);

                // Collect sample
                samples.push_back(midEye);
                frameCollected++;
            }

            // Press ESC to abort calibration
            int key = cv::waitKey(1);
            if (key == 27) { // ESC
                std::cout << "Calibration aborted by user.\n";
                return 0;
            }
        }

        // We have N=CALIB_FRAMES samples. Let's remove outliers.
        auto filtered = removeOutliers(samples, OUTLIER_THRESHOLD);
        // Average what's left
        if (filtered.empty()) {
            // fallback if all were outliers
            std::cerr << "All outliers for point " << i << ". Using raw samples.\n";
            filtered = samples;
        }
        cv::Point2f sum(0, 0);
        for (auto& p : filtered) sum += p;
        sum.x /= (float)filtered.size();
        sum.y /= (float)filtered.size();

        eyePoints.push_back(sum);
        std::cout << "[Calib " << (i + 1) << "/" << screenPoints.size() << "] final eye=("
            << sum.x << "," << sum.y << "), from " << filtered.size()
            << " valid samples\n";
    }

    // Close the calibration window
    cv::destroyWindow(calibWindowName);

    // 7) Compute homography (eyePoints -> screenPoints)
    cv::Mat transformMat = cv::findHomography(eyePoints, screenPoints, cv::RANSAC, 3.0);
    if (transformMat.empty()) {
        std::cerr << "Homography computation failed! Exiting.\n";
        return 0;
    }
    std::cout << "Calibration complete. Entering tracking mode...\n";

    // 8) --- PHASE 2: TRACKING ---
    // We'll keep showing the "Camera" window. We won't open a second big window now.
    bool tracking = true;
    while (tracking) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Face detect
        if (detectFrameCount % detectInterval == 0) {
            dlib::cv_image<dlib::bgr_pixel> dlibImg(frame);
            faces = detector(dlibImg);
        }
        detectFrameCount++;

        if (!faces.empty()) {
            auto faceRect = *std::max_element(
                faces.begin(), faces.end(),
                [](auto& a, auto& b) {
                    return a.area() < b.area();
                }
            );
            dlib::full_object_detection shape = predictor(dlib::cv_image<dlib::bgr_pixel>(frame), faceRect);

            // Blink
            double leftEAR = calculateEAR(shape, true);
            double rightEAR = calculateEAR(shape, false);
            double ear = std::min(leftEAR, rightEAR);
            if (ear < EAR_THRESHOLD) {
                blinkCounter++;
            }
            else {
                if (blinkCounter >= BLINK_MIN_FRAMES) {
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

            // Eye center
            cv::Point2f leftAvg = averageEyeLandmarks(shape, 36);
            cv::Point2f rightAvg = averageEyeLandmarks(shape, 42);
            cv::Point2f midEye = 0.5f * (leftAvg + rightAvg);

            // Transform to screen
            std::vector<cv::Point2f> src(1, midEye), dst(1);
            cv::perspectiveTransform(src, dst, transformMat);

            // Check if valid
            if (std::isfinite(dst[0].x) && std::isfinite(dst[0].y)) {
                // Exponential smoothing
                float alpha = 0.1f;  // smaller => more smoothing
                cursorPos.x = (int)(alpha * cursorPos.x + (1.0f - alpha) * dst[0].x);
                cursorPos.y = (int)(alpha * cursorPos.y + (1.0f - alpha) * dst[0].y);

                // Clamp
                cursorPos.x = std::max(0, std::min(cursorPos.x, screenW - 1));
                cursorPos.y = std::max(0, std::min(cursorPos.y, screenH - 1));

                moveMouse(cursorPos.x, cursorPos.y);
            }
        }

        // Show camera
        cv::putText(frame, "Tracking Active", cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Camera", frame);

        int key = cv::waitKey(1);
        if (key == 27) { // ESC
            tracking = false;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
