// Двухэтапный видеодетектор: Haar-каскад (поиск лица) -> CNN в ONNX (очки / без очков)
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    // Аргументы (все необязательные): модель, каскад, видеофайл
    std::string modelPath   = argc > 1 ? argv[1] : "glasses_cnn.onnx";
    std::string cascadePath = argc > 2 ? argv[2] : "haarcascade_frontalface_default.xml";
    std::string videoPath   = argc > 3 ? argv[3] : "";

    // Порядок классов СТРОГО как в Python (ImageFolder сортирует папки по алфавиту)
    const std::vector<std::string> labels = {"No glasses", "Glasses"};
    const int IMG = 96;  // размер входа сети, как при обучении

    // ---- Этап 0: загрузка ресурсов ----
    cv::CascadeClassifier cascade;
    if (!cascade.load(cascadePath)) {
        std::cerr << "Не удалось загрузить каскад: " << cascadePath << std::endl;
        return 1;
    }
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    if (net.empty()) {
        std::cerr << "Не удалось загрузить модель: " << modelPath << std::endl;
        return 1;
    }

    cv::VideoCapture cap;
    if (videoPath.empty()) cap.open(0);
    else cap.open(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Не удалось открыть камеру/видео" << std::endl;
        return 1;
    }

    bool recording = false;
    int saved = 0;
    std::filesystem::create_directories("saved_roi");

    cv::Mat frame, gray;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;  // конец видео / камера отвалилась

        // ---- Этап 1: классический поиск ROI (Haar-каскад) ----
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);  // выравнивание яркости
        std::vector<cv::Rect> faces;
        // minNeighbors=6 и minSize отсекают ложные срабатывания
        cascade.detectMultiScale(gray, faces, 1.1, 6, 0, cv::Size(80, 80));

        // Фильтрация: оставляем только самое большое лицо
        cv::Rect best;
        for (const auto& f : faces)
            if (f.area() > best.area()) best = f;

        if (best.area() > 0) {
            // ---- Защита границ: пересечение рамки с границами кадра ----
            cv::Rect safe = best & cv::Rect(0, 0, frame.cols, frame.rows);
            if (safe.width >= 20 && safe.height >= 20) {
                // ROI без копирования пикселей (подматрица ссылается на тот же буфер)
                cv::Mat roi = frame(safe);

                // ---- Этап 2: тензор NCHW, RGB, значения 0..1, 96x96 ----
                cv::Mat blob = cv::dnn::blobFromImage(
                    roi, 1.0 / 255.0, cv::Size(IMG, IMG), cv::Scalar(),
                    /*swapRB=*/true, /*crop=*/false);
                net.setInput(blob);
                cv::Mat out = net.forward();  // 1 x num_classes (вероятности)

                cv::Point maxLoc;
                double maxVal;
                cv::minMaxLoc(out, nullptr, &maxVal, nullptr, &maxLoc);
                int cls = maxLoc.x;

                // Рамка и подпись с контрастной обводкой
                cv::Scalar color = (cls == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
                cv::rectangle(frame, safe, color, 2);
                std::string text = labels[cls] + " " + cv::format("%.0f%%", maxVal * 100);
                cv::Point org(safe.x, std::max(safe.y - 10, 25));
                cv::putText(frame, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 4);
                cv::putText(frame, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);

                // Сохранение ROI на диск, если включена запись
                if (recording) {
                    std::string name = cv::format("saved_roi/roi_%05d.jpg", saved++);
                    cv::imwrite(name, roi);
                }
            }
        }

        // Индикатор REC
        if (recording) {
            cv::circle(frame, cv::Point(30, 30), 10, cv::Scalar(0, 0, 255), cv::FILLED);
            cv::putText(frame, "REC", cv::Point(50, 40), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                        cv::Scalar(0, 0, 0), 4);
            cv::putText(frame, "REC", cv::Point(50, 40), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                        cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("Two-stage detector", frame);
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27) break;                          // ESC
        if (key == 'r' || key == 'R') recording = !recording;  // триггер записи
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
