#include <VmbCPP/VmbCPP.h>
#include <VmbCPP/thirdparty/OpenCV.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace VmbCPP;

namespace {

constexpr int kBufferCount = 96;
constexpr int kPreviewPollMs = 1;
constexpr int kDefaultDurationSeconds = 30;
constexpr double kDefaultFps = 68.0;
constexpr double kFallbackExposureUs = 3000.0;

std::string ErrorToString(const std::string& context, VmbErrorType err)
{
    return context + ", err=" + std::to_string(static_cast<int>(err));
}

void ThrowIfError(const std::string& context, VmbErrorType err)
{
    if (err != VmbErrorSuccess)
    {
        throw std::runtime_error(ErrorToString(context, err));
    }
}

std::string GetCameraLabel(const CameraPtr& camera)
{
    std::string name;
    std::string id;
    camera->GetName(name);
    camera->GetID(id);
    if (name.empty())
    {
        return id;
    }
    return name + " (" + id + ")";
}

bool IsAlliedVisionCamera(const CameraPtr& camera)
{
    std::string name;
    std::string model;
    std::string id;
    camera->GetName(name);
    camera->GetModel(model);
    camera->GetID(id);

    const std::string combined = name + " " + model + " " + id;
    return combined.find("Allied Vision") != std::string::npos
        || combined.find("ALVIUM") != std::string::npos
        || combined.find("Alvium") != std::string::npos
        || combined.find("DEV_1AB2") != std::string::npos;
}

void SetEnumFeatureIfAvailable(const CameraPtr& camera, const char* featureName, const char* value)
{
    FeaturePtr feature;
    if (camera->GetFeatureByName(featureName, feature) == VmbErrorSuccess && feature)
    {
        feature->SetValue(value);
    }
}

void SetFloatFeatureIfAvailable(const CameraPtr& camera, const char* featureName, double value)
{
    FeaturePtr feature;
    if (camera->GetFeatureByName(featureName, feature) == VmbErrorSuccess && feature)
    {
        feature->SetValue(value);
    }
}

void SetBoolFeatureIfAvailable(const CameraPtr& camera, const char* featureName, bool value)
{
    FeaturePtr feature;
    if (camera->GetFeatureByName(featureName, feature) == VmbErrorSuccess && feature)
    {
        feature->SetValue(value);
    }
}

void ConfigureCamera(const CameraPtr& camera, double fps, double exposureUs)
{
    SetEnumFeatureIfAvailable(camera, "TriggerMode", "Off");
    SetEnumFeatureIfAvailable(camera, "AcquisitionMode", "Continuous");
    SetEnumFeatureIfAvailable(camera, "ExposureAuto", "Off");
    SetBoolFeatureIfAvailable(camera, "AcquisitionFrameRateEnable", true);
    SetFloatFeatureIfAvailable(camera, "AcquisitionFrameRate", fps);
    SetFloatFeatureIfAvailable(camera, "ExposureTime", exposureUs);
    SetFloatFeatureIfAvailable(camera, "ExposureTimeAbs", exposureUs);

    FeaturePtr pixelFormat;
    if (camera->GetFeatureByName("PixelFormat", pixelFormat) == VmbErrorSuccess && pixelFormat)
    {
        if (pixelFormat->SetValue("Mono8") != VmbErrorSuccess)
        {
            pixelFormat->SetValue("Bgr8");
        }
    }
}

fs::path DetectLocalGentlDirectory()
{
    const fs::path localPath = fs::current_path() / "VimbaX_2026-1" / "cti";
    if (fs::exists(localPath / "VimbaUSBTL.cti"))
    {
        return localPath;
    }
    return {};
}

void EnsureGentlPath()
{
    const char* existing = std::getenv("GENICAM_GENTL64_PATH");
    if (existing && *existing)
    {
        return;
    }

    const fs::path localGentl = DetectLocalGentlDirectory();
    if (!localGentl.empty())
    {
        setenv("GENICAM_GENTL64_PATH", localGentl.c_str(), 1);
    }
}

template <typename T>
T PromptValue(const std::string& label, T defaultValue)
{
    for (;;)
    {
        std::cout << label << " [" << defaultValue << "]: " << std::flush;
        std::string line;
        if (!std::getline(std::cin, line))
        {
            throw std::runtime_error("Input stream closed.");
        }
        if (line.empty())
        {
            return defaultValue;
        }

        std::istringstream parser(line);
        T value {};
        if ((parser >> value) && parser.eof() && value > static_cast<T>(0))
        {
            return value;
        }
        std::cout << "Enter a positive number.\n";
    }
}

fs::path CreateNextRecordingDirectory(const fs::path& root)
{
    fs::create_directories(root);

    std::regex pattern(R"(recording_(\d+))");
    int maxIndex = 0;

    for (const auto& entry : fs::directory_iterator(root))
    {
        if (!entry.is_directory())
        {
            continue;
        }

        std::smatch match;
        const std::string name = entry.path().filename().string();
        if (std::regex_match(name, match, pattern))
        {
            maxIndex = std::max(maxIndex, std::stoi(match[1].str()));
        }
    }

    const fs::path recordingDir = root / ("recording_" + std::to_string(maxIndex + 1));
    fs::create_directories(recordingDir);
    return recordingDir;
}

struct RecorderStats
{
    std::uint64_t receivedFrames = 0;
    std::uint64_t writtenFrames = 0;
    std::uint64_t droppedFrames = 0;
};

struct RecordingSettings
{
    int durationSeconds = kDefaultDurationSeconds;
    double fps = kDefaultFps;
    double exposureUs = kFallbackExposureUs;
};

double ReadCurrentExposureUs(const CameraPtr& camera, double fallbackValue)
{
    for (const char* featureName : {"ExposureTime", "ExposureTimeAbs"})
    {
        FeaturePtr feature;
        if (camera->GetFeatureByName(featureName, feature) == VmbErrorSuccess && feature)
        {
            double value = 0.0;
            if (feature->GetValue(value) == VmbErrorSuccess && value > 0.0)
            {
                return value;
            }
        }
    }

    return fallbackValue;
}

class CameraRecorder;

class FrameObserver final : public IFrameObserver
{
public:
    FrameObserver(const CameraPtr& camera, CameraRecorder& recorder);
    void FrameReceived(const FramePtr frame) override;

private:
    CameraRecorder& m_recorder;
};

class CameraRecorder
{
public:
    CameraRecorder(CameraPtr camera, std::string windowName, fs::path outputPath, double fps) :
        m_camera(std::move(camera)),
        m_windowName(std::move(windowName)),
        m_outputPath(std::move(outputPath)),
        m_fps(fps),
        m_frameInterval(std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(1.0 / fps)
        )),
        m_observer(new FrameObserver(m_camera, *this))
    {}

    void Start()
    {
        m_stopRequested = false;
        m_started = false;
        m_worker = std::thread(&CameraRecorder::WriterLoop, this);
        ThrowIfError("Could not start acquisition", m_camera->StartContinuousImageAcquisition(kBufferCount, m_observer));
    }

    void Stop()
    {
        if (!m_stopRequested.exchange(true))
        {
            m_frameCondition.notify_all();
            m_camera->StopContinuousImageAcquisition();
        }

        if (m_worker.joinable())
        {
            m_worker.join();
        }

        if (m_writer.isOpened())
        {
            m_writer.release();
        }
    }

    ~CameraRecorder()
    {
        try
        {
            Stop();
        }
        catch (...)
        {
        }
    }

    void OnFrameReceived(const FramePtr& frame)
    {
        cv::Mat image;
        const VmbErrorType convertErr = VmbFrameToMat(frame, image);
        if (convertErr == VmbErrorSuccess && !image.empty())
        {
            std::lock_guard<std::mutex> lock(m_frameMutex);
            ++m_stats.receivedFrames;
            m_latestFrame = image.clone();
            if (!m_started)
            {
                m_started = true;
                m_nextWriteTime = std::chrono::steady_clock::now();
            }
        }

        if (!m_stopRequested.load())
        {
            m_camera->QueueFrame(frame);
        }
        m_frameCondition.notify_one();
    }

    bool IsStopping() const
    {
        return m_stopRequested.load();
    }

    std::optional<cv::Mat> GetLatestPreview()
    {
        std::lock_guard<std::mutex> lock(m_previewMutex);
        if (m_latestPreview.empty())
        {
            return std::nullopt;
        }
        return m_latestPreview.clone();
    }

    RecorderStats GetStats() const
    {
        return m_stats;
    }

    const std::string& GetWindowName() const
    {
        return m_windowName;
    }

private:
    void WriterLoop()
    {
        for (;;)
        {
            cv::Mat frameToWrite;
            {
                std::unique_lock<std::mutex> lock(m_frameMutex);
                m_frameCondition.wait(lock, [&] {
                    return m_stopRequested.load() || !m_latestFrame.empty();
                });

                if (m_stopRequested.load())
                {
                    break;
                }

                if (m_latestFrame.empty())
                {
                    continue;
                }

                if (!m_started)
                {
                    continue;
                }

                const auto now = std::chrono::steady_clock::now();
                if (now < m_nextWriteTime)
                {
                    m_frameCondition.wait_until(lock, m_nextWriteTime, [&] {
                        return m_stopRequested.load();
                    });
                    continue;
                }

                frameToWrite = m_latestFrame.clone();
                m_nextWriteTime += m_frameInterval;
            }

            if (!frameToWrite.empty())
            {
                EnsureWriter(frameToWrite);
                m_writer.write(frameToWrite);
                ++m_stats.writtenFrames;

                {
                    std::lock_guard<std::mutex> lock(m_previewMutex);
                    m_latestPreview = frameToWrite;
                }
            }
        }
    }

    void EnsureWriter(const cv::Mat& image)
    {
        if (m_writer.isOpened())
        {
            return;
        }

        const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        const bool isColor = image.channels() != 1;
        if (!m_writer.open(m_outputPath.string(), fourcc, m_fps, image.size(), isColor))
        {
            throw std::runtime_error("Could not open video writer: " + m_outputPath.string());
        }
    }

    CameraPtr m_camera;
    std::string m_windowName;
    fs::path m_outputPath;
    double m_fps;
    std::chrono::steady_clock::duration m_frameInterval;
    IFrameObserverPtr m_observer;

    std::atomic<bool> m_stopRequested {false};
    bool m_started = false;
    std::thread m_worker;
    std::mutex m_frameMutex;
    std::condition_variable m_frameCondition;
    cv::Mat m_latestFrame;
    std::chrono::steady_clock::time_point m_nextWriteTime;

    mutable std::mutex m_previewMutex;
    cv::Mat m_latestPreview;
    cv::VideoWriter m_writer;
    RecorderStats m_stats;
};

FrameObserver::FrameObserver(const CameraPtr& camera, CameraRecorder& recorder) :
    IFrameObserver(camera),
    m_recorder(recorder)
{
}

void FrameObserver::FrameReceived(const FramePtr frame)
{
    VmbFrameStatusType status = VmbFrameStatusInvalid;
    const VmbErrorType err = frame->GetReceiveStatus(status);
    if (err == VmbErrorSuccess && status == VmbFrameStatusComplete)
    {
        m_recorder.OnFrameReceived(frame);
        return;
    }

    if (!m_recorder.IsStopping())
    {
        m_pCamera->QueueFrame(frame);
    }
}

class VmbSession
{
public:
    VmbSession() :
        m_system(VmbSystem::GetInstance())
    {
        ThrowIfError("Could not start Vimba X API", m_system.Startup());
    }

    ~VmbSession()
    {
        m_system.Shutdown();
    }

    VmbSystem& Get()
    {
        return m_system;
    }

private:
    VmbSystem& m_system;
};

void PrintBanner()
{
    std::cout << "\n";
    std::cout << "=====================================\n";
    std::cout << "         Alvium Recorder             \n";
    std::cout << "=====================================\n\n";
}

bool PromptYesNo(const std::string& question, bool defaultYes)
{
    for (;;)
    {
        std::cout << question << (defaultYes ? " [Y/n]: " : " [y/N]: ") << std::flush;
        std::string line;
        if (!std::getline(std::cin, line))
        {
            return false;
        }

        if (line.empty())
        {
            return defaultYes;
        }

        if (line == "y" || line == "Y")
        {
            return true;
        }
        if (line == "n" || line == "N")
        {
            return false;
        }
    }
}

std::vector<std::size_t> PromptCameraSelection(const std::vector<CameraPtr>& cameras)
{
    for (;;)
    {
        std::cout << "Available Allied Vision cameras:\n";
        for (std::size_t index = 0; index < cameras.size(); ++index)
        {
            std::cout << "  " << (index + 1) << ". " << GetCameraLabel(cameras[index]) << "\n";
        }
        std::cout << "\nSelect cameras to record "
                  << "(examples: 1,2 or 1-3 or all) [all]: "
                  << std::flush;

        std::string line;
        if (!std::getline(std::cin, line))
        {
            throw std::runtime_error("Input stream closed.");
        }

        if (line.empty() || line == "all" || line == "ALL")
        {
            std::vector<std::size_t> allIndexes;
            for (std::size_t index = 0; index < cameras.size(); ++index)
            {
                allIndexes.push_back(index);
            }
            return allIndexes;
        }

        std::vector<std::size_t> selectedIndexes;
        std::unordered_set<std::size_t> seen;
        std::stringstream parser(line);
        std::string token;
        bool valid = true;

        while (std::getline(parser, token, ','))
        {
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t") + 1);
            if (token.empty())
            {
                valid = false;
                break;
            }

            const auto dashPos = token.find('-');
            if (dashPos != std::string::npos)
            {
                const std::string startPart = token.substr(0, dashPos);
                const std::string endPart = token.substr(dashPos + 1);
                int start = 0;
                int end = 0;
                std::istringstream startStream(startPart);
                std::istringstream endStream(endPart);
                if (!(startStream >> start) || !(endStream >> end) || start <= 0 || end <= 0 || start > end)
                {
                    valid = false;
                    break;
                }

                for (int current = start; current <= end; ++current)
                {
                    const std::size_t zeroBased = static_cast<std::size_t>(current - 1);
                    if (zeroBased >= cameras.size())
                    {
                        valid = false;
                        break;
                    }
                    if (seen.insert(zeroBased).second)
                    {
                        selectedIndexes.push_back(zeroBased);
                    }
                }

                if (!valid)
                {
                    break;
                }
            }
            else
            {
                int value = 0;
                std::istringstream valueStream(token);
                if (!(valueStream >> value) || value <= 0)
                {
                    valid = false;
                    break;
                }

                const std::size_t zeroBased = static_cast<std::size_t>(value - 1);
                if (zeroBased >= cameras.size())
                {
                    valid = false;
                    break;
                }

                if (seen.insert(zeroBased).second)
                {
                    selectedIndexes.push_back(zeroBased);
                }
            }
        }

        if (valid && !selectedIndexes.empty())
        {
            return selectedIndexes;
        }

        std::cout << "Invalid selection. Try again.\n\n";
    }
}

} // namespace

int main()
{
    try
    {
        EnsureGentlPath();
        cv::setNumThreads(1);
        PrintBanner();

        VmbSession session;
        CameraPtrVector allCameras;
        ThrowIfError("Could not enumerate cameras", session.Get().GetCameras(allCameras));

        std::vector<CameraPtr> alliedVisionCameras;
        for (const auto& camera : allCameras)
        {
            if (IsAlliedVisionCamera(camera))
            {
                alliedVisionCameras.push_back(camera);
            }
        }

        if (alliedVisionCameras.empty())
        {
            throw std::runtime_error("No Allied Vision cameras found.");
        }

        const std::vector<std::size_t> selectedIndexes = PromptCameraSelection(alliedVisionCameras);
        std::vector<CameraPtr> selectedCameras;
        selectedCameras.reserve(selectedIndexes.size());
        for (std::size_t selectedIndex : selectedIndexes)
        {
            selectedCameras.push_back(alliedVisionCameras[selectedIndex]);
        }

        std::cout << "Using cameras:\n";
        for (std::size_t index = 0; index < selectedCameras.size(); ++index)
        {
            ThrowIfError(
                "Could not open selected camera " + std::to_string(index + 1),
                selectedCameras[index]->Open(VmbAccessModeFull)
            );
            std::cout << "  " << (index + 1) << ". " << GetCameraLabel(selectedCameras[index]) << "\n";
        }
        std::cout << "\n";

        const fs::path recordingsRoot = fs::current_path() / "recordings";

        for (;;)
        {
            RecordingSettings settings;
            const double defaultExposureUs = ReadCurrentExposureUs(selectedCameras.front(), kFallbackExposureUs);
            settings.durationSeconds = PromptValue<int>("Recording length in seconds", kDefaultDurationSeconds);
            settings.fps = PromptValue<double>("Target FPS", kDefaultFps);
            settings.exposureUs = PromptValue<double>("Exposure in microseconds", defaultExposureUs);

            const fs::path recordingDir = CreateNextRecordingDirectory(recordingsRoot);

            std::cout << "\nOutput directory: " << recordingDir << "\n";
            std::cout << "Starting recording. Press 'q' in a preview window to stop early.\n\n";

            std::vector<std::string> windowNames;
            std::vector<fs::path> outputPaths;
            std::vector<std::unique_ptr<CameraRecorder>> recorders;
            windowNames.reserve(selectedCameras.size());
            outputPaths.reserve(selectedCameras.size());
            recorders.reserve(selectedCameras.size());

            for (std::size_t index = 0; index < selectedCameras.size(); ++index)
            {
                ConfigureCamera(selectedCameras[index], settings.fps, settings.exposureUs);

                const std::string windowName = "Camera " + std::to_string(index + 1);
                const fs::path outputPath = recordingDir / ("camera_" + std::to_string(index + 1) + ".avi");

                cv::namedWindow(windowName, cv::WINDOW_NORMAL);
                windowNames.push_back(windowName);
                outputPaths.push_back(outputPath);
                recorders.push_back(std::make_unique<CameraRecorder>(
                    selectedCameras[index],
                    windowNames.back(),
                    outputPaths.back(),
                    settings.fps
                ));
            }

            for (auto& recorder : recorders)
            {
                recorder->Start();
            }

            const auto startTime = std::chrono::steady_clock::now();
            bool stopEarly = false;

            while (!stopEarly)
            {
                const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - startTime
                );
                if (elapsed.count() >= settings.durationSeconds)
                {
                    break;
                }

                for (auto& recorder : recorders)
                {
                    if (auto preview = recorder->GetLatestPreview())
                    {
                        cv::imshow(recorder->GetWindowName(), *preview);
                    }
                }

                const int key = cv::waitKey(kPreviewPollMs);
                if (key == 'q' || key == 'Q' || key == 27)
                {
                    stopEarly = true;
                }
            }

            for (auto& recorder : recorders)
            {
                recorder->Stop();
            }
            for (const auto& windowName : windowNames)
            {
                cv::destroyWindow(windowName);
            }
            cv::waitKey(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            cv::waitKey(1);

            std::cout << "\nSaved recording to " << recordingDir << "\n";
            for (std::size_t index = 0; index < recorders.size(); ++index)
            {
                const RecorderStats stats = recorders[index]->GetStats();
                std::cout << "  Camera " << (index + 1) << ": " << outputPaths[index].filename().string()
                          << " | received " << stats.receivedFrames
                          << " | written " << stats.writtenFrames
                          << " | dropped " << stats.droppedFrames << "\n";
            }
            std::cout << "\n";

            if (!PromptYesNo("Start another recording?", true))
            {
                break;
            }
            std::cout << "\n";
        }

        cv::destroyAllWindows();
        cv::waitKey(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        for (auto& camera : selectedCameras)
        {
            camera->Close();
        }
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
