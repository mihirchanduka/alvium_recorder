#include <VmbCPP/VmbCPP.h>
#include <VmbCPP/thirdparty/OpenCV.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
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

constexpr int kBufferCount = 128;
constexpr int kPreviewPollMs = 1;
constexpr int kDefaultDurationSeconds = 30;
constexpr double kDefaultFps = 68.0;
constexpr double kFallbackExposureUs = 3000.0;
constexpr std::size_t kMaxQueuedFrames = 1024;
constexpr std::uint64_t kPeriodicLogFrames = 60;

// Scientific timing constants
constexpr double kUsToS = 1e-6;

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

std::string IsoUtcNow()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm {};
    gmtime_r(&t, &tm);

    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()
    ).count() % 1000000;

    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S")
        << "." << std::setw(6) << std::setfill('0') << micros << "Z";
    return out.str();
}

std::string FormatDouble(double value, int precision = 6)
{
    if (!std::isfinite(value)) return "null";
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

std::string JsonString(const std::string& s) {
    return "\"" + s + "\"";
}

std::string GetCameraLabel(const CameraPtr& camera)
{
    std::string name, id, model;
    camera->GetName(name);
    camera->GetID(id);
    camera->GetModel(model);
    if (name.empty()) return model + " (" + id + ")";
    return name + " (" + id + ")";
}

bool IsAlliedVisionCamera(const CameraPtr& camera)
{
    std::string name, model, id;
    camera->GetName(name);
    camera->GetModel(model);
    camera->GetID(id);
    const std::string combined = name + " " + model + " " + id;
    return combined.find("Allied Vision") != std::string::npos
        || combined.find("ALVIUM") != std::string::npos
        || combined.find("Alvium") != std::string::npos
        || combined.find("DEV_1AB2") != std::string::npos;
}

bool SetEnumFeatureIfAvailable(const CameraPtr& camera, const char* featureName, const char* value)
{
    FeaturePtr feature;
    if (camera->GetFeatureByName(featureName, feature) == VmbErrorSuccess && feature)
    {
        return feature->SetValue(value) == VmbErrorSuccess;
    }
    return false;
}

bool SetFloatFeatureIfAvailable(const CameraPtr& camera, const char* featureName, double value)
{
    FeaturePtr feature;
    if (camera->GetFeatureByName(featureName, feature) == VmbErrorSuccess && feature)
    {
        return feature->SetValue(value) == VmbErrorSuccess;
    }
    return false;
}

bool SetBoolFeatureIfAvailable(const CameraPtr& camera, const char* featureName, bool value)
{
    FeaturePtr feature;
    if (camera->GetFeatureByName(featureName, feature) == VmbErrorSuccess && feature)
    {
        return feature->SetValue(value) == VmbErrorSuccess;
    }
    return false;
}

template <typename ContainerPtr>
std::optional<std::int64_t> ReadIntFeature(const ContainerPtr& container, const char* featureName)
{
    FeaturePtr feature;
    if (container == nullptr || container->GetFeatureByName(featureName, feature) != VmbErrorSuccess || !feature)
        return std::nullopt;

    VmbInt64_t value = 0;
    if (feature->GetValue(value) == VmbErrorSuccess)
        return static_cast<std::int64_t>(value);
    return std::nullopt;
}

template <typename ContainerPtr>
std::optional<double> ReadFloatFeature(const ContainerPtr& container, const char* featureName)
{
    FeaturePtr feature;
    if (container == nullptr || container->GetFeatureByName(featureName, feature) != VmbErrorSuccess || !feature)
        return std::nullopt;

    double value = 0.0;
    if (feature->GetValue(value) == VmbErrorSuccess)
        return value;
    return std::nullopt;
}

template <typename ContainerPtr>
std::optional<std::string> ReadStringFeature(const ContainerPtr& container, const char* featureName)
{
    FeaturePtr feature;
    if (container == nullptr || container->GetFeatureByName(featureName, feature) != VmbErrorSuccess || !feature)
        return std::nullopt;

    std::string value;
    if (feature->GetValue(value) == VmbErrorSuccess)
        return value;
    return std::nullopt;
}

class RecorderFileLogger;

void ConfigureCamera(const CameraPtr& camera, double fps, double exposureUs, RecorderFileLogger& logger)
{
    SetEnumFeatureIfAvailable(camera, "TriggerMode", "Off");
    SetEnumFeatureIfAvailable(camera, "AcquisitionMode", "Continuous");
    SetEnumFeatureIfAvailable(camera, "ExposureAuto", "Off");

    SetBoolFeatureIfAvailable(camera, "AcquisitionFrameRateEnable", true);
    SetFloatFeatureIfAvailable(camera, "AcquisitionFrameRate", fps);
    SetFloatFeatureIfAvailable(camera, "AcquisitionFrameRateAbs", fps);
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

double ReadCurrentExposureUs(const CameraPtr& camera, double fallbackValue)
{
    for (const char* f : {"ExposureTime", "ExposureTimeAbs"})
    {
        if (auto v = ReadFloatFeature(camera, f)) if (*v > 0.0) return *v;
    }
    return fallbackValue;
}

double ReadActualFps(const CameraPtr& camera, double fallbackValue)
{
    for (const char* f : {"ResultingFrameRate", "ResultingFrameRateAbs", "AcquisitionFrameRate", "AcquisitionFrameRateAbs"})
    {
        if (auto v = ReadFloatFeature(camera, f)) if (*v > 0.0) return *v;
    }
    return fallbackValue;
}

fs::path DetectLocalGentlDirectory()
{
    const fs::path localPath = fs::current_path() / "VimbaX_2026-1" / "cti";
    if (fs::exists(localPath / "VimbaUSBTL.cti")) return localPath;
    return {};
}

void EnsureGentlPath()
{
    if (const char* existing = std::getenv("GENICAM_GENTL64_PATH"); existing && *existing) return;
    const fs::path localGentl = DetectLocalGentlDirectory();
    if (!localGentl.empty()) setenv("GENICAM_GENTL64_PATH", localGentl.c_str(), 1);
}

template <typename T>
T PromptValue(const std::string& label, T defaultValue)
{
    for (;;)
    {
        std::cout << label << " [" << defaultValue << "]: " << std::flush;
        std::string line;
        if (!std::getline(std::cin, line)) throw std::runtime_error("Input stream closed.");
        if (line.empty()) return defaultValue;
        std::istringstream parser(line);
        T value {};
        if ((parser >> value) && parser.eof() && value > static_cast<T>(0)) return value;
        std::cout << "Enter a positive number.\n";
    }
}

bool PromptYesNo(const std::string& question, bool defaultYes)
{
    for (;;)
    {
        std::cout << question << (defaultYes ? " [Y/n]: " : " [y/N]: ") << std::flush;
        std::string line;
        if (!std::getline(std::cin, line)) return false;
        if (line.empty()) return defaultYes;
        if (line == "y" || line == "Y") return true;
        if (line == "n" || line == "N") return false;
    }
}

fs::path CreateNextRecordingDirectory(const fs::path& root)
{
    fs::create_directories(root);
    std::regex pattern(R"(recording_(\d+))");
    int maxIndex = 0;
    if (fs::exists(root))
    {
        for (const auto& entry : fs::directory_iterator(root))
        {
            if (!entry.is_directory()) continue;
            std::smatch match;
            const std::string name = entry.path().filename().string();
            if (std::regex_match(name, match, pattern))
                maxIndex = std::max(maxIndex, std::stoi(match[1].str()));
        }
    }
    const fs::path recDir = root / ("recording_" + std::to_string(maxIndex + 1));
    fs::create_directories(recDir);
    return recDir;
}

class RecorderFileLogger
{
public:
    explicit RecorderFileLogger(fs::path path) :
        m_path(std::move(path)),
        m_stream(m_path, std::ios::out | std::ios::trunc)
    {
        if (!m_stream) throw std::runtime_error("Could not open log file: " + m_path.string());
    }

    void Log(const std::string& level, const std::string& message)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stream << IsoUtcNow() << " [" << level << "] " << message << "\n";
        m_stream.flush();
    }

    const fs::path& Path() const { return m_path; }

private:
    fs::path m_path;
    std::ofstream m_stream;
    std::mutex m_mutex;
};

struct RunningIntervalStats
{
    std::uint64_t count = 0;
    double mean = 0.0, m2 = 0.0, min = std::numeric_limits<double>::infinity(), max = 0.0;
    void Add(double value)
    {
        if (!std::isfinite(value)) return;
        ++count;
        const double delta = value - mean;
        mean += delta / static_cast<double>(count);
        m2 += delta * (value - mean);
        min = std::min(min, value); max = std::max(max, value);
    }
    double StdDev() const { return count < 2 ? 0.0 : std::sqrt(m2 / static_cast<double>(count - 1)); }
};

struct CameraTimingSupport
{
    bool chunkModeEnabled = false;
    double timestampTickHz = 0.0;
    std::string timestampTickFeature;
};

struct RecorderStats
{
    std::uint64_t receivedFrames = 0, writtenFrames = 0, invalidFrames = 0, queueDrops = 0, inferredMissingIds = 0;
    std::size_t maxQueueDepth = 0;
    double lastTemperature = 0.0;
    RunningIntervalStats hwIntervals, hostIntervals;
};

struct TimeLatch
{
    std::uint64_t cameraTicks = 0;
    std::chrono::system_clock::time_point systemTime;
    std::chrono::steady_clock::time_point steadyTime;
};

struct FramePacket
{
    cv::Mat image;
    std::uint64_t captureIndex = 0;
    std::optional<std::uint64_t> frameId, frameTimestamp, chunkTimestamp;
    std::optional<double> exposureUs, gainDb;
    std::chrono::system_clock::time_point hostArrivalSystem;
    std::chrono::steady_clock::time_point hostArrivalSteady;
    std::uint64_t inferredMissingBefore = 0;
};

CameraTimingSupport SetupTiming(const CameraPtr& camera, RecorderFileLogger& logger)
{
    CameraTimingSupport s;
    for (const char* f : {"GevTimestampTickFrequency", "TimestampTickFrequency", "DeviceTimestampFrequency", "TimestampFrequency"})
    {
        if (auto v = ReadIntFeature(camera, f)) if (*v > 0) {
            s.timestampTickHz = static_cast<double>(*v);
            s.timestampTickFeature = f;
            logger.Log("INFO", "Timestamp tick frequency: " + std::to_string(*v) + " Hz");
            break;
        }
    }

    FeaturePtr chunkActive;
    if (camera->GetFeatureByName("ChunkModeActive", chunkActive) == VmbErrorSuccess)
    {
        chunkActive->SetValue(false);
        FeaturePtr selector, enable;
        if (camera->GetFeatureByName("ChunkSelector", selector) == VmbErrorSuccess && 
            camera->GetFeatureByName("ChunkEnable", enable) == VmbErrorSuccess)
        {
            for (const char* chunk : {"Timestamp", "ExposureTime", "Gain"})
            {
                if (selector->SetValue(chunk) == VmbErrorSuccess) enable->SetValue(true);
            }
            if (chunkActive->SetValue(true) == VmbErrorSuccess) s.chunkModeEnabled = true;
        }
    }
    return s;
}

class CameraRecorder;

class FrameObserver final : public IFrameObserver
{
public:
    FrameObserver(const CameraPtr& camera, CameraRecorder& recorder) : IFrameObserver(camera), m_recorder(recorder) {}
    void FrameReceived(const FramePtr frame) override;
private:
    CameraRecorder& m_recorder;
};

class CameraRecorder
{
public:
    CameraRecorder(CameraPtr camera, std::string windowName, fs::path out, fs::path csv, fs::path log, fs::path metadata, double fps, CameraTimingSupport timing) :
        m_camera(std::move(camera)), m_windowName(std::move(windowName)), m_outputPath(std::move(out)),
        m_csvPath(std::move(csv)), m_metadataPath(std::move(metadata)), m_targetFps(fps), m_timing(timing), m_logger(std::move(log)),
        m_observer(new FrameObserver(m_camera, *this))
    {
        m_csv.open(m_csvPath, std::ios::out | std::ios::trunc);
        m_csv << "capture_index,frame_id,hw_time_s,hw_dt_s,exposure_mid_s,system_time_s,steady_time_s,host_dt_s,exposure_us,gain_db,missing_before\n";
    }

    void Start()
    {
        m_stopRequested = false;
        m_worker = std::thread(&CameraRecorder::WriterLoop, this);
        m_startLatch = LatchTime();
        ThrowIfError("Start acquisition", m_camera->StartContinuousImageAcquisition(kBufferCount, m_observer));
    }

    TimeLatch LatchTime()
    {
        TimeLatch l;
        FeaturePtr latch;
        if (m_camera->GetFeatureByName("TimestampLatch", latch) == VmbErrorSuccess)
        {
            latch->RunCommand();
            if (auto v = ReadIntFeature(m_camera, "TimestampLatchValue")) l.cameraTicks = *v;
        }
        else if (auto v = ReadIntFeature(m_camera, "TimestampValue")) l.cameraTicks = *v;
        l.systemTime = std::chrono::system_clock::now();
        l.steadyTime = std::chrono::steady_clock::now();
        return l;
    }

    void Stop()
    {
        if (!m_stopRequested.exchange(true))
        {
            m_frameCondition.notify_all();
            m_camera->StopContinuousImageAcquisition();
        }
        if (m_worker.joinable()) m_worker.join();
        m_endLatch = LatchTime();
        if (m_writer.isOpened()) m_writer.release();
        WriteMetadata();
        LogSummary();
    }

    ~CameraRecorder() { try { Stop(); } catch(...) {} }

    void OnFrame(const FramePtr& frame)
    {
        FramePacket p;
        p.hostArrivalSystem = std::chrono::system_clock::now();
        p.hostArrivalSteady = std::chrono::steady_clock::now();
        p.captureIndex = ++m_captureSequence;
        
        VmbUint64_t fid = 0, ts = 0;
        if (frame->GetFrameID(fid) == VmbErrorSuccess) p.frameId = fid;
        if (frame->GetTimestamp(ts) == VmbErrorSuccess) p.frameTimestamp = ts;

        if (m_timing.chunkModeEnabled)
        {
            frame->AccessChunkData([&p](ChunkFeatureContainerPtr& c) {
                if (auto v = ReadIntFeature(c, "ChunkTimestamp")) p.chunkTimestamp = *v;
                if (auto v = ReadFloatFeature(c, "ChunkExposureTime")) p.exposureUs = *v;
                if (auto v = ReadFloatFeature(c, "ChunkGain")) p.gainDb = *v;
                return VmbErrorSuccess;
            });
        }
        if (!p.exposureUs) p.exposureUs = ReadCurrentExposureUs(m_camera, kFallbackExposureUs);

        cv::Mat image;
        if (VmbFrameToMat(frame, image) == VmbErrorSuccess && !image.empty())
        {
            p.image = image.clone();
            std::lock_guard<std::mutex> lock(m_mutex);
            ++m_stats.receivedFrames;
            if (p.frameId && m_lastFrameId && *p.frameId > *m_lastFrameId + 1)
            {
                p.inferredMissingBefore = *p.frameId - *m_lastFrameId - 1;
                m_stats.inferredMissingIds += p.inferredMissingBefore;
            }
            if (p.frameId) m_lastFrameId = p.frameId;

            auto ticks = p.chunkTimestamp ? p.chunkTimestamp : p.frameTimestamp;
            if (ticks && m_lastHwTicks && *ticks > *m_lastHwTicks && m_timing.timestampTickHz > 0)
                m_stats.hwIntervals.Add(static_cast<double>(*ticks - *m_lastHwTicks) / m_timing.timestampTickHz);
            if (ticks) m_lastHwTicks = ticks;

            if (m_lastHostSteady) m_stats.hostIntervals.Add(std::chrono::duration<double>(p.hostArrivalSteady - *m_lastHostSteady).count());
            m_lastHostSteady = p.hostArrivalSteady;

            if (m_queue.size() < kMaxQueuedFrames)
            {
                m_queue.push_back(std::move(p));
                m_stats.maxQueueDepth = std::max(m_stats.maxQueueDepth, m_queue.size());
            }
            else ++m_stats.queueDrops;
        }
        else { std::lock_guard<std::mutex> lock(m_mutex); ++m_stats.invalidFrames; }

        if (!m_stopRequested.load()) m_camera->QueueFrame(frame);
        m_frameCondition.notify_one();
    }

    void HandleInvalid(const FramePtr& frame, VmbFrameStatusType s)
    {
        m_logger.Log("WARN", "Invalid frame status: " + std::to_string(s));
        { std::lock_guard<std::mutex> lock(m_mutex); ++m_stats.invalidFrames; }
        if (!m_stopRequested.load()) m_camera->QueueFrame(frame);
    }

    std::optional<cv::Mat> GetPreview()
    {
        std::lock_guard<std::mutex> lock(m_previewMutex);
        if (m_latestPreview.empty()) return std::nullopt;
        return m_latestPreview.clone();
    }

    RecorderStats GetStats() const { std::lock_guard<std::mutex> lock(m_mutex); return m_stats; }
    const std::string& GetName() const { return m_windowName; }

private:
    void WriterLoop()
    {
        for (;;)
        {
            FramePacket p;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_frameCondition.wait(lock, [&] { return m_stopRequested.load() || !m_queue.empty(); });
                if (m_queue.empty() && m_stopRequested.load()) break;
                if (m_queue.empty()) continue;
                p = std::move(m_queue.front()); m_queue.pop_front();
            }

            if (!m_writer.isOpened())
            {
                const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                if (!m_writer.open(m_outputPath.string(), fourcc, m_targetFps, p.image.size(), p.image.channels() > 1))
                    throw std::runtime_error("Could not open video writer: " + m_outputPath.string());
                m_logger.Log("INFO", "Video writer opened at authoritative FPS=" + FormatDouble(m_targetFps));
            }
            
            m_writer.write(p.image);
            { std::lock_guard<std::mutex> lock(m_mutex); ++m_stats.writtenFrames; }
            { std::lock_guard<std::mutex> l(m_previewMutex); m_latestPreview = p.image; }

            WriteCsvRow(p);
            
            if (m_stats.writtenFrames % kPeriodicLogFrames == 0) {
                if (auto temp = ReadFloatFeature(m_camera, "DeviceTemperature")) {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_stats.lastTemperature = *temp;
                    if (*temp > 65.0) m_logger.Log("WARN", "High temperature: " + FormatDouble(*temp, 1) + " C");
                }
                LogProgress(p);
            }
        }
    }

    void WriteCsvRow(const FramePacket& p)
    {
        auto ticks = p.chunkTimestamp ? p.chunkTimestamp : p.frameTimestamp;
        double hwS = 0, hwDt = 0, expMid = 0;
        if (ticks && m_timing.timestampTickHz > 0)
        {
            hwS = static_cast<double>(*ticks - m_startLatch.cameraTicks) / m_timing.timestampTickHz;
            if (m_prevHwTicks) hwDt = static_cast<double>(*ticks - *m_prevHwTicks) / m_timing.timestampTickHz;
            expMid = hwS - (p.exposureUs.value_or(0.0) * 0.5 * kUsToS);
            m_prevHwTicks = ticks;
        }
        double sysS = std::chrono::duration<double>(p.hostArrivalSystem - m_startLatch.systemTime).count();
        double steadyS = std::chrono::duration<double>(p.hostArrivalSteady - m_startLatch.steadyTime).count();
        double hDt = m_prevHostSteady ? std::chrono::duration<double>(p.hostArrivalSteady - *m_prevHostSteady).count() : 0;
        m_prevHostSteady = p.hostArrivalSteady;

        m_csv << p.captureIndex << "," << (p.frameId ? std::to_string(*p.frameId) : "") << ","
              << FormatDouble(hwS, 9) << "," << FormatDouble(hwDt, 9) << "," << FormatDouble(expMid, 9) << ","
              << FormatDouble(sysS, 9) << "," << FormatDouble(steadyS, 9) << "," << FormatDouble(hDt, 9) << ","
              << FormatDouble(p.exposureUs.value_or(0), 3) << "," << FormatDouble(p.gainDb.value_or(0), 3) << ","
              << p.inferredMissingBefore << "\n";
    }

    void LogProgress(const FramePacket& p)
    {
        auto s = GetStats();
        double elapsed = std::chrono::duration<double>(p.hostArrivalSteady - m_startLatch.steadyTime).count();
        double curFps = elapsed > 0 ? static_cast<double>(s.writtenFrames) / elapsed : 0;
        std::size_t qLen = (s.receivedFrames - s.writtenFrames);
        double qPressure = (static_cast<double>(qLen) / kMaxQueuedFrames) * 100.0;
        
        std::cout << "\r" << m_windowName << ": "
                  << "frame=" << std::setw(6) << s.writtenFrames << " "
                  << "fps=" << std::fixed << std::setprecision(1) << std::setw(5) << curFps << " "
                  << "buf=" << std::setw(3) << static_cast<int>(qPressure) << "% "
                  << "temp=" << std::setw(4) << std::fixed << std::setprecision(1) << s.lastTemperature << "C "
                  << "drops=" << s.queueDrops << std::flush;
    }

    void WriteMetadata()
    {
        std::ofstream out(m_metadataPath);
        auto s = GetStats();
        std::string model, serial, version;
        m_camera->GetModel(model);
        m_camera->GetID(serial);
        auto firmware = ReadStringFeature(m_camera, "DeviceFirmwareVersion");
        auto width = ReadIntFeature(m_camera, "Width");
        auto height = ReadIntFeature(m_camera, "Height");

        out << "{\n"
            << "  \"camera\": {\n"
            << "    \"model\": " << JsonString(model) << ",\n"
            << "    \"serial\": " << JsonString(serial) << ",\n"
            << "    \"firmware\": " << JsonString(firmware.value_or("unknown")) << "\n"
            << "  },\n"
            << "  \"settings\": {\n"
            << "    \"target_fps\": " << FormatDouble(m_targetFps, 3) << ",\n"
            << "    \"width\": " << (width ? std::to_string(*width) : "null") << ",\n"
            << "    \"height\": " << (height ? std::to_string(*height) : "null") << ",\n"
            << "    \"pixel_format\": " << JsonString(ReadStringFeature(m_camera, "PixelFormat").value_or("unknown")) << "\n"
            << "  },\n"
            << "  \"statistics\": {\n"
            << "    \"written_frames\": " << s.writtenFrames << ",\n"
            << "    \"dropped_frames\": " << s.queueDrops << ",\n"
            << "    \"missing_frame_ids\": " << s.inferredMissingIds << ",\n"
            << "    \"max_queue_depth\": " << s.maxQueueDepth << ",\n"
            << "    \"final_temperature\": " << FormatDouble(s.lastTemperature, 1) << "\n"
            << "  }\n"
            << "}\n";
    }

    void LogSummary()
    {
        auto s = GetStats();
        m_logger.Log("INFO", "Summary: written=" + std::to_string(s.writtenFrames) + " drops=" + std::to_string(s.queueDrops) + " temp=" + FormatDouble(s.lastTemperature, 1));
    }

    CameraPtr m_camera; std::string m_windowName; fs::path m_outputPath, m_csvPath, m_metadataPath; double m_targetFps;
    CameraTimingSupport m_timing; RecorderFileLogger m_logger; IFrameObserverPtr m_observer;
    std::atomic<bool> m_stopRequested {false}; std::atomic<std::uint64_t> m_captureSequence {0};
    TimeLatch m_startLatch, m_endLatch; mutable std::mutex m_mutex; std::condition_variable m_frameCondition;
    std::deque<FramePacket> m_queue; std::thread m_worker; std::mutex m_previewMutex; cv::Mat m_latestPreview;
    cv::VideoWriter m_writer; RecorderStats m_stats; std::ofstream m_csv;
    std::optional<std::uint64_t> m_lastFrameId, m_lastHwTicks, m_prevHwTicks;
    std::optional<std::chrono::steady_clock::time_point> m_lastHostSteady, m_prevHostSteady;
};

void FrameObserver::FrameReceived(const FramePtr frame)
{
    VmbFrameStatusType s;
    if (frame->GetReceiveStatus(s) == VmbErrorSuccess && s == VmbFrameStatusComplete) m_recorder.OnFrame(frame);
    else m_recorder.HandleInvalid(frame, s);
}

std::vector<std::size_t> PromptSelection(const std::vector<CameraPtr>& cameras)
{
    for (;;)
    {
        std::cout << "\nAvailable Cameras:\n";
        for (std::size_t i = 0; i < cameras.size(); ++i) std::cout << "  " << (i + 1) << ". " << GetCameraLabel(cameras[i]) << "\n";
        std::cout << "\nSelect cameras (e.g., 1,2 or 1-3 or all) [all]: " << std::flush;
        std::string line; if (!std::getline(std::cin, line)) throw std::runtime_error("Input stream closed.");
        if (line.empty() || line == "all" || line == "ALL") {
            std::vector<std::size_t> v(cameras.size()); std::iota(v.begin(), v.end(), 0); return v;
        }
        std::vector<std::size_t> res; std::unordered_set<std::size_t> seen;
        std::stringstream ss(line); std::string t; bool ok = true;
        while (std::getline(ss, t, ',')) {
            t.erase(0, t.find_first_not_of(" ")); t.erase(t.find_last_not_of(" ") + 1);
            if (t.find('-') != std::string::npos) {
                int a, b; if (sscanf(t.c_str(), "%d-%d", &a, &b) == 2 && a > 0 && b >= a) {
                    for (int j = a; j <= b; ++j) if (static_cast<std::size_t>(j-1) < cameras.size()) {
                        if (seen.insert(j-1).second) res.push_back(j-1);
                    } else ok = false;
                } else ok = false;
            } else {
                int a; if (sscanf(t.c_str(), "%d", &a) == 1 && a > 0 && static_cast<std::size_t>(a-1) < cameras.size()) {
                    if (seen.insert(a-1).second) res.push_back(a-1);
                } else ok = false;
            }
        }
        if (ok && !res.empty()) return res;
        std::cout << "Invalid selection.\n";
    }
}

} // namespace

int main()
{
    try {
        EnsureGentlPath(); cv::setNumThreads(1);
        std::cout << "=====================================\n        Alvium Recorder\n=====================================\n";
        VmbSystem& sys = VmbSystem::GetInstance(); ThrowIfError("Vmb Startup", sys.Startup());
        CameraPtrVector all; ThrowIfError("Get cameras", sys.GetCameras(all));
        if (all.empty()) throw std::runtime_error("No cameras found.");
        
        auto selectedIds = PromptSelection(std::vector<CameraPtr>(all.begin(), all.end()));
        std::vector<CameraPtr> cameras;
        for (auto id : selectedIds) {
            ThrowIfError("Open camera", all[id]->Open(VmbAccessModeFull));
            cameras.push_back(all[id]);
        }

        for (;;) {
            double targetFps = PromptValue<double>("Target FPS", kDefaultFps);
            int duration = PromptValue<int>("Duration (s)", kDefaultDurationSeconds);
            double exp = PromptValue<double>("Exposure (us)", ReadCurrentExposureUs(cameras[0], kFallbackExposureUs));

            fs::path recDir = CreateNextRecordingDirectory(fs::current_path() / "recordings");
            RecorderFileLogger sessionLog(recDir / "session.log");
            sessionLog.Log("INFO", "Session started. Dir: " + recDir.string());

            std::vector<std::unique_ptr<CameraRecorder>> recorders;
            for (size_t i = 0; i < cameras.size(); ++i) {
                auto& cam = cameras[i];
                ConfigureCamera(cam, targetFps, exp, sessionLog);
                auto timing = SetupTiming(cam, sessionLog);
                double actualFps = ReadActualFps(cam, targetFps);
                
                std::string name = "Cam" + std::to_string(i+1);
                recorders.push_back(std::make_unique<CameraRecorder>(
                    cam, name, recDir / (name + ".avi"), recDir / (name + "_times.csv"), 
                    recDir / (name + ".log"), recDir / (name + "_meta.json"), actualFps, timing));
            }

            std::cout << "\nRecording to " << recDir << ". Press 'q' in console to stop.\n\n";
            for (auto& r : recorders) r->Start();
            auto start = std::chrono::steady_clock::now();
            while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() < duration) {
                for (auto& r : recorders) if (auto p = r->GetPreview()) cv::imshow(r->GetName(), *p);
                int k = cv::waitKey(kPreviewPollMs); if (k == 'q' || k == 'Q' || k == 27) break;
            }

            for (auto& r : recorders) r->Stop();
            cv::destroyAllWindows();
            std::cout << "\n\nFinished. Stats recorded in logs.\n";
            if (!PromptYesNo("Another recording?", true)) break;
        }
        sys.Shutdown(); return 0;
    } catch (const std::exception& e) { std::cerr << "\nERROR: " << e.what() << "\n"; return 1; }
}
