#include <VmbCPP/VmbCPP.h>
#include <VmbCPP/thirdparty/OpenCV.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace fs = std::filesystem;
using namespace VmbCPP;

namespace {

// ── camera helpers (mirrors alvium_recorder) ────────────────────────────────

void EnsureGentlPath()
{
    if (const char* e = std::getenv("GENICAM_GENTL64_PATH"); e && *e) return;
    const fs::path local = fs::current_path() / "VimbaX_2026-1" / "cti";
    if (fs::exists(local / "VimbaUSBTL.cti"))
        setenv("GENICAM_GENTL64_PATH", local.c_str(), 1);
}

template <typename C>
std::optional<double> ReadFloat(const C& c, const char* name)
{
    FeaturePtr f;
    if (c->GetFeatureByName(name, f) != VmbErrorSuccess || !f) return std::nullopt;
    double v = 0.0;
    return f->GetValue(v) == VmbErrorSuccess ? std::optional{v} : std::nullopt;
}

void SetFloat(const CameraPtr& cam, const char* name, double v)
{
    FeaturePtr f;
    if (cam->GetFeatureByName(name, f) == VmbErrorSuccess && f) f->SetValue(v);
}

void SetEnum(const CameraPtr& cam, const char* name, const char* v)
{
    FeaturePtr f;
    if (cam->GetFeatureByName(name, f) == VmbErrorSuccess && f) f->SetValue(v);
}

void SetBool(const CameraPtr& cam, const char* name, bool v)
{
    FeaturePtr f;
    if (cam->GetFeatureByName(name, f) == VmbErrorSuccess && f) f->SetValue(v);
}

double ReadExposureUs(const CameraPtr& cam)
{
    for (const char* n : {"ExposureTime", "ExposureTimeAbs"})
        if (auto v = ReadFloat(cam, n)) if (*v > 0) return *v;
    return 5000.0;
}

// ── live frame observer ──────────────────────────────────────────────────────

class LiveObserver : public IFrameObserver
{
public:
    explicit LiveObserver(const CameraPtr& cam) : IFrameObserver(cam), m_cam(cam) {}

    void Stop() { m_stopped.store(true); }

    void FrameReceived(const FramePtr frame) override
    {
        VmbFrameStatusType status;
        if (frame->GetReceiveStatus(status) == VmbErrorSuccess && status == VmbFrameStatusComplete)
        {
            cv::Mat img;
            if (VmbFrameToMat(frame, img) == VmbErrorSuccess && !img.empty())
            {
                std::lock_guard<std::mutex> lk(m_mutex);
                m_latest = img.clone();
                m_frameCount++;
            }
        }
        // Do not re-queue after stop — otherwise StopContinuousImageAcquisition deadlocks
        // waiting for callbacks that keep re-queuing frames indefinitely.
        if (!m_stopped.load())
            m_cam->QueueFrame(frame);
    }

    std::optional<cv::Mat> Take()
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        if (m_latest.empty()) return std::nullopt;
        return m_latest;
    }

    std::uint64_t FrameCount()
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        return m_frameCount;
    }

private:
    CameraPtr m_cam;
    std::atomic<bool> m_stopped {false};
    std::mutex m_mutex;
    cv::Mat m_latest;
    std::uint64_t m_frameCount {0};
};

// ── helpers ──────────────────────────────────────────────────────────────────

void PrintHelp()
{
    std::cout <<
        "\n=== Calibration Viewer ===\n"
        "  +/-   exposure +/- 500 us\n"
        "  g/G   gain     +/- 1 dB\n"
        "  f/F   FPS      +/- 5\n"
        "  q / ESC  quit\n"
        "==========================\n\n";
}

std::string GetLabel(const CameraPtr& cam)
{
    std::string model, id;
    cam->GetModel(model);
    cam->GetID(id);
    return model + " (" + id + ")";
}

} // namespace

int main()
{
    try {
        EnsureGentlPath();

        VmbSystem& sys = VmbSystem::GetInstance();
        if (sys.Startup() != VmbErrorSuccess)
            throw std::runtime_error("VmbSystem startup failed");

        CameraPtrVector all;
        sys.GetCameras(all);
        if (all.empty()) throw std::runtime_error("No cameras found");

        // Pick camera
        CameraPtr cam;
        if (all.size() == 1)
        {
            cam = all[0];
        }
        else
        {
            std::cout << "Cameras:\n";
            for (std::size_t i = 0; i < all.size(); ++i)
                std::cout << "  " << (i + 1) << ". " << GetLabel(all[i]) << "\n";
            std::cout << "Select [1]: ";
            std::string line; std::getline(std::cin, line);
            int idx = line.empty() ? 0 : std::stoi(line) - 1;
            if (idx < 0 || idx >= (int)all.size()) throw std::runtime_error("Bad selection");
            cam = all[idx];
        }

        if (cam->Open(VmbAccessModeFull) != VmbErrorSuccess)
            throw std::runtime_error("Could not open camera");

        // Configure for streaming
        SetEnum(cam, "TriggerMode",  "Off");
        SetEnum(cam, "AcquisitionMode", "Continuous");
        SetEnum(cam, "ExposureAuto", "Off");
        SetBool(cam, "AcquisitionFrameRateEnable", true);

        double expUs = ReadExposureUs(cam);
        double gainDb = 0.0;
        if (auto g = ReadFloat(cam, "Gain")) gainDb = *g;

        std::cout << "Target FPS [68]: ";
        std::string fpsLine; std::getline(std::cin, fpsLine);
        double fps = fpsLine.empty() ? 68.0 : std::stod(fpsLine);

        SetFloat(cam, "AcquisitionFrameRate",    fps);
        SetFloat(cam, "AcquisitionFrameRateAbs", fps);
        SetFloat(cam, "ExposureTime",    expUs);
        SetFloat(cam, "ExposureTimeAbs", expUs);

        // Pixel format: prefer Mono8, fall back to Bgr8
        {
            FeaturePtr pf;
            if (cam->GetFeatureByName("PixelFormat", pf) == VmbErrorSuccess && pf)
                if (pf->SetValue("Mono8") != VmbErrorSuccess) pf->SetValue("Bgr8");
        }

        auto observer = std::make_shared<LiveObserver>(cam);
        if (cam->StartContinuousImageAcquisition(8, IFrameObserverPtr(observer)) != VmbErrorSuccess)
            throw std::runtime_error("Could not start acquisition");

        PrintHelp();
        std::cout << "Streaming " << GetLabel(cam) << " — press q to quit\n\n";

        const std::string win = "Calibration Viewer";
        cv::namedWindow(win, cv::WINDOW_NORMAL);

        auto sessionStart   = std::chrono::steady_clock::now();
        auto lastFpsReport  = sessionStart;
        std::uint64_t lastCount = 0;

        for (;;)
        {
            int k = cv::waitKey(20);
            if (k == 'q' || k == 'Q' || k == 27) break;

            bool changed = false;
            if (k == '+' || k == '=') { expUs  += 500;  changed = true; }
            if (k == '-' || k == '_') { expUs   = std::max(100.0, expUs - 500); changed = true; }
            if (k == 'g')             { gainDb  += 1.0;  changed = true; }
            if (k == 'G')             { gainDb   = std::max(0.0, gainDb - 1.0); changed = true; }
            if (k == 'f')             { fps      += 5.0;  changed = true; }
            if (k == 'F')             { fps       = std::max(5.0, fps - 5.0);   changed = true; }

            if (changed)
            {
                SetFloat(cam, "ExposureTime",            expUs);
                SetFloat(cam, "ExposureTimeAbs",         expUs);
                SetFloat(cam, "Gain",                    gainDb);
                SetFloat(cam, "AcquisitionFrameRate",    fps);
                SetFloat(cam, "AcquisitionFrameRateAbs", fps);
                std::cout << "exp=" << (int)expUs << "us  gain=" << gainDb << "dB  fps=" << fps << "\n";
            }

            if (auto frame = observer->Take())
            {
                // Print live FPS every 2 seconds
                auto now = std::chrono::steady_clock::now();
                double sinceReport = std::chrono::duration<double>(now - lastFpsReport).count();
                if (sinceReport >= 2.0)
                {
                    std::uint64_t count = observer->FrameCount();
                    double liveFps = (count - lastCount) / sinceReport;
                    double totalSec = std::chrono::duration<double>(now - sessionStart).count();
                    int m = static_cast<int>(totalSec) / 60;
                    int s = static_cast<int>(totalSec) % 60;
                    std::cout << "\r"
                              << "elapsed=" << std::setfill('0') << std::setw(2) << m << ":"
                                           << std::setw(2) << s << std::setfill(' ')
                              << "  fps=" << std::fixed << std::setprecision(1) << std::setw(5) << liveFps
                              << "/" << std::setprecision(0) << fps
                              << "  exp=" << (int)expUs << "us"
                              << "  gain=" << std::setprecision(1) << gainDb << "dB   "
                              << std::flush;
                    lastFpsReport = now;
                    lastCount = count;
                }

                cv::imshow(win, *frame);
            }
        }

        cv::destroyAllWindows();
        observer->Stop();
        cam->StopContinuousImageAcquisition();
        cam->Close();
        sys.Shutdown();
        std::cout << "\nDone.\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
