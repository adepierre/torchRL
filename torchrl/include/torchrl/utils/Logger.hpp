#pragma once

#include <string>
#include <fstream>
#include <map>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>

struct GLFWwindow;

struct LoggedEntry
{
    double play_steps;
    double update_steps;
    double train_time;
    double value;
};

class Logger
{
public:
    /// @brief Logger class to store training progress
    /// @param logfile_path_ Path to a file to store csv data
    /// @param log_in_console_ If true will log data in console too
    /// @param log_curves_ If true will open a window with curves (assuming compiled WITH_IMPLOT)
    Logger(const std::string& logfile_path_, const bool log_in_console_ = true,
    const bool draw_curves_ = true);
    ~Logger();

    /// @brief Add a new line to the csv file output, store data internally in the map 
    /// @param play_steps play steps for this line
    /// @param update_steps update steps for this line
    /// @param train_time time in s since the beginning of training
    /// @param values A map with names/values for each log entry
    void Log(const uint64_t play_steps, const uint64_t update_steps, const float train_time,
        const std::map<std::string, float>& values);

private:
    void Dump();
    void Plot() const; // Defined even without WITH_IMPLOT so .hpp is always the same
    void InternalPlotLoop(GLFWwindow* window) const; // Defined even without WITH_IMPLOT so .hpp is always the same

private:
    std::string logfile_path;
    std::ofstream file;

    bool log_in_console;

    std::map<std::string, std::vector<LoggedEntry> > logged_data;
    mutable std::mutex data_mutex;

    /// @brief max number of points on a log curve
    static constexpr int MAX_POINTS = 10000;
    std::thread plot_thread;
    std::atomic<bool> should_close;

};
