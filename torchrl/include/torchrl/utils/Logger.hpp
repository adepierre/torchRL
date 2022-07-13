#pragma once

#include <string>
#include <fstream>
#include <map>
#include <vector>
#include <thread>
#include <mutex>

struct GLFWwindow;

struct LoggedEntry
{
    double update_steps;
    double play_steps;
    double value;
};

class Logger
{
public:
    Logger(const std::string& log_file, const bool real_time_logging_ = true);
    ~Logger();

    /// @brief Set column names
    /// @param columns_name names of all data columns, column names can't be changed in case of real time logging
    void SetColumns(const std::vector<std::string>& columns_name);

    /// @brief Add data values to columns. If real_time_logging, all missing columns will have a blank value
    /// @param columns_name
    /// @param values
    /// @brief Add data values to columns. If real_time_logging, all missing columns will have a blank value for this line
    /// @param update_steps Number of update steps done during this training
    /// @param play_steps Number of play steps done during this training
    /// @param values Values to log at this step
    void Log(const uint64_t update_steps, const uint64_t play_steps,
        const std::map<std::string, float>& values);

private:
    void Dump() const;
    void Plot() const;
    void InternalPlotLoop(GLFWwindow* window) const;

private:
    std::ofstream file;
    bool real_time_logging;

    std::map<std::string, std::vector<LoggedEntry> > logged_data;
    mutable std::mutex data_mutex;

    /// @brief max number of points on a log curve
    static constexpr int MAX_POINTS = 10000;
    std::thread plot_thread;

};
