#pragma once

#include <string>
#include <fstream>
#include <map>
#include <vector>

struct LoggedEntry
{
    uint64_t update_steps;
    uint64_t play_steps;
    float value;
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
    /// @param columns_name The column header names
    /// @param values A value for each column, should be the same size than columns_name
    void Log(const std::vector<std::string>& columns_name, const std::vector<LoggedEntry>& value);

private:
    void Dump() const;

private:
    std::ofstream file;
    bool real_time_logging;

    std::map<std::string, std::vector<LoggedEntry> > logged_data;
};
