#include "torchrl/utils/Logger.hpp"

Logger::Logger(const std::string& log_file, const bool real_time_logging_) : file(std::ofstream(log_file, std::ios::out))
{
    real_time_logging = real_time_logging_;
}

Logger::~Logger()
{
    if (!real_time_logging)
    {
        Dump();
    }
    file.close();
}

void Logger::SetColumns(const std::vector<std::string>& columns_name)
{
    logged_data.clear();
    for (auto& s : columns_name)
    {
        logged_data[s];
    }
    if (real_time_logging)
    {
        for (auto& s : logged_data)
        {
            file << "\t\t" << s.first << "\t";
        }
        file << std::endl;
    }
}

void Logger::Log(const std::vector<std::string>& columns_name, const std::vector<LoggedEntry>& value)
{
    //assert(columns_name.size() == value.size());

    for (int i = 0; i < columns_name.size(); ++i)
    {
        // If not real time, we can add columns during the logging process if we want to
        if (!real_time_logging)
        {
            logged_data[columns_name[i]];
        }

        // The column is assumed to exist
        logged_data.at(columns_name[i]).push_back(value[i]);
    }

    // Dump the new line to the log file
    if (real_time_logging)
    {
        for (auto& s : logged_data)
        {
            if (std::find(columns_name.begin(), columns_name.end(), s.first) != columns_name.end())
            {
                file << s.second.back().update_steps << "\t" << s.second.back().play_steps << "\t" << s.second.back().value << "\t";
            }
            else
            {
                file << "\t\t\t";
            }
        }
        file << std::endl;
    }
}

void Logger::Dump() const
{

}
