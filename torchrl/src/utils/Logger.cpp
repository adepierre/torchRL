#include "torchrl/utils/Logger.hpp"

#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <sstream>

#ifdef WITH_IMPLOT
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <implot.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#else
struct GLFWwindow {};
#endif


Logger::Logger(const std::string& logfile_path_, const bool log_in_console_, const bool draw_curves_) :
    logfile_path(logfile_path_), file(std::ofstream(logfile_path_, std::ios::out)), log_in_console(log_in_console_)
{
#ifdef WITH_IMPLOT
    if (draw_curves_)
    {
        plot_thread = std::move(std::thread(&Logger::Plot, this));
    }
#endif
    should_close = false;
}

Logger::~Logger()
{
    should_close = true;
    file.close();
    if (plot_thread.joinable())
    {
        plot_thread.join();
    }
}

void Logger::Log(const uint64_t play_steps, const uint64_t update_steps, const float train_time,
    const std::map<std::string, float>& values)
{
    // Console log
    if (log_in_console)
    {
        uint64_t max_col_size = 12;
        for (const auto& [key, value] : values)
        {
            max_col_size = std::max(max_col_size, key.size());
        }

        std::stringstream s;
        s << std::left << std::setw(max_col_size) << "Play steps" << ": " << std::fixed << std::setw(10) << play_steps << "\n"
            << std::left << std::setw(max_col_size) << "Update steps" << ": " << std::fixed << std::setw(10) << update_steps << "\n"
            << std::left << std::setw(max_col_size) << "Train time" << ": " << std::fixed << std::setw(10) << std::setprecision(3) << train_time << "\n";

        for (const auto& [key, value] : values)
        {
            s
                << std::left << std::setw(max_col_size) << key << ": "
                << std::left << std::fixed << std::setprecision(3) << std::setw(10) << value << "\n";
        }
        const std::string logged = s.str();
        std::cout << logged << std::endl;
    }

    // File log
    if (file.is_open())
    {
        std::lock_guard<std::mutex> data_lock(data_mutex);

        bool new_columns = false;
        for (const auto& s : values)
        {
            if (logged_data.find(s.first) == logged_data.end())
            {
                new_columns = true;
                logged_data[s.first];
            }

            // Add the new value to the column
            logged_data.at(s.first).push_back({ static_cast<double>(play_steps), static_cast<double>(update_steps), static_cast<double>(train_time), static_cast<double>(s.second) });
        }

        // If a new column has been added, we need to rewrite the whole file
        if (new_columns)
        {
            file.close();
            file = std::ofstream(logfile_path, std::ios::out);
            Dump();
        }
        // Else we just add the new line
        else
        {
            file << play_steps << "\t" << update_steps << "\t" << train_time << "\t";
            for (auto& s : logged_data)
            {
                const auto it = values.find(s.first);

                // If we have a value for this column
                if (it != values.end())
                {
                    file << it->second << "\t";
                }
                // Else blank entry
                else
                {
                    file << "\t";
                }
            }
            file << std::endl;
        }
    }
}

void Logger::Dump()
{
    file << "Play steps\tUpdate steps\tTrain time\t";
    // Header lines
    for (const auto& [key, value] : logged_data)
    {
        file << key << "\t";
    }
    file << std::endl;

    std::unordered_map<std::string, uint64_t> column_index;
    for (const auto& [key, value] : logged_data)
    {
        column_index[key] = 0;
    }

    uint64_t index = 0;
    while (true)
    {
        bool has_value = false;
        uint64_t min_play_steps = std::numeric_limits<uint64_t>::max();
        uint64_t min_update_steps = 0;
        double min_train_time = 0.0;
        // Find the lowest play_step value for the next line
        for (const auto& [key, value] : logged_data)
        {
            if (column_index[key] < value.size())
            {
                has_value = true;
                if (static_cast<uint64_t>(value[column_index[key]].play_steps) < min_play_steps)
                {
                    min_play_steps = static_cast<uint64_t>(value[column_index[key]].play_steps);
                    min_update_steps = static_cast<uint64_t>(value[column_index[key]].update_steps);
                    min_train_time = value[column_index[key]].train_time;
                }
            }
        }

        if (!has_value)
        {
            break;
        }

        file
            << static_cast<uint64_t>(min_play_steps) << "\t"
            << static_cast<uint64_t>(min_update_steps) << "\t"
            << min_train_time << "\t";

        for (const auto& [key, value] : logged_data)
        {
            if (column_index[key] < value.size() && value[column_index[key]].play_steps == min_play_steps)
            {
                file << value[column_index[key]].value << "\t";
                column_index[key] += 1;
            }
            else // Blank entry
            {
                file << "\t";
            }
        }
        file << std::endl;
    }
}

void Logger::Plot() const
{
#ifdef WITH_IMPLOT
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "TrainingLogs", NULL, NULL);
    if (window == NULL)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return;
    }

    // imgui: setup context
    // ---------------------------------------
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    // Style
    ImGui::StyleColorsDark();

    // Setup platform/renderer
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    InternalPlotLoop(window);

    // ImGui cleaning
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
#endif
}

void Logger::InternalPlotLoop(GLFWwindow* window) const
{
#ifdef WITH_IMPLOT
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    bool fit_data = true;
    std::unordered_map<std::string, bool> displayed_values;

    bool display_popup = true;

    while (glfwWindowShouldClose(window) == 0)
    {
        // clear the window
        glClear(GL_COLOR_BUFFER_BIT);

        // Init imgui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImVec2(width, height));

            if (should_close && display_popup)
            {
                ImGui::OpenPopup("Done");
                ImGui::SetNextWindowSize(ImVec2(0, 0));
                ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
                if (ImGui::BeginPopupModal("Done", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
                {
                    ImGui::Text("Training is done.\nYou can still interact with the plot window, close it when you're done.\nMake sure you take a screen shot if you want to save it !");
                    display_popup = !ImGui::Button("Close");
                    if (!display_popup)
                    {
                        ImGui::CloseCurrentPopup();
                    }
                    ImGui::EndPopup();
                }
            }


            ImGui::Begin("Logged values", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
            ImGui::Checkbox("Auto-Fit", &fit_data);

            std::lock_guard<std::mutex> data_lock(data_mutex);

            // Check which curves should be displayed
            int num_displayed = 0;
            int num_seen = 0;
            for (const auto& kv : logged_data)
            {
                if (displayed_values.find(kv.first) == displayed_values.end())
                {
                    displayed_values[kv.first] = true;
                }
                ImGui::Checkbox(kv.first.c_str(), &displayed_values[kv.first]);
                if (num_seen < logged_data.size() - 1)
                {
                    ImGui::SameLine();
                }
                if (displayed_values[kv.first])
                {
                    num_displayed += 1;
                }
                num_seen += 1;
            }

            // Find sub plot layout
            const int cols = std::ceil(std::sqrtf(static_cast<float>(num_displayed)));
            const int rows = std::ceil(static_cast<float>(num_displayed) / cols);

            if (cols > 0 && rows > 0 && ImPlot::BeginSubplots("Logged data", rows, cols, ImVec2(-1, -1), ImPlotSubplotFlags_LinkAllX))
            {
                // For each curve
                for (const auto& [key, value] : logged_data)
                {
                    if (displayed_values[key] && value.size() > 0 && ImPlot::BeginPlot(key.c_str(), ImVec2(), ImPlotFlags_NoMouseText | ImPlotFlags_AntiAliased))
                    {
                        ImPlot::SetupAxes("Play steps", "", fit_data ? ImPlotAxisFlags_AutoFit : ImPlotAxisFlags_None, fit_data ? ImPlotAxisFlags_AutoFit : ImPlotAxisFlags_None);

                        // Make sure we don't have more than MAX_POINTS per curve
                        int stride_factor = 1;
                        if (value.size() > MAX_POINTS)
                        {
                            stride_factor = static_cast<int>(std::ceil(static_cast<float>(value.size()) / MAX_POINTS));
                        }

                        // Plot
                        ImPlot::PlotLine("", &value[0].play_steps, &value[0].value, static_cast<int>(value.size() / stride_factor), 0, stride_factor * sizeof(LoggedEntry));

                        // Tooltip
                        if (ImPlot::IsPlotHovered())
                        {
                            ImPlotPoint mouse = ImPlot::GetPlotMousePos();

                            // Find the lowest value higher than mouse.x
                            const auto lowest = std::lower_bound(value.begin(), value.end(), mouse.x,
                                [](const LoggedEntry& entry, double val)
                                {
                                    return entry.play_steps < val;
                                });

                            // If we have one, display tooltip
                            if (lowest != value.end())
                            {
                                ImGui::BeginTooltip();
                                ImGui::Text("Play steps  : %d", static_cast<int>(lowest->play_steps));
                                ImGui::Text("Update steps: %d", static_cast<int>(lowest->update_steps));
                                ImGui::Text("Train time  : %.2f", lowest->train_time);
                                ImGui::Text("Value       : %.2f", lowest->value);
                                ImGui::EndTooltip();

                                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, -1.0f, ImVec4(1, 0.43f, 0.26f, 1));
                                ImPlot::PlotScatter("", &lowest->play_steps, &lowest->value, 1, 0, sizeof(LoggedEntry));
                            }
                        }
                        ImPlot::EndPlot();
                    }
                }
                ImPlot::EndSubplots();
            }
            ImGui::End();
        }

        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // swap buffer
        glfwSwapBuffers(window);

        // process user events
        glfwPollEvents();
    }
#endif
}
