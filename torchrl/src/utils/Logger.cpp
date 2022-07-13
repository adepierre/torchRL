#include "torchrl/utils/Logger.hpp"

#include <iostream>
#include <unordered_map>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <implot.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>


Logger::Logger(const std::string& log_file, const bool real_time_logging_) : file(std::ofstream(log_file, std::ios::out))
{
    real_time_logging = real_time_logging_;

    plot_thread = std::move(std::thread(&Logger::Plot, this));
}

Logger::~Logger()
{
    if (!real_time_logging)
    {
        Dump();
    }
    file.close();
    if (plot_thread.joinable())
    {
        plot_thread.join();
    }
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

void Logger::Log(const uint64_t update_steps, const uint64_t play_steps,
    const std::map<std::string, float>& values)
{
    std::lock_guard<std::mutex> data_lock(data_mutex);
    //assert(columns_name.size() == value.size());

    for (const auto& s: values)
    {
        // If not real time, we can add columns during the logging process if we want to
        if (!real_time_logging)
        {
            logged_data[s.first];
        }

        // The column is assumed to exist
        logged_data.at(s.first).push_back({static_cast<double>(update_steps), static_cast<double>(play_steps), static_cast<double>(s.second) });
    }

    // Dump the new line to the log file
    if (real_time_logging)
    {
        // For each known column
        for (auto& s : logged_data)
        {
            const auto it = values.find(s.first);

            // If we have a value for this column
            if (it != values.end())
            {
                file << update_steps << "\t" << play_steps << "\t" << it->second << "\t";
            }
            // Else blank entry
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
    //TODO
}

void Logger::Plot() const
{
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

    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

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
}

void Logger::InternalPlotLoop(GLFWwindow* window) const
{
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    bool fit_data = true;
    std::unordered_map<std::string, bool> displayed_values;

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
                                ImGui::Text("Play steps:   %d", static_cast<int>(lowest->play_steps));
                                ImGui::Text("Update steps: %d", static_cast<int>(lowest->update_steps));
                                ImGui::Text("Value:        %.2f", lowest->value);
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
}
