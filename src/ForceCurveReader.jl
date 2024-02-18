#!/bin/env julia

# Required packages
using DelimitedFiles, Plots
using Statistics: mean
gr()

# Define the moving average lambda since julia doesn't have a smooth function
movingaverage(g, n) = [i < n ? mean(g[begin:i]) : mean(g[i-n+1:i]) for i in 1:length(g)]

# Set axis limits
MyXlim = (0, 1e-7)
MyYlim = (-1e-4, 2e-4)

# Function to parse a single line of data
function parse_line(line)
    # Split the line by spaces and filter out empty strings
    parts = filter(!isempty, split(line, ' '))
    # Extract and convert numerical values, skipping labels
    values = Float64[parse(Float64, parts[i]) for i in 2:2:length(parts)]
    return values
end

# Read and parse the file
function plotForceCurveFile(file_path, title_text)
    lines = readlines(file_path)
    data = [parse_line(line) for line in lines]

    # Convert the list of vectors into a matrix
    Amat = hcat(data...)'

    # Extract columns for t, FZ1, FZ2, and FZ3
    t = Amat[:, 1]
    FZ1 = Amat[:, 4]
    FZ2 = Amat[:, 7]
    FZ3 = Amat[:, 10]
    #println(length(t))
    #println(length(FZ1))

    # Smooth FZ1 with a window size of 30
    FZ1smooth = movingaverage(FZ1, 30)
    #println(length(FZ1smooth))

    # Plotting
    plot(t, [FZ3 FZ2 FZ1 FZ1smooth],
         label=["FZ3: Pressure Gradient Force" "FZ2: Separated Accel Force" "FZ1: Avg Accel Force" "FZ1s: Avg Accel Force (Smoothed)"],
         grid=true,
         xlabel="Time",
         ylabel="Force",
         title=title_text,
         xlims=MyXlim,
         ylims=MyYlim)
end

file_path = "ForceCurveGPU2024_02_16__23_59_49.txt"
corrected_file_path = "ForceCurveGPU2024_02_16__23_59_49.txt"
before_plot = plotForceCurveFile(file_path, "Before Error Correction")
after_plot = plotForceCurveFile(corrected_file_path, "After Error Correction")
plot(before_plot, after_plot, layout=(2, 1), size=(1187,768), legend_bg_opacity=0.4)
gui()
println("Press Enter to exit.")
readline(stdin)
