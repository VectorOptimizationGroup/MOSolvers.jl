const _OPTIMIZER_HEADER_WIDTH = 70

# Box rendering primitives. These helpers centralize width handling so the
# solver header and table header stay visually consistent.

"""
    print_box_section_header(title::String, width::Int = 70)

Print a section label inside an ASCII metadata box.

Long titles are truncated to fit the available inner width.
"""
function print_box_section_header(title::String, width::Int = 70)
    max_content_len = width - 4
    header = " $(title):"
    if length(header) > max_content_len
        header = header[1:max_content_len-3] * "..."
    else
        header = header * " "^(max_content_len - length(header))
    end
    @info "| " * header * " |"
end

"""
    print_box_empty_line(width::Int = 70)

Print an empty line inside an ASCII metadata box.
"""
function print_box_empty_line(width::Int = 70)
    @info "| " * " "^(width - 4) * " |"
end

"""
    _print_box_raw_line(line::String, width::Int = _OPTIMIZER_HEADER_WIDTH)

Print a preformatted string inside an ASCII metadata box.

This preserves the alignment already present in `line` and only applies the
outer box padding/truncation.
"""
function _print_box_raw_line(line::String, width::Int = _OPTIMIZER_HEADER_WIDTH)
    max_content_len = width - 4
    item = length(line) > max_content_len ? line[1:max_content_len-3] * "..." : line
    @info "| " * item * " "^(max_content_len - length(item)) * " |"
end

"""
    print_box_single_line(key::String, val::String, width::Int = 70)

Print one key-value pair inside an ASCII metadata box.
"""
function print_box_single_line(key::String, val::String, width::Int = 70)
    max_content_len = width - 4
    item = "  * $(key): $(val)"
    if length(item) > max_content_len
        item = item[1:max_content_len-3] * "..."
    else
        item = item * " "^(max_content_len - length(item))
    end
    @info "| " * item * " |"
end

"""
    print_box_dual_line(key1::String, val1::String, key2::String, val2::String, width::Int = 70)

Print two key-value pairs side by side inside an ASCII metadata box.

Each pair occupies half of the box content area and is truncated if needed.
"""
function print_box_dual_line(key1::String, val1::String, key2::String, val2::String, width::Int = 70)
    max_content_len = width - 4
    col_width = div(max_content_len, 2)

    item1 = "  * $(key1): $(val1)"
    if length(item1) > col_width
        item1 = item1[1:col_width-3] * "..."
    else
        item1 = item1 * " "^(col_width - length(item1))
    end

    item2 = "  * $(key2): $(val2)"
    if length(item2) > col_width
        item2 = item2[1:col_width-3] * "..."
    else
        item2 = item2 * " "^(col_width - length(item2))
    end

    @info "| " * item1 * item2 * " |"
end

"""
    _print_box_pairs(params::Vector{Pair{String,String}}, width::Int = _OPTIMIZER_HEADER_WIDTH)

Print a list of display pairs in two-column form.

The order of `params` is part of the rendered output: adjacent pairs are printed
on the same line.
"""
function _print_box_pairs(params::Vector{Pair{String,String}}, width::Int = _OPTIMIZER_HEADER_WIDTH)
    for i in 1:2:length(params)
        if i + 1 <= length(params)
            print_box_dual_line(params[i].first, params[i].second, params[i+1].first, params[i+1].second, width)
        else
            print_box_single_line(params[i].first, params[i].second, width)
        end
    end
end

"""
    _print_box_section(title::String, params::Vector{Pair{String,String}}, width::Int = _OPTIMIZER_HEADER_WIDTH)

Print a titled metadata section inside the solver header box.
"""
function _print_box_section(title::String, params::Vector{Pair{String,String}}, width::Int = _OPTIMIZER_HEADER_WIDTH)
    print_box_section_header(title, width)
    _print_box_pairs(params, width)
end

"""
    _format_bool(value::Bool)

Format booleans with lowercase strings for stable header output.
"""
function _format_bool(value::Bool)
    return value ? "true" : "false"
end

# Metadata extraction helpers. Each method returns ordered `Pair{String,String}`
# values and leaves all box rendering to the functions above.

"""
    _global_header_pairs(options, n::Int, m::Int)

Return common global solver settings for display in `print_optimizer_header`.
"""
function _global_header_pairs(options, n::Int, m::Int)
    pairs = Pair{String,String}[
        "Variables (n)" => string(n),
        "Objectives (m)" => string(m),
        "Max Iterations" => string(options.max_iter),
        "Optimality Tol" => @sprintf("%.2e", options.opt_tol),
        "Progress Tol" => @sprintf("%.2e", options.ftol),
    ]
    isfinite(options.max_time) && push!(pairs, "Max Time (s)" => @sprintf("%.2e", options.max_time))
    return pairs
end

"""
    _solver_header_pairs(options)

Return solver-specific display metadata for `print_optimizer_header`.
"""
_solver_header_pairs(options) = Pair{String,String}[]

function _solver_header_pairs(options::CondGParams)
    return Pair{String,String}[
        "Stop Criteria" => string(options.stop_criteria),
        "Store Trace" => _format_bool(options.store_trace),
    ]
end

function _solver_header_pairs(options::PDFPMParams)
    lambda_factor_str = options.lambda_factor isa Function ? "Function" : @sprintf("%.2e", options.lambda_factor)
    return Pair{String,String}[
        "Stop Criteria" => string(options.stop_criteria),
        "Store Trace" => _format_bool(options.store_trace),
        "Epsilon (ϵ)" => @sprintf("%.2e", options.epsilon),
        "Sigma (σ)" => @sprintf("%.2e", options.sigma),
        "Alpha (α)" => @sprintf("%.2e", options.alpha),
        "Max Subprob Iter" => string(options.max_subproblem_iter),
        "Lambda Factor" => lambda_factor_str,
        "Lambda Min" => @sprintf("%.2e", options.lambda_min),
        "Step Tol Rule" => string(options.step_tolerance_rule),
    ]
end

function _solver_header_pairs(options::ProxGradParams)
    return Pair{String,String}[
        "Stop Criteria" => string(options.stop_criteria),
        "Store Trace" => _format_bool(options.store_trace),
        "mu (μ)" => @sprintf("%.2e", options.mu),
    ]
end

"""
    print_optimizer_header(optimizer_name::String, options, n::Int, m::Int)

Print a structured ASCII summary card representing the optimization solver setup.
"""
function print_optimizer_header(optimizer_name::String, options, n::Int, m::Int)
    width = _OPTIMIZER_HEADER_WIDTH
    border = "+" * "-"^(width-2) * "+"

    @info border

    header_content = "$(optimizer_name) : Multiobjective Optimization"
    max_content_len = width - 4
    padding = max(0, max_content_len - length(header_content))
    left_pad = div(padding, 2)
    right_pad = padding - left_pad
    header_line = "| " * " "^left_pad * header_content * " "^right_pad * " |"
    @info header_line

    @info border

    _print_box_section("Global Settings", _global_header_pairs(options, n, m), width)

    solver_params = _solver_header_pairs(options)
    if !isempty(solver_params)
        print_box_empty_line()
        _print_box_section("Solver-specific", solver_params, width)
    end

    @info border
end

"""
    print_table(result::OptimResult)

Print the iteration table header and one progress row for an optimization result.

On the initial iteration (`result.iter == 0`), this prints a boxed table header
followed by the initial progress row. On later iterations, only the progress row
is printed. The progress column uses `result.norm_progress` except at the
initial point, where the displayed progress is `--`.
"""
function print_table(result::OptimResult)
    if result.iter == 0
        width = _OPTIMIZER_HEADER_WIDTH
        border = "+" * "-"^(width-2) * "+"
        @info border
        headers1 = ["", "", "||x^k - x^{k-1}||"]
        headers2 = ["Iteration", "|θ(x^k)|", "---------------------"]
        headers3 = ["", "", "max(1,||x^{k-1}||)"]
        
        format_str1 = "%12s  %16s  %26s"
        format_str2 = "%12s  %16s  %28s"
        format_str3 = "%12s  %16s  %27s"

        fmt1 = Printf.Format(format_str1)
        fmt2 = Printf.Format(format_str2)
        fmt3 = Printf.Format(format_str3)

        header_str1 = Printf.format(fmt1, headers1...)
        header_str2 = Printf.format(fmt2, headers2...)
        header_str3 = Printf.format(fmt3, headers3...)

        _print_box_raw_line(header_str1, width)
        _print_box_raw_line(header_str2, width)
        _print_box_raw_line(header_str3, width)
        @info border

        # Print progress row
        progress_str = @sprintf("%10d  %22.6e  %16s",
                result.iter, result.abs_theta, "--")
        @info progress_str
    else
        # Print progress row
        progress_str = @sprintf("%10d  %22.6e  %22.6e",
                result.iter, result.abs_theta, result.norm_progress)
        @info progress_str
    end
end
