import os
import re
from pathlib import Path
import matplotlib.pyplot as plt

def parse_dot_file(dot_file):
    """
    Parse a .dot file to extract model information such as node count and tensor sizes.
    
    Args:
        dot_file (str): Path to the .dot file.
    
    Returns:
        dict: Dictionary containing model node count and parameter sizes.
    """
    node_count = 0
    param_sizes = []
    
    with open(dot_file, 'r') as file:
        for line in file:
            # Count nodes
            if '->' not in line and '[' in line:
                node_count += 1
            
            # Extract tensor sizes from node labels
            size_match = re.search(r'\((.*?)\)', line)
            if size_match:
                sizes = tuple(map(int, size_match.group(1).split(',')))
                param_sizes.append(sizes)

    return {
        "node_count": node_count,
        "param_sizes": param_sizes
    }

def save_comparison_results(model_1_info, model_2_info, output_file):
    """
    Save comparison results to a text file.
    
    Args:
        model_1_info (dict): Information about the first model.
        model_2_info (dict): Information about the second model.
        output_file (str): Path to the output file.
    """
    with open(output_file, 'w') as file:
        # Node count comparison
        file.write(f"Model 1 Node Count: {model_1_info['node_count']}\n")
        file.write(f"Model 2 Node Count: {model_2_info['node_count']}\n\n")

        # Parameter size comparison
        file.write("Comparing Parameter Sizes:\n")
        for idx, (param_1, param_2) in enumerate(zip(model_1_info['param_sizes'], model_2_info['param_sizes']), start=1):
            if param_1 != param_2:
                file.write(f"Layer {idx}: Model 1 = {param_1}, Model 2 = {param_2}\n")
            else:
                file.write(f"Layer {idx}: No Change\n")

def plot_comparison(model_1_info, model_2_info, plot_line_output, plot_bar_output, plot_diff_output):
    """
    Plot and save parameter size comparison using line plot, bar plot, and difference plot.
    
    Args:
        model_1_info (dict): Information about the first model.
        model_2_info (dict): Information about the second model.
        plot_line_output (str): Path to the output line plot image file.
        plot_bar_output (str): Path to the output bar plot image file.
        plot_diff_output (str): Path to the output difference plot image file.
    """
    # Prepare data
    model_1_sizes = [sum(size) for size in model_1_info['param_sizes']]
    model_2_sizes = [sum(size) for size in model_2_info['param_sizes']]
    param_diff = [m1 - m2 for m1, m2 in zip(model_1_sizes, model_2_sizes)]
    layers = list(range(1, len(model_1_sizes) + 1))

    # Line plot
    plt.figure(figsize=(10, 6))
    plt.plot(layers, model_1_sizes, label="Model 1 Parameter Sizes", marker='o', markersize=4, linewidth=1, color='blue')
    plt.plot(layers, model_2_sizes, label="Model 2 Parameter Sizes", marker='x', markersize=4, linewidth=1, color='green')
    plt.xlabel("Layer")
    plt.ylabel("Parameter Size (Sum of Dimensions)")
    plt.title("Model Parameter Size Comparison (Line Plot)")
    plt.legend()
    plt.grid()
    plt.savefig(plot_line_output)
    plt.close()

    # Bar plot
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    bar_positions_1 = [x - bar_width / 2 for x in layers]
    bar_positions_2 = [x + bar_width / 2 for x in layers]
    plt.bar(bar_positions_1, model_1_sizes, width=bar_width, label="Model 1 Parameter Sizes", color='blue', alpha=0.7)
    plt.bar(bar_positions_2, model_2_sizes, width=bar_width, label="Model 2 Parameter Sizes", color='green', alpha=0.7)
    plt.xlabel("Layer")
    plt.ylabel("Parameter Size (Sum of Dimensions)")
    plt.title("Model Parameter Size Comparison (Bar Plot)")
    plt.legend()
    plt.grid()
    plt.savefig(plot_bar_output)
    plt.close()

    # Difference plot
    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--', label='Zero Difference')
    plt.bar(layers, param_diff, color=['red' if d < 0 else 'blue' for d in param_diff], alpha=0.7)
    plt.xlabel("Layer")
    plt.ylabel("Parameter Size Difference (Model 1 - Model 2)")
    plt.title("Layer-wise Parameter Size Difference")
    plt.legend()
    plt.grid()
    plt.savefig(plot_diff_output)
    plt.close()

def compare_model_files(file_1, file_2, text_output, plot_line_output, plot_bar_output, plot_diff_output):
    """
    Compare two model structure .dot files and save results.
    
    Args:
        file_1 (str): Path to the first .dot file.
        file_2 (str): Path to the second .dot file.
        text_output (str): Path to save text comparison results.
        plot_line_output (str): Path to save line plot image file.
        plot_bar_output (str): Path to save bar plot image file.
        plot_diff_output (str): Path to save difference plot image file.
    """
    
    if not Path(file_1).exists() or not Path(file_2).exists():
        print("One or both .dot files do not exist. Ensure valid paths.")
        return

    # Parse the .dot files
    model_1_info = parse_dot_file(file_1)
    model_2_info = parse_dot_file(file_2)

    # Save comparison results
    save_comparison_results(model_1_info, model_2_info, text_output)
    plot_comparison(model_1_info, model_2_info, plot_line_output, plot_bar_output, plot_diff_output)
    print(f"Comparison results saved to \n{text_output}, \n{plot_line_output}, \n{plot_bar_output}, \nand {plot_diff_output}")


def main():
    """
    Main function to compare model .dot files.
    """
    dot_base_dir = "./model_structures"
    save_base_dir = "./comparison_files"
    
    # Example .dot file paths
    file_1 = "baseline_structure"
    file_2 = "face_alignment_rat-prune-02_hrnet_w18"
    
    file_1_path = os.path.join(dot_base_dir, file_1)
    file_2_path = os.path.join(dot_base_dir, file_2)

    # Output file paths
    text_output = f"{save_base_dir}/{os.path.splitext(file_1)[0]}-{os.path.splitext(file_2)[0]}_comparison.txt"
    plot_line_output = f"{save_base_dir}/{os.path.splitext(file_1)[0]}-{os.path.splitext(file_2)[0]}_comparison_line_plot.png"
    plot_bar_output = f"{save_base_dir}/{os.path.splitext(file_1)[0]}-{os.path.splitext(file_2)[0]}_comparison_bar_plot.png"
    plot_diff_output = f"{save_base_dir}/{os.path.splitext(file_1)[0]}-{os.path.splitext(file_2)[0]}_comparison_diff_plot.png"
    
    print("Comparing model structures...")
    compare_model_files(file_1_path, file_2_path, text_output, plot_line_output, plot_bar_output, plot_diff_output)

if __name__ == "__main__":
    main()
