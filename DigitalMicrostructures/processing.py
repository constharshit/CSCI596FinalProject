import subprocess
import matplotlib.pyplot as plt
import re

def run_serial():
    result = subprocess.run(["python3", "serial.py"], capture_output=True, text=True)
    return result.stdout

def run_parallel():
    result = subprocess.run(["mpiexec", "-n", "4", "python3", "parallel.py"], capture_output=True, text=True)
    return result.stdout

def extract_fraction_gb(output):
    # Use regular expression to find the fraction value
    match = re.search(r'Fraction\s+of\s+grain\s+boundary\s+pixels\s+=\s+([0-9.]+)', output)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            print(f"Error extracting fraction from: {match.group(1)}")
    else:
        print("Fraction not found in the output.")

def plot_results(serial_output, parallel_output):
    print("Serial Output:")
    print(serial_output)

    print("\nParallel Output:")
    print(parallel_output)

    serial_execution_time = float(serial_output.split()[3])
    parallel_execution_time = float(parallel_output.split()[3])

    serial_fraction_gb = extract_fraction_gb(serial_output)
    parallel_fraction_gb = extract_fraction_gb(parallel_output)

    labels = ['Serial', 'Parallel']
    execution_times = [serial_execution_time, parallel_execution_time]
    fraction_gb = [serial_fraction_gb, parallel_fraction_gb]

    # Plot execution time
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(labels, execution_times, marker='o', color='blue', label='Execution Time')
    plt.title('Execution Time Comparison')
    plt.ylabel('Time (ms)')
    plt.legend()

    # Plot fraction of grain boundary pixels
    plt.subplot(1, 2, 2)
    plt.plot(labels, fraction_gb, marker='o', color='orange', label='Fraction of GB Pixels')
    plt.title('Fraction of Grain Boundary Pixels Comparison')
    plt.ylabel('Fraction')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    serial_output = run_serial()
    parallel_output = run_parallel()

    plot_results(serial_output, parallel_output)


