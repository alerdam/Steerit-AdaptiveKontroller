import numpy as np
import time
from pipe import Pipeline

def run_stability_test(frame_count=200):
    """
    Measures the precision and stability of the inference system 
    by analyzing the variance of the output while the input is stationary.
    """
    pipe = Pipeline()
    inference_results = []
    
    print(f"--- STABILITY TEST STARTED ---")
    print(f"Instruction: Keep your hands in a fixed steering position.")
    print(f"Monitoring {frame_count} frames...")

    # Warm-up period to let MediaPipe stabilize its tracking
    for _ in range(15):
        pipe.step()

    try:
        for i in range(frame_count):
            result = pipe.step()
            
            # Check if hands are detected
            if result and result.get("output") is not None:
                current_val = result["output"]
                inference_results.append(current_val)
                
                if i % 25 == 0:
                    print(f"Progress: [{i}/{frame_count}] Current Output: {current_val:.4f}")
            else:
                print(f"Warning: Tracking lost at frame {i}")

        if len(inference_results) == 0:
            print("Error: No data captured. Test failed.")
            return

        # Statistical Calculations
        data = np.array(inference_results)
        mean_val = np.mean(data)
        std_dev = np.std(data)
        max_jitter = np.max(np.abs(data - mean_val))
        stability_score = max(0, 100 * (1 - (std_dev / (abs(mean_val) + 1e-6))))

        print("\n" + "="*30)
        print("   FINAL STABILITY REPORT")
        print("="*30)
        print(f"Total Valid Frames    : {len(data)}")
        print(f"Average Output (Mean) : {mean_val:.6f}")
        print(f"Standard Deviation    : {std_dev:.6f} (Precision Metric)")
        print(f"Maximum Jitter        : {max_jitter:.6f} (Peak Deviation)")
        print(f"Stability Percentage  : {stability_score:.2f}%")
        print("="*30)
        
        if std_dev < 0.02:
            print("Result: EXCELLENT STABILITY")
        elif std_dev < 0.05:
            print("Result: GOOD STABILITY")
        else:
            print("Result: STABILITY NEEDS FILTERING")

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        pipe.release()

if __name__ == "__main__":
    run_stability_test()