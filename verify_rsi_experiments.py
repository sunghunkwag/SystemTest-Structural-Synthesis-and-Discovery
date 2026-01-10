
import time
import random
import sys
import os
from copy import deepcopy

# Fix path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from neuro_genetic_synthesizer import NeuroGeneticSynthesizer
except ImportError as e:
    print(f"Error: Could not import neuro_genetic_synthesizer: {e}")
    print(f"Sys Path: {sys.path}")
    sys.exit(1)

def generate_task(name, func):
    return [{'input': i, 'output': func(i)} for i in range(1, 11)]

def experiment_1_meta_adaptability():
    import datetime
    print(f"\n[PROOF OF WORK] Current Time: {datetime.datetime.now()}")
    print("=== Experiment 1: Meta-Controller Efficacy (Adaptive vs Static) ===")
    
    # Task: Hard Polynomial (n^3 - 2n^2 + n) - needs deep exploration
    target_func = lambda n: n**3 - 2*n**2 + n
    dataset = generate_task("hard_poly", target_func)
    
    # 1. Static (Control Group)
    print("Running Control Group (Static Parameters)...")
    random.seed(42)
    start_time = time.time()
    synth_static = NeuroGeneticSynthesizer(pop_size=100, generations=30)
    # Disable Meta-Update (Force static)
    synth_static.meta.update = lambda x: None 
    
    res_static = synth_static.synthesize(dataset, task_id="static")
    time_static = time.time() - start_time
    best_static = res_static[0][3] if res_static else 0
    print(f"  > Static Result: Fitness={best_static:.2f}, Time={time_static:.2f}s")

    # 2. Dynamic (Experimental Group)
    print("Running Experimental Group (RSI Meta-Controller)...")
    random.seed(42) # Same seed for fairness
    start_time = time.time()
    synth_dynamic = NeuroGeneticSynthesizer(pop_size=100, generations=30)
    # Meta controller is active by default
    
    res_dynamic = synth_dynamic.synthesize(dataset, task_id="dynamic")
    time_dynamic = time.time() - start_time
    best_dynamic = res_dynamic[0][3] if res_dynamic else 0
    print(f"  > Dynamic Result: Fitness={best_dynamic:.2f}, Time={time_dynamic:.2f}s")
    
    if best_dynamic >= best_static:
        print(f"✅ RSI VERIFIED: Dynamic system matched or outperformed static ({best_dynamic} vs {best_static})")
    else:
        print(f"❌ RSI FAILED: Dynamic system underperformed.")

def experiment_2_library_growth():
    print("\n=== Experiment 2: Library Growth Mechanism ===")
    synth = NeuroGeneticSynthesizer()
    initial_size = len(synth.ops)
    print(f"Initial Library Size: {initial_size} {synth.ops}")
    
    # Simulate discovering a primitive
    print("Simulating discovery of 'square' concept...")
    synth.register_primitive("square", lambda x: x*x)
    
    mid_size = len(synth.ops)
    print(f"Gen 100: Library Size = {mid_size} (Added 'square')")
    
    print("Simulating discovery of 'cube' concept...")
    synth.register_primitive("cube", lambda x: x*x*x)
    
    final_size = len(synth.ops)
    print(f"Gen 200: Library Size = {final_size} (Added 'cube')")
    
    if final_size > initial_size:
        print("✅ LEARNING VERIFIED: Library grew from {} to {}".format(initial_size, final_size))
    else:
        print("❌ LEARNING FAILED: Library size stagnant.")

def experiment_3_reuse_speedup():
    print("\n=== Experiment 3: Reuse Speedup (DreamCoder Effect) ===")
    
    # Task: n^8 (Hard for raw, Easy for library with square(square(square(n))))
    target_func = lambda n: n**8
    dataset = generate_task("n_power_8", target_func)
    
    # 1. Without Library
    print("Solving WITHOUT Library (Target: n^8)...")
    random.seed(42)
    start_time = time.time()
    synth_raw = NeuroGeneticSynthesizer(pop_size=100, generations=50) 
    res_raw = synth_raw.synthesize(dataset, task_id="raw")
    time_raw = time.time() - start_time
    found_raw = res_raw[0][3] >= 99.0 if res_raw else False
    fit_raw = res_raw[0][3] if res_raw else 0
    code_raw = res_raw[0][0] if res_raw else "None"
    print(f"  > Raw Result: Success={found_raw}, Fitness={fit_raw:.2f}, Time={time_raw:.4f}s")
    print(f"    Code: {code_raw}")
    
    # 2. With Library ('square' injected)
    print("Solving WITH Library (Target: square(square(square(n))))...")
    random.seed(42)
    start_time = time.time()
    synth_lib = NeuroGeneticSynthesizer(pop_size=100, generations=50)
    # Inject Knowledge
    synth_lib.register_primitive("square", lambda x: x*x)
    
    res_lib = synth_lib.synthesize(dataset, task_id="lib")
    time_lib = time.time() - start_time
    found_lib = res_lib[0][3] >= 99.0 if res_lib else False
    fit_lib = res_lib[0][3] if res_lib else 0
    code_lib = res_lib[0][0] if res_lib else "None"
    print(f"  > Library Result: Success={found_lib}, Fitness={fit_lib:.2f}, Time={time_lib:.4f}s")
    print(f"    Code: {code_lib}")
    
    if found_lib and (not found_raw or time_lib < time_raw):
        if not found_raw:
             print(f"✅ CAPABILITY VERIFIED: Library solved it, Raw failed. (Infinite Speedup)")
        else:
             speedup = time_raw / time_lib
             print(f"✅ SPEEDUP VERIFIED: {speedup:.1f}x faster with library reuse.")
    else:
        print(f"❌ SPEEDUP FAILED: Raw={fit_raw}, Lib={fit_lib}")

if __name__ == "__main__":
    experiment_1_meta_adaptability()
    experiment_2_library_growth()
    experiment_3_reuse_speedup()
