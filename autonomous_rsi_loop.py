import importlib
import sys
import time
import re
import random
import os

# Ensure rs_machine is available
try:
    import rs_machine
except ImportError:
    print("rs_machine not found. Please install it first.")
    sys.exit(1)

try:
    import neuro_genetic_synthesizer
    import self_purpose_engine
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

class AutonomousOrchestrator:
    def __init__(self):
        self.engine = self_purpose_engine.SelfPurposeEngine()
        self.cycle_count = 0

    def run_loop(self, max_cycles=5):
        print("[RSI-Orchestrator] Starting Autonomous Loop...")

        while self.cycle_count < max_cycles:
            self.cycle_count += 1
            print(f"\n=== CYCLE {self.cycle_count} ===")

            # 1. Reload Synthesizer (to apply previous self-rewrites)
            try:
                importlib.reload(neuro_genetic_synthesizer)
                print("[System] Module reloaded.")
            except Exception as e:
                print(f"[System] Reload failed: {e}")

            # 2. Formulate Goal
            goal = self.engine.formulate_goal()
            print(f"[Goal] {goal}")

            # 3. Run Synthesis
            # We pass an empty IO set because we are doing physical grounding
            synth = neuro_genetic_synthesizer.NeuroGeneticSynthesizer(pop_size=50, generations=5)
            results = synth.synthesize([], goal=goal)

            if not results:
                print("[Result] No viable solution found.")
                # Even if no solution, we might want to adjust weights if it's too hard?
                continue

            best_expr_str, best_expr, size, fitness = results[0]
            print(f"[Result] Best Fitness: {fitness:.2f}, Size: {size}")

            # 4. Measure Physical Metrics (Ground Truth)
            metrics = {'energy': 0.0, 'structural_entropy': 0.0}
            try:
                compiler = neuro_genetic_synthesizer.RustCompiler()
                instructions = compiler.compile(best_expr)
                if instructions:
                    vm = rs_machine.VirtualMachine(1000, 64, 16)
                    st = vm.execute(instructions, [0.0])
                    metrics['energy'] = st.energy
                    metrics['structural_entropy'] = st.structural_entropy
                    print(f"[Metrics] Energy: {st.energy:.2f}, Entropy: {st.structural_entropy:.2f}")
            except Exception as e:
                print(f"[Error] Measurement failed: {e}")

            # 5. Feedback to Engine
            self.engine.observe(metrics)

            # 6. Self-Modification (Rewriting Evaluation Function)
            self._rewrite_evaluation_function(metrics, goal)

    def _rewrite_evaluation_function(self, metrics, goal):
        """
        Reads neuro_genetic_synthesizer.py and tweaks the fitness constants.
        """
        filepath = "neuro_genetic_synthesizer.py"
        try:
            with open(filepath, "r") as f:
                content = f.read()

            # Regex to find the entropy weight: "entropy_dist * 5.0"
            # We look for float number
            pattern = re.compile(r"entropy_dist \* (\d+\.\d+)")
            match = pattern.search(content)

            if match:
                current_weight_str = match.group(1)
                current_weight = float(current_weight_str)

                # Heuristic: If entropy error is high, increase weight
                entropy_error = abs(metrics['structural_entropy'] - goal.target_entropy)

                new_weight = current_weight
                if entropy_error > 0.5:
                    new_weight *= 1.2 # Increase penalty
                    print(f"[Self-Rewrite] High entropy error ({entropy_error:.2f}). Increasing weight: {current_weight:.2f} -> {new_weight:.2f}")
                elif entropy_error < 0.1 and current_weight > 1.0:
                    new_weight *= 0.9 # Relax constraint
                    print(f"[Self-Rewrite] Low entropy error. Relaxing weight: {current_weight:.2f} -> {new_weight:.2f}")

                if abs(new_weight - current_weight) > 0.01:
                    # Replace only the first occurrence (which is in _fitness hopefully)
                    # We construct the new string carefully
                    new_content = content.replace(f"entropy_dist * {current_weight_str}", f"entropy_dist * {new_weight:.2f}")

                    with open(filepath, "w") as f:
                        f.write(new_content)
                    print("[Self-Rewrite] neuro_genetic_synthesizer.py UPDATED.")
            else:
                print("[Self-Rewrite] Could not find entropy weight pattern.")

        except Exception as e:
            print(f"[Self-Rewrite] Failed: {e}")

if __name__ == "__main__":
    orch = AutonomousOrchestrator()
    orch.run_loop()
