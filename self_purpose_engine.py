"""
SELF-PURPOSE ENGINE WITH PHYSICAL GROUNDING
============================================
The system defines its own purpose based on physical resource constraints.
Abolishes bottom-up axiom search in favor of top-down physical targeting.

Mechanism:
1. Observe current physical state (Energy, Entropy)
2. Formulate PhysicalGoal (Target Energy, Target Entropy)
3. Drift goals to find critical states (Singularity)
"""

import random
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class PhysicalGoal:
    """A target physical state for the system to achieve."""
    target_energy: float
    target_entropy: float
    tolerance: float = 0.1
    
    def __repr__(self):
        return f"PhysicalGoal(E={self.target_energy:.2f}, S={self.target_entropy:.2f})"

# ==============================================================================
# SELF-PURPOSE ENGINE
# ==============================================================================

class SelfPurposeEngine:
    """
    Top-Down Discovery Engine.
    Drives the synthesis loop by setting physical resource targets.
    """
    
    def __init__(self):
        # Initial low-energy, low-entropy state
        self.current_goal = PhysicalGoal(target_energy=5.0, target_entropy=1.0)
        self.history: List[PhysicalGoal] = []
        self.metrics_history: List[Dict[str, float]] = []
        self.cycle = 0
        
        # Drift parameters
        self.energy_drift = 0.5
        self.entropy_drift = 0.1
    
    def formulate_goal(self) -> PhysicalGoal:
        """
        Formulate a new physical goal.
        Drifts towards higher complexity (Higher Energy + Higher Entropy).
        """
        # Linear drift for now, can be made sigmoid or chaotic later
        new_energy = self.current_goal.target_energy + self.energy_drift * random.uniform(0.5, 1.5)
        new_entropy = self.current_goal.target_entropy + self.entropy_drift * random.uniform(0.5, 1.5)
        
        # Cap values to reasonable limits for the VM
        new_energy = min(new_energy, 1000.0) # Max steps
        new_entropy = min(new_entropy, 10.0) # Log2 of trace length
        
        self.current_goal = PhysicalGoal(
            target_energy=new_energy,
            target_entropy=new_entropy
        )
        self.history.append(self.current_goal)
        return self.current_goal
    
    def observe(self, metrics: Dict[str, float]):
        """
        Observe the actual metrics from the last synthesis cycle.
        Adjust drift if we are too far off (feedback loop).
        """
        self.metrics_history.append(metrics)
        self.cycle += 1
        
        # Calculate error
        e_err = metrics.get('energy', 0) - self.current_goal.target_energy
        s_err = metrics.get('structural_entropy', 0) - self.current_goal.target_entropy
        
        print(f"[SelfPurpose] Cycle {self.cycle}: Goal={self.current_goal}, Actual=(E={metrics.get('energy',0):.2f}, S={metrics.get('structural_entropy',0):.2f})")
        
        # Adaptive Drift: If we undershot, drift slower. If overshot, maybe pull back?
        # For Singularity, we want to push boundaries, so we keep drifting up unless we hit a wall.
        if abs(e_err) < 2.0 and abs(s_err) < 0.5:
            # We met the goal, accelerate!
            self.energy_drift *= 1.1
            self.entropy_drift *= 1.1
            print("[SelfPurpose] Goal met. Accelerating drift.")
        else:
            # We missed, decelerate to let synthesizer catch up
            self.energy_drift *= 0.9
            self.entropy_drift *= 0.9
            print("[SelfPurpose] Goal missed. Decelerating drift.")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "current_goal": str(self.current_goal),
            "energy_drift": self.energy_drift,
            "entropy_drift": self.entropy_drift
        }
    
    # Legacy support wrappers if needed, but we are refactoring consumer too
    def get_dominant_purpose(self) -> str:
        return f"Seeking State: {self.current_goal}"

if __name__ == "__main__":
    engine = SelfPurposeEngine()
    print("Initial Goal:", engine.formulate_goal())
    engine.observe({'energy': 6.0, 'structural_entropy': 1.2})
    print("Next Goal:", engine.formulate_goal())
