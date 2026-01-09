import random
import time
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Callable

# ==============================================================================
# Domain Variable/Primitives
# ==============================================================================

@dataclass(frozen=True)
class Expr:
    pass

@dataclass(frozen=True)
class BSVar(Expr):
    name: str = 'n'
    def __repr__(self): return self.name

@dataclass(frozen=True)
class BSVal(Expr):
    val: Any
    def __repr__(self): return str(self.val)

@dataclass(frozen=True)
class BSApp(Expr):
    func: str
    args: tuple
    def __repr__(self): 
        return f"{self.func}({', '.join(repr(a) for a in self.args)})"

class NeuroInterpreter:
    PRIMS = {
        'add': lambda a, b: a + b,
        'mul': lambda a, b: a * b,
        'sub': lambda a, b: a - b,
        'div': lambda a, b: a // b if b != 0 else 0,
        'mod': lambda a, b: a % b if b != 0 else 0,
        'if_gt': lambda a, b, c, d: c if a > b else d,
    }
    
    def run(self, expr, env):
        try:
            return self._eval(expr, env, 50)
        except:
            return None
    
    def _eval(self, expr, env, gas):
        if gas <= 0: return None
        if isinstance(expr, BSVar): return env.get(expr.name, 0)
        if isinstance(expr, BSVal): return expr.val
        if isinstance(expr, BSApp):
            fn = expr.func
            if fn in self.PRIMS:
                args = [self._eval(a, env, gas-1) for a in expr.args]
                if None in args: return None
                try: return self.PRIMS[fn](*args)
                except: return None
        return None

    
    def register_primitive(self, name: str, func: Callable):
        """Add a new primitive (discovered concept) to the interpreter."""
        self.PRIMS[name] = func

# ==============================================================================
# Genetic Programming Components
# ==============================================================================

class NeuroGeneticSynthesizer:
    """
    Combines Evolutionary Search (Genetic Algorithm) with Neural Guidance.
    NO TRANSFORMERS. Uses simple probability distributions from the Neural Guide.
    """
    def __init__(self, neural_guide=None, pop_size=200, generations=20):
        self.guide = neural_guide  # Object with get_priors(io_pairs) -> Dict[op, prob]
        self.pop_size = pop_size
        self.generations = generations
        self.interp = NeuroInterpreter()
        self.rng = random.Random()
        
        # Base Atoms
        self.atoms = [BSVar('n'), BSVal(0), BSVal(1), BSVal(2), BSVal(3)]
        self.ops = list(NeuroInterpreter.PRIMS.keys())
        
    def register_primitive(self, name: str, func: Callable):
        """Register a new primitive for synthesis."""
        self.interp.register_primitive(name, func)
        if name not in self.ops:
            self.ops.append(name)
            print(f"[NeuroGen] Registered new primitive: {name}")

    def synthesize(self, io_pairs: List[Dict[str, Any]], deadline=None, task_id="", task_params=None, **kwargs) -> List[Tuple[str, Expr, float, float]]:
        start_time = time.time()
        
        # 1. Get Neural Guidance (Priors)
        priors = {'add': 1.0, 'mul': 1.0, 'sub': 1.0, 'div': 1.0, 'mod': 0.5, 'if_gt': 0.1}
        if self.guide:
            learned_priors = self.guide.get_priors(io_pairs)
            if learned_priors:
                priors.update(learned_priors)
        
        # Normalize priors to probabilities
        total_p = sum(priors.values())
        op_probs = {k: v/total_p for k, v in priors.items()}
        
        # 2. Initialize Population
        population = [self._random_expr(2, op_probs) for _ in range(self.pop_size)]
        
        best_solution = None
        best_fitness = -1.0
        
        for gen in range(self.generations):
            if deadline and time.time() > deadline: break
            
            # Evaluate Fitness
            scored_pop = []
            for expr in population:
                fit = self._fitness(expr, io_pairs)
                scored_pop.append((fit, expr))
                
                if fit >= 100.0:
                    # Early Exit on Solution
                    return [(str(expr), expr, self._size(expr), fit)]
            
            # Sort by fitness
            scored_pop.sort(key=lambda x: x[0], reverse=True)
            current_best = scored_pop[0]
            
            if current_best[0] > best_fitness:
                best_fitness = current_best[0]
                best_solution = current_best
            
            # Selection (Elitism + Tournament)
            next_gen = [p[1] for p in scored_pop[:10]] # Elitism
            
            while len(next_gen) < self.pop_size:
                parent1 = self._tournament(scored_pop)
                parent2 = self._tournament(scored_pop)
                
                if self.rng.random() < 0.7:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1
                    
                if self.rng.random() < 0.3:
                    child = self._mutate(child, op_probs)
                    
                next_gen.append(child)
                
            population = next_gen
            
        return [(str(best_solution[1]), best_solution[1], self._size(best_solution[1]), best_fitness)] if best_solution else []

    def _random_expr(self, depth, op_probs):
        if depth <= 0 or self.rng.random() < 0.3:
            return self.rng.choice(self.atoms)
        
        # Choose op based on Neural Priors
        op = self.rng.choices(list(op_probs.keys()), weights=list(op_probs.values()))[0]
        
        # Arity check (special case for if_gt which is 4-ary)
        arity = 4 if op == 'if_gt' else 2
        args = tuple(self._random_expr(depth-1, op_probs) for _ in range(arity))
        return BSApp(op, args)

    def _fitness(self, expr, ios):
        score = 0
        hits = 0
        for io in ios:
            out = self.interp.run(expr, { 'n': io['input'] })
            if out == io['output']:
                hits += 1
                score += 1
            else:
                # Distance-based partial credit?
                if isinstance(out, (int, float)) and isinstance(io['output'], (int, float)):
                   diff = abs(out - io['output'])
                   if diff < 100: score += 1.0 / (1.0 + diff)
        
        # Normalize to 0-100
        return (score / len(ios)) * 100.0

    def _tournament(self, scored_pop):
        # Pick k random, return best
        k = 5
        candidates = self.rng.sample(scored_pop, k)
        return max(candidates, key=lambda x: x[0])[1]

    def _crossover(self, p1, p2):
        # Subtree Exchange
        if isinstance(p1, BSApp) and isinstance(p2, BSApp) and self.rng.random() < 0.5:
            # Swap arguments
            new_args = list(p1.args)
            idx = self.rng.randint(0, len(new_args)-1)
            new_args[idx] = p2 # Graft p2 onto p1
            return BSApp(p1.func, tuple(new_args))
        return p1 # Fallback

    def _mutate(self, p, op_probs):
        # Point Mutation or Subtree Regrowth
        if self.rng.random() < 0.5:
            # Regrowth
            return self._random_expr(2, op_probs)
        else:
            # Op mutation
            if isinstance(p, BSApp):
                new_op = self.rng.choices(list(op_probs.keys()), weights=list(op_probs.values()))[0]
                arity = 4 if new_op == 'if_gt' else 2
                current_arity = 4 if p.func == 'if_gt' else 2
                
                if arity == current_arity:
                    return BSApp(new_op, p.args)
        return p

    def _size(self, expr):
        if isinstance(expr, BSApp):
            return 1 + sum(self._size(a) for a in expr.args)
        return 1
