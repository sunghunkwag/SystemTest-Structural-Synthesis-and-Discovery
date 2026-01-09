#!/usr/bin/env python3
"""
HARDCORE OEIS BENCHMARK - RANDOM MODE
=====================================
Fetches TRULY RANDOM sequences from oeis.org.
Absolutely no selection bias.
Most of these will be unsolvable within 10 seconds.
"""
import urllib.request
import json
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Set


# ==============================================================================
# RANDOM OEIS FETCHER
# ==============================================================================

def fetch_random_oeis() -> Dict[str, Any]:
    """
    Fetch a RANDOM sequence by picking a random A-number.
    Standard A-numbers go from A000001 to ~A370000.
    """
    # Generate random A-number: A + 6 digits
    rand_num = random.randint(1, 360000)
    a_number = f"A{rand_num:06d}"
    
    url = f"https://oeis.org/search?fmt=json&q=id:{a_number}"
    print(f"  [RANDOM FETCH] {url}")
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode())
            
            def parse_result(res):
                seq_str = res.get('data', '')
                sequence = []
                for x in seq_str.split(','):
                    x = x.strip()
                    if x.lstrip('-').isdigit():
                        sequence.append(int(x))
                
                return {
                    'id': a_number,
                    'name': res.get('name', 'Unknown'),
                    'sequence': sequence[:15],
                }

            if isinstance(data, list) and len(data) > 0:
                if data[0] is None: return {'error': 'Null result'}
                return parse_result(data[0])
                
            if isinstance(data, dict):
                results = data.get('results')
                if results and isinstance(results, list) and len(results) > 0:
                    if results[0] is None: return {'error': 'Null result'}
                    return parse_result(results[0])
            
            return {'error': 'No results found', 'id': a_number}
    except Exception as e:
        return {'error': str(e), 'id': a_number}


# ==============================================================================
# Synthesizer
# ==============================================================================

@dataclass(frozen=True)
class Expr:
    pass

@dataclass(frozen=True)
class Var(Expr):
    name: str
    def __repr__(self): return self.name

@dataclass(frozen=True)
class Const(Expr):
    val: Any
    def __repr__(self): return repr(self.val)

@dataclass(frozen=True)
class App(Expr):
    func: str
    args: tuple
    def __repr__(self): 
        return f"{self.func}({', '.join(repr(a) for a in self.args)})"

class Interpreter:
    PRIMS = {
        'add': lambda a, b: a + b,
        'mul': lambda a, b: a * b,
        'sub': lambda a, b: a - b,
        'div': lambda a, b: a // b if b != 0 else 0,
        'mod': lambda a, b: a % b if b != 0 else 0,
        'pow': lambda a, b: a ** b if 0 <= b <= 5 and abs(a) < 100 else 0, # Restricted pow
    }
    
    def run(self, expr, env):
        try:
            return self._eval(expr, env, 200)
        except:
            return None
    
    def _eval(self, expr, env, gas):
        if gas <= 0: return None
        if isinstance(expr, Var): return env.get(expr.name)
        if isinstance(expr, Const): return expr.val
        if isinstance(expr, App):
            fn = expr.func
            if fn in self.PRIMS:
                args = [self._eval(a, env, gas-1) for a in expr.args]
                if None in args: return None
                try: return self.PRIMS[fn](*args)
                except: return None
        return None

class SequenceSynthesizer:
    def __init__(self, max_size=5, timeout=10.0):
        self.max_size = max_size
        self.timeout = timeout
        self.interp = Interpreter()
    
    def synthesize(self, sequence: List[int]) -> Tuple[Optional[Expr], Dict]:
        start = time.time()
        stats = {'tested': 0, 'generated': 0, 'timeout': False}
        
        # Try both 0-indexed and 1-indexed
        ios_0 = [{"input": i, "output": v} for i, v in enumerate(sequence[:6])]
        ios_1 = [{"input": i+1, "output": v} for i, v in enumerate(sequence[:6])]
        
        all_exprs = self._enumerate('n', self.max_size)
        stats['generated'] = len(all_exprs)
        
        for expr in all_exprs:
            if time.time() - start > self.timeout:
                stats['timeout'] = True
                return None, stats, "TIMEOUT"
            
            stats['tested'] += 1
            
            # Check 0-indexed
            if self._check(expr, 'n', ios_0):
                return expr, stats, "0-indexed"
            
            # Check 1-indexed
            if self._check(expr, 'n', ios_1):
                return expr, stats, "1-indexed"
        
        return None, stats, "NOT_FOUND"
    
    def _enumerate(self, var_name, max_size):
        exprs = {1: [Var(var_name), Const(0), Const(1), Const(2), Const(3), Const(10)]}
        
        for size in range(2, max_size + 1):
            exprs[size] = []
            
            # NO PRUNING - True Brute Force
            # Generates HUGE number of expressions. Be ready.
            
            for s1 in range(1, size):
                s2 = size - s1
                # Use ALL expressions from previous levels
                for e1 in exprs.get(s1, []):
                    for e2 in exprs.get(s2, []):
                        for p in ['add', 'mul', 'sub', 'div', 'pow']:
                            exprs[size].append(App(p, (e1, e2)))
        
        result = []
        for s in range(1, max_size + 1):
            result.extend(exprs.get(s, []))
        return result
    
    def _check(self, expr, var_name, ios):
        for io in ios:
            result = self.interp.run(expr, {var_name: io['input']})
            if result != io['output']:
                return False
        return True


def main():
    print("\n" + "=" * 70)
    print("  HARDCORE OEIS BENCHMARK - RANDOM MODE")
    print("  Fetches 20 completely random sequences from OEIS.")
    print("  Most will fail. This is EXPECTED.")
    print("=" * 70)
    
    # Reduced to 4 for feasibility without pruning (~800k candidates)
    synth = SequenceSynthesizer(max_size=4, timeout=5.0)
    
    solved = 0
    failed = 0
    errors = 0
    
    for i in range(20):
        print(f"\n[{i+1}/20]")
        
        # RANDOM FETCH
        data = fetch_random_oeis()
        
        if 'error' in data:
            print(f"  [ERROR] {data['error']}")
            errors += 1
            continue
        
        print(f"  ID: {data['id']}")
        name = data.get('name', 'Unknown')
        print(f"  Name: {name[:60]}..." if len(name)>60 else f"  Name: {name}")
        
        seq = data.get('sequence', [])
        if not seq or len(seq) < 5:
            print("  [SKIP] Sequence too short")
            continue
            
        print(f"  Sequence: {seq[:7]}...")
        
        # Synthesize
        result, stats, mode = synth.synthesize(seq)
        
        if result:
            print(f"  [SUCCESS] ✅ {result} ({mode})")
            solved += 1
        else:
            print(f"  [FAILED] ❌ ({mode})")
            failed += 1
        
        print(f"  Stats: tested={stats['tested']}, generated={stats['generated']}")
        
        time.sleep(1.5)
    
    total = solved + failed
    print(f"\n{'='*70}")
    print("  FINAL RANDOM RESULTS")
    print(f"{'='*70}")
    print(f"  Attempts: {total}")
    print(f"  ✅ Solved: {solved} ({100*solved/max(1,total):.1f}%)")
    print(f"  ❌ Failed: {failed} ({100*failed/max(1,total):.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
