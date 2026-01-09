#!/usr/bin/env python3
"""
REAL-TIME OEIS BENCHMARK - NO CHEATING! (FIXED)
================================================
Fetches sequences directly from oeis.org API.
"""
import urllib.request
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional


def fetch_oeis_sequence(a_number: str) -> Dict[str, Any]:
    """Fetch sequence from OEIS in real-time."""
    url = f"https://oeis.org/search?fmt=json&q=id:{a_number}"
    print(f"  [FETCHING] {url}")
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode())
            
            # Helper to extract from result object
            def parse_result(res):
                seq_str = res.get('data', '')
                sequence = []
                for x in seq_str.split(','):
                    x = x.strip()
                    if x.lstrip('-').isdigit():
                        sequence.append(int(x))
                
                formula = ''
                if res.get('formula') and isinstance(res['formula'], list):
                    formula = res['formula'][0] if res['formula'] else ''
                
                return {
                    'id': a_number,
                    'name': res.get('name', 'Unknown'),
                    'sequence': sequence[:15],
                    'formula': formula,
                }

            # Handle list response (what we saw in debug)
            if isinstance(data, list) and len(data) > 0:
                return parse_result(data[0])
                
            # Handle dict response (old format?)
            if isinstance(data, dict):
                results = data.get('results')
                if results and isinstance(results, list) and len(results) > 0:
                    return parse_result(results[0])
            
            return {'error': 'No results found', 'id': a_number}
    except Exception as e:
        return {'error': str(e), 'id': a_number}


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
        'div': lambda a, b: a // b if b != 0 else None,
        'mod': lambda a, b: a % b if b != 0 else None,
    }
    
    def run(self, expr, env):
        try:
            return self._eval(expr, env, 200)
        except:
            return None
    
    def _eval(self, expr, env, gas):
        if gas <= 0:
            return None
        
        if isinstance(expr, Var):
            return env.get(expr.name)
        if isinstance(expr, Const):
            return expr.val
        
        if isinstance(expr, App):
            fn = expr.func
            if fn in self.PRIMS:
                args = [self._eval(a, env, gas-1) for a in expr.args]
                if None in args:
                    return None
                try:
                    return self.PRIMS[fn](*args)
                except:
                    return None
        return None


class SequenceSynthesizer:
    def __init__(self, max_size=4, timeout=5.0):
        self.max_size = max_size
        self.timeout = timeout
        self.interp = Interpreter()
    
    def synthesize(self, sequence: List[int]) -> Tuple[Optional[Expr], Dict]:
        start = time.time()
        stats = {'tested': 0, 'generated': 0, 'timeout': False}
        
        # Use first 7 terms as I/O examples
        ios = [{"input": i, "output": v} for i, v in enumerate(sequence[:7])]
        
        all_exprs = self._enumerate('n', self.max_size)
        stats['generated'] = len(all_exprs)
        
        for expr in all_exprs:
            if time.time() - start > self.timeout:
                stats['timeout'] = True
                return None, stats
            
            stats['tested'] += 1
            if self._check(expr, 'n', ios):
                return expr, stats
        
        return None, stats
    
    def _enumerate(self, var_name, max_size):
        exprs = {1: [Var(var_name), Const(0), Const(1), Const(2), Const(3)]}
        
        for size in range(2, max_size + 1):
            exprs[size] = []
            for s1 in range(1, size):
                s2 = size - s1
                for e1 in exprs.get(s1, [])[:40]:
                    for e2 in exprs.get(s2, [])[:40]:
                        for p in ['add', 'mul', 'sub', 'div', 'mod']:
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
    print("  REAL-TIME OEIS BENCHMARK - LIVE FROM oeis.org")
    print("  NO hardcoded data - fetching in real-time!")
    print("=" * 70)
    
    # OEIS A-numbers to test
    TEST_SEQUENCES = [
        "A000027",  # Natural numbers: 1,2,3,4,...
        "A000290",  # Squares: 0,1,4,9,16,...
        "A000045",  # Fibonacci: 0,1,1,2,3,5,...
        "A000079",  # Powers of 2: 1,2,4,8,16,...
        "A000217",  # Triangular: 0,1,3,6,10,...
        "A005843",  # Even numbers: 0,2,4,6,...
        "A005408",  # Odd numbers: 1,3,5,7,...
        "A000578",  # Cubes: 0,1,8,27,64,...
        "A000004",  # All zeros: 0,0,0,0,...
        "A000012",  # All ones: 1,1,1,1,...
    ]
    
    synth = SequenceSynthesizer(max_size=4, timeout=5.0)
    
    solved = 0
    failed = 0
    errors = 0
    
    for a_num in TEST_SEQUENCES:
        print(f"\n{'='*60}")
        print(f"[{a_num}]")
        
        # REAL-TIME FETCH
        data = fetch_oeis_sequence(a_num)
        
        if 'error' in data:
            print(f"  [FETCH ERROR] {data['error']}")
            errors += 1
            continue
        
        print(f"  Name: {data['name'][:60]}")
        print(f"  Sequence (from OEIS): {data['sequence'][:7]}")
        
        # Synthesize
        result, stats = synth.synthesize(data['sequence'])
        
        if result:
            print(f"  [SUCCESS] ✅ {result}")
            solved += 1
        else:
            reason = "timeout" if stats.get('timeout') else "not found"
            print(f"  [FAILED] ❌ ({reason})")
            failed += 1
        
        print(f"  Stats: tested={stats['tested']}, generated={stats['generated']}")
        
        # Don't hammer the OEIS server
        time.sleep(1.5)
    
    total = solved + failed
    print(f"\n{'='*70}")
    print("  FINAL RESULTS - REAL-TIME FROM oeis.org")
    print(f"{'='*70}")
    print(f"  Fetch errors: {errors}")
    print(f"  ✅ Solved: {solved}/{total} ({100*solved/max(1,total):.1f}%)")
    print(f"  ❌ Failed: {failed}/{total} ({100*failed/max(1,total):.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
