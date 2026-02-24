import sys
import os
import random

# FIX: Use a flag so later code can safely check availability
# instead of crashing with NameError when rs_machine is not installed.
RS_MACHINE_AVAILABLE = False
try:
    sys.path.append(os.getcwd())
    import rs_machine
    RS_MACHINE_AVAILABLE = True
    print(f"[TEST] rs_machine imported: {rs_machine}")
except ImportError:
    print("[TEST] rs_machine NOT found. Rust VM execution steps will be skipped.")

from neuro_genetic_synthesizer import RustCompiler, NeuroInterpreter, BSApp, BSVal, BSVar


def test_rust_jit():
    print("=" * 60)
    print("TEST: Rust JIT Compiler & Execution")
    print("=" * 60)

    # 1. Create a simple expression: add(mul(n, 2), 5)  =>  n*2 + 5
    #    Input n=3 -> expected 11
    expr = BSApp('add', (
        BSApp('mul', (BSVar('n'), BSVal(2))),
        BSVal(5)
    ))
    print(f"Expression: {expr}")

    # 2. Compile to Rust bytecode
    compiler = RustCompiler()
    try:
        insts = compiler.compile(expr)
        print(f"[OK] Compiled to {len(insts)} instructions.")
        for i, inst in enumerate(insts):
            print(f"  {i}: {inst.op} {inst.a} {inst.b} {inst.c}")
    except Exception as e:
        print(f"[FAIL] Compilation error: {e}")
        return

    # 3. Execute on Rust VM (skip gracefully if not available)
    if not RS_MACHINE_AVAILABLE:
        print("[SKIP] rs_machine not available — skipping Rust VM execution step.")
    else:
        try:
            vm = rs_machine.VirtualMachine(100, 64, 16)
            st = vm.execute(insts, [3.0])
            res = st.regs[0]
            print(f"[rust] Result(n=3): {res}")
            expected = 11.0
            if abs(res - expected) < 1e-9:
                print("[PASS] Rust result matches expected.")
            else:
                print(f"[FAIL] Rust result {res} != expected {expected}")
        except Exception as e:
            print(f"[FAIL] Runtime error: {e}")

    # 4. Test MOD operator compilation
    print("\n[TEST] MOD operator")
    expr_mod = BSApp('mod', (BSVar('n'), BSVal(2)))
    compiler_mod = RustCompiler()
    try:
        insts_mod = compiler_mod.compile(expr_mod)
    except Exception as e:
        print(f"[FAIL] MOD compilation error: {e}")
        return

    if insts_mod is not None:
        print(f"[PASS] Compilation succeeded for MOD op: {insts_mod}")
        if not RS_MACHINE_AVAILABLE:
            print("[SKIP] rs_machine not available — skipping MOD execution.")
        else:
            try:
                vm_mod = rs_machine.VirtualMachine(100, 64, 16)
                st_mod = vm_mod.execute(insts_mod, [3.0])  # 3 % 2 = 1
                res_mod = st_mod.regs[0]
                if abs(res_mod - 1.0) < 1e-9:
                    print(f"[PASS] MOD result correct: {res_mod}")
                else:
                    print(f"[FAIL] MOD result {res_mod} != expected 1.0")
            except Exception as e:
                print(f"[FAIL] MOD execution error: {e}")
    else:
        print("[FAIL] Compilation returned None for MOD op.")


if __name__ == "__main__":
    test_rust_jit()
