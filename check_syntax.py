import py_compile, sys
for f in ['src/ui/workers.py', 'src/ui/main_window.py']:
    try:
        py_compile.compile(f, doraise=True)
        print(f'{f}: OK')
    except py_compile.PyCompileError as e:
        print(f'{f}: ERROR - {e}')
        sys.exit(1)
