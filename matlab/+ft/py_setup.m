function py_setup(root)
%FT.PY_SETUP  use the project's Python venv (final/.venv) out of process, with final/ on sys.path
pe = pyenv;
exe = fullfile(root, '.venv', 'Scripts', 'python.exe');
if pe.Status == "NotLoaded"
    pyenv('Version', exe, 'ExecutionMode', 'OutOfProcess');
end
P = py.sys.path;
if count(P, root) == 0, insert(P, int32(0), root); end
end
