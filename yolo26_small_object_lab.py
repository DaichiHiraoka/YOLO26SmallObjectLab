"""Double-clickable entry point for the YOLO26 small-object lab."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _use_project_environment() -> None:
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)
    venv_python = project_dir / ".venv" / "Scripts" / "python.exe"
    if (
        venv_python.exists()
        and Path(sys.executable).resolve() != venv_python.resolve()
    ):
        os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve())])
    if venv_python.exists():
        return

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "Python環境がありません。PCへuvをインストールしてから再実行してください。"
        )
    subprocess.run([uv, "sync"], cwd=project_dir, check=True)
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve())])


if __name__ == "__main__":
    try:
        _use_project_environment()
        from small_object_lab.launcher import main

        raise SystemExit(main())
    except Exception as error:
        try:
            import tkinter
            from tkinter import messagebox

            root = tkinter.Tk()
            root.withdraw()
            messagebox.showerror("YOLO26 Small Object Lab", str(error))
            root.destroy()
        except Exception:
            print(f"エラー: {error}", file=sys.stderr)
        raise SystemExit(1)
