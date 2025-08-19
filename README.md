
# Autobot Task (Sim-1 Autonomous Robot)

Minimal instructions to run the project and open the hosted UI.

## Hosts
- Simulator (default): http://localhost:5000
- App UI (Streamlit): http://localhost:8501

## Setup (Windows PowerShell)
```powershell
cd "C:\Users\abhay sharma\Desktop\Terraf Task"
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run (CLI)
- Level 1 (static):
```powershell
python .\main.py --base-url http://localhost:5000 --steps-per-run 1500 --step-pixels 8 --repeat 1 --corners tl tr br bl
```
- Level 2 (moving obstacles):
```powershell
python .\main.py --base-url http://localhost:5000 --moving-obstacles --obstacle-speed 1.0 --steps-per-run 1500
```
- Level 3 (plot):
```powershell
python .\main.py --plot-speed-curve
```

## Run (UI)
```powershell
streamlit run .\ui.py
```
Open: http://localhost:8501

## Notes
- Uses `/capture` for frames and `/move_rel` (fallback `/move` with `relative=true`).
- Results append to `results.csv`; Level 3 graph saved as `obstacle_speed_vs_collisions.png`.
- For offline demo: `python .\main.py --mock` or enable "Use Mock Simulator" in the UI.
