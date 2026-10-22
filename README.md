# Gas-Turbine Compressor Degradation from Sensor Data (MMA3001 Project)

## Purpose
This project estimates the compressor decay coefficient (kMc) of a simulated frigate gas-turbine propulsion plant from steady-state sensor measurements. Compressor degradation cannot be measured directly in service, so condition-based maintenance depends on inferring it from temperatures, pressures, speeds and fuel flow. I compare a mean-prediction baseline with linear regression, decision-tree regression and support-vector regression, and study how sensor selection and model settings affect accuracy and runtime. It is a research prototype on simulated data and is not a certified monitoring tool.

## Inputs and outputs

**Input:** one row per operating case, 12 numeric columns.

| Column | Meaning | Unit |
|---|---|---|
| lp | Lever position (operating setting) | dimensionless, 1.138 to 9.3 |
| GTT | Gas turbine shaft torque | kN m |
| GTn | Gas turbine speed | rpm |
| GGn | Gas generator speed | rpm |
| Ts | Propeller torque (starboard) | kN (as labelled) |
| T48 | HP turbine exit temperature | labelled C; values look like kelvin |
| T2 | Compressor outlet temperature | labelled C; values look like kelvin |
| P48 | HP turbine exit pressure | bar |
| P2 | Compressor outlet pressure | bar |
| Pexh | Exhaust gas pressure | bar |
| TIC | Turbine injection control | % |
| mf | Fuel flow | kg/s |

Not used as inputs: T1 and P1 (constant), Tp (identical to Ts), v (one-to-one with lp).

**Output:** an estimate of kMc, the compressor decay coefficient, dimensionless, from 0.95 (most degraded) to 1.0 (healthy). Models are scored with MAE, RMSE and R2 on held-out data.

**Invalid input handling:** missing values or the wrong number of columns raise an error when the data is loaded. Inputs outside the training ranges are outside the supported domain. The turbine decay coefficient kMt is not available as an input.

## Repository structure
- `src/gtdeg/` - source code
- `tests/` - pytest tests
- `notebooks/` - exploration and experiments
- `docs/` - generated HTML documentation and data notes
- `data/raw/` - dataset (not included; see Data section)

## Setup
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Data
Coraddu, A., Oneto, L., Ghio, A., Savio, S., Anguita, D., & Figari, M. (2014).
Condition Based Maintenance of Naval Propulsion Plants [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5K31K

Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

The data is not included in this repository. Download the zip from
https://archive.ics.uci.edu/dataset/316/condition+based+maintenance+of+naval+propulsion+plants
and extract it into `data/raw/`, so that `data/raw/UCI CBM Dataset/data.txt` exists.

## Usage
(To be completed.)

## Testing
    pytest

## Licence
MIT