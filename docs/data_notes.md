\# Data notes



Observations about the UCI naval propulsion plant dataset that affect how it is used. Where a statement is my own inference, it is marked as such.



\## Temperature units

\- `Features.txt` labels the temperature columns (T48, T1, T2) as degrees C.

\- `T1` (compressor inlet temperature) is constant at 288. As degrees C that would be ambient air at 288 C, which is not physical. As kelvin it is 15 C, the standard sea-level temperature.

\- Check on `T2`: for ideal air compression with gamma = 1.4 (my assumption), outlet temperature is about T1 x (P2/P1)^0.286. Over the pressure ratios in the data (about 5.8 to 23.2) this gives roughly 477 K to 707 K from a 288 K start. The observed `T2` runs from about 540 to 789, just above the ideal values, as expected for a real compressor. If `T1` were 288 C (561 K), the ideal outlet at the lowest pressure ratio would exceed the observed `T2`, which is impossible.

\- Inference (not stated by the dataset authors): the temperatures are in kelvin and are mislabelled as C. I have not relabelled anything in the code. The models are unaffected, because shifting a temperature by a constant does not change a linear model, a decision tree or a scaled SVR. It would matter for any physical feature built from temperatures.



\## Torque units

\- `Ts` and `Tp` (propeller torque) are labelled kN, while `GTT` (gas turbine shaft torque) is labelled kN m. A torque would normally be in kN m, so the label may be loose. I could not verify this, so the labels are reported as given.



\## Columns not used as model inputs

\- `T1`, `P1`: constant, so they carry no information.

\- `Tp`: identical to `Ts`.

\- `v`: one-to-one with `lp` (each lever position has exactly one speed).

These are checked by tests in `tests/test\_load\_data.py`.



\## Structure of the data

\- The 11,934 rows are a complete grid: 9 lever positions x 51 compressor decay values x 26 turbine decay values, with no duplicates (checked in `grid\_report`). The rows are not independent random samples, which matters for how the data is split for validation.

\- `Pexh` is constant at the lowest lever position (1.138), so its correlation with the decay coefficients is undefined there. Averages for `Pexh` use 8 lever positions, not 9.



\## Why pooled correlations are weak

\- Across all lever positions together, no sensor has a correlation above 0.05 with `kMc`. Within a single lever position, correlations reach about 0.87 (`GGn`). Operating-point variation dominates the pooled readings. These are linear, absolute, averaged correlations, so they suggest which sensors matter and do not prove it.



\## Dataset documentation (README.txt inside the zip)

\- Noise: the README does not mention measurement noise, so I cannot say whether noise was added to the simulated sensor values.

\- Generation: simulator of a frigate gas-turbine plant (propeller, hull, gas turbine, gearbox and controller blocks), steady-state measurements, uniform grid of spacing 0.001 over kMc in \[0.95, 1] and kMt in \[0.975, 1], speeds from 3 to 27 knots in steps of 3 (the file says "tree knots", a typo).

\- Units: the README lists the same units as Features.txt (temperatures in C, torques in kN), so the temperature and torque concerns above are not resolved by the authors' documentation.

\- Documentation errors: the README header names a different dataset (Human Activity Recognition Using Smartphones) and has typos. This makes a label error plausible but does not prove one.

\- Licence: the UCI page states CC BY 4.0. The README in the zip requires citing the Coraddu et al. paper and prohibits commercial use. Citing both and keeping use non-commercial satisfies both.

\- Features are not normalised (README), so scaling is done inside the models.

