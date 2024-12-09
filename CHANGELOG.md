# Changelog

## 0.1.2 - 2024-12-06

### Fixed

- Events `SmallDerivatives` with solver default tolerances was not working with LSODA method,
  which did not expose its tolerances.

## 0.1.1 - 2024-12-06

### Fixed

- Events `SmallDerivatives` was not working with LSODA and BDF methods,
  which did not expose the last function evaluation `f`.

## 0.1.0 - 2024-12-05

### Added

- `SmallDerivatives` event to solve upto a steady state.
- `Progress` event to monitor the current time with a progress bar.
