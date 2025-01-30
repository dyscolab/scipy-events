# Changelog

## 0.4.2 - 2025-01-30

### Fixed

- Fixed issue with `ChangeAt` events and `t_eval` (issue #1).

## 0.4.1 - 2025-01-16

### Fixed

- `Progress` now works for functions requiring `*args`.

## 0.4.0 - 2024-12-17

### Changed

- `ChangeAt` and `ChangeWhen` events now allow to change parameters.
  This changed the signature of `Change` functions
  from returning only the new initial conditions `y`
  to returning a tuple of the initial conditions and parameters `(y, args)`.

## 0.3.1 - 2024-12-09

### Changed

- Remove `*args` from `Condition` and `Change` protocols.
  Both of these are now correct for `pyright`:
  ```python
  def cond(t, y): ...
  def cond(t, y, *args): ...
  ```

## 0.3.0 - 2024-12-09

### Added

- `Events` type union of all event types accepted by `solve_ivp`.
- Allow passing `args` to Conditions and Change functions.

### Changed

- Change events to `kw_only` and `frozen` dataclasses.

## 0.2.0 - 2024-12-09

### Changed

- Rename `core` module to `_core`, to reflect its private nature.
- Add a `factor` to `SmallDerivatives` to alter the default tolerances taken from the solver.

### Added

- `Event` class to create events with terminal conditions and directions.
- `ChangeWhen` and `ChangeAt` events that continue the solution after modifying the state `y`.

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
