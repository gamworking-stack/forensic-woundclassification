# Ticket T10: Runbook

- **Ticket ID**: `T10`
- **Phase**: Phase 4 (HPC)
- **Spec Reference**: [ROADMAP.md](../ROADMAP.md) §5 T10
- **Status**: Completed

---

## 1. Goal

`runbook.md` lets another person connect, run and reproduce every result.

## 2. Coverage

| ROADMAP item | Runbook section |
|---|---|
| Connection (`ssh`, keys) | MedCMU-HPC Raptor → Connecting (CMU network/VPN, laptop key, `tmux`) |
| Environment | MedCMU-HPC Raptor (container), T05 training library (weights cache) |
| Data locations | Data locations (Raptor) |
| Submit | Training jobs, AlexNet ensemble, Submit everything (S01–S16, T08b grid) |
| Monitor | Checking progress and failures, Watching jobs live |
| Resume | Training jobs (`--resume`), Submit everything (rerun after timeout) |
| Where results go | Training jobs (output folders), Evaluation (T08), Summary (T15), Code to Raptor, results back |
| Support contact | MedCMU-HPC Raptor |

## 3. Acceptance Check (2026-10-14)

Every section's command was used for the current results (T03, T05–T08b, T11–T13, T15), and the gaps found while doing so (network requirement, where to run `ssh-copy-id`, fetching results, failure checks) are now in the runbook. ✔
