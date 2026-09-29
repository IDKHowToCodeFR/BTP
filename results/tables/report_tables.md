### Table 1: Performance in Stable Environments

| Condition                       |   Mean_Regret |   Std_Regret | Latency_s   |   API_Calls |
|:--------------------------------|--------------:|-------------:|:------------|------------:|
| Static Baseline (Global)        |        0.0334 |       0.0317 | 0.00s       |           0 |
| Static Baseline (Task-Specific) |        0.0332 |       0.0337 | 0.00s       |           0 |
| Agent (Weights Only)            |        0.0776 |       0.1095 | 7.90s       |           1 |
| Agent (Full Autonomy)           |        0.0319 |       0.0468 | 0.00s       |           0 |
| embedding_knn                   |        0.029  |       0.0335 | 0.00s       |           0 |
| uniform                         |        0.0334 |       0.0317 | 0.00s       |           0 |

### Table 2: Adaptation to QoS Drift (Failover)

| Condition                       |   Mean_Lag_Rounds |
|:--------------------------------|------------------:|
| Agent (Full Autonomy)           |                 0 |
| Agent (Weights Only)            |                 0 |
| Static Baseline (Global)        |                 0 |
| Static Baseline (Global)        |                 2 |
| Static Baseline (Task-Specific) |                 0 |
| Static Baseline (Task-Specific) |                 2 |

### Table 3: Autonomy Ablation (Regret by Domain)

| Task Domain   |   Agent (Full Autonomy) |   Agent (Weights Only) |
|:--------------|------------------------:|-----------------------:|
| Held Out      |                  0.0472 |                 0.0876 |
| Profile       |                  0.0062 |                 0.061  |

