# Complete WS-DREAM QoS Results

These experiments use the deterministic offline demo controller. They validate
the complete data and evaluation pipeline, but they are not live-LLM evidence.

## Coverage

- Dataset #1: 339 users, 5,825 services, one static snapshot, 2 QoS attributes.
- Dataset #2: 142 users, 4,500 services, 64 time slices, 2 QoS attributes.
- Full static protocol: 30 pools per task, 20 services per pool, 2 task profiles.
- Temporal protocol: the same 30 pools tracked through all 64 observed slices.

## Dataset #1 Static Mean Regret

| condition          |   regret |
|:-------------------|---------:|
| lookup_table       | 0.000008 |
| agent_weights_only | 0.000190 |
| agent_full         | 0.001210 |
| global_fixed       | 0.013037 |

### Dataset #1 by task profile

| task_key                | condition          |   regret |
|:------------------------|:-------------------|---------:|
| iot_telemetry_ingestion | lookup_table       | 0.000016 |
| iot_telemetry_ingestion | agent_weights_only | 0.000036 |
| iot_telemetry_ingestion | agent_full         | 0.001077 |
| iot_telemetry_ingestion | global_fixed       | 0.016413 |
| streaming               | lookup_table       | 0.000000 |
| streaming               | agent_weights_only | 0.000344 |
| streaming               | agent_full         | 0.001344 |
| streaming               | global_fixed       | 0.009661 |

## Dataset #2 Static Mean Regret

| condition          |   regret |
|:-------------------|---------:|
| lookup_table       | 0.000210 |
| agent_weights_only | 0.002682 |
| agent_full         | 0.002729 |
| global_fixed       | 0.015164 |

### Dataset #2 static results by task profile

| task_key                | condition          |   regret |
|:------------------------|:-------------------|---------:|
| iot_telemetry_ingestion | lookup_table       | 0.000245 |
| iot_telemetry_ingestion | agent_weights_only | 0.000664 |
| iot_telemetry_ingestion | agent_full         | 0.000674 |
| iot_telemetry_ingestion | global_fixed       | 0.016606 |
| streaming               | lookup_table       | 0.000176 |
| streaming               | agent_weights_only | 0.004699 |
| streaming               | agent_full         | 0.004785 |
| streaming               | global_fixed       | 0.013722 |

## Dataset #2 Temporal Results

| condition          |   mean_regret |   top1_accuracy |   mean_switches |
|:-------------------|--------------:|----------------:|----------------:|
| lookup_dynamic     |      0.000142 |        0.926302 |       13.283333 |
| lookup_static      |      0.002640 |        0.742448 |        0.000000 |
| agent_full         |      0.005520 |        0.836979 |       14.483333 |
| agent_weights_only |      0.006590 |        0.775781 |       13.600000 |
| global_dynamic     |      0.020500 |        0.589844 |        9.500000 |
| global_static      |      0.024532 |        0.512760 |        0.000000 |

### Dataset #2 temporal results by task profile

| task_key                | condition          |   mean_regret |   top1_accuracy |   mean_switches |
|:------------------------|:-------------------|--------------:|----------------:|----------------:|
| iot_telemetry_ingestion | lookup_dynamic     |      0.000273 |        0.869271 |       23.433333 |
| iot_telemetry_ingestion | agent_full         |      0.000282 |        0.907813 |       21.233333 |
| iot_telemetry_ingestion | agent_weights_only |      0.000706 |        0.789583 |       19.100000 |
| iot_telemetry_ingestion | lookup_static      |      0.002853 |        0.592187 |        0.000000 |
| iot_telemetry_ingestion | global_static      |      0.014345 |        0.458854 |        0.000000 |
| iot_telemetry_ingestion | global_dynamic     |      0.020612 |        0.501042 |        9.500000 |
| streaming               | lookup_dynamic     |      0.000011 |        0.983333 |        3.133333 |
| streaming               | lookup_static      |      0.002427 |        0.892708 |        0.000000 |
| streaming               | agent_full         |      0.010757 |        0.766146 |        7.733333 |
| streaming               | agent_weights_only |      0.012474 |        0.761979 |        8.100000 |
| streaming               | global_dynamic     |      0.020387 |        0.678646 |        9.500000 |
| streaming               | global_static      |      0.034719 |        0.566667 |        0.000000 |

## Interpretation

Dynamic baselines re-rank every time slice and therefore isolate ranking quality.
Static baselines hold the initial recommendation and represent a deployment that
is not automatically re-evaluated. Agent conditions re-evaluate every slice.
Regret uses researcher-authored task weights as the reference, not external
ground truth.

## Availability Audit

The official Zenodo record contains Dataset #1 and Dataset #2. The historical
ICWS 2012 and Cloud 2013 download links currently return HTTP 404 and are not
included in the Zenodo record. Log and review datasets are separate research
modalities and do not contain the QoS matrices required by this selection method.
