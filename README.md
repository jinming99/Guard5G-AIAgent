# 5G-Spector FBS Detection System with LLM Integration

## Overview

This project implements an LLM-driven Fake Base Station (FBS) detection system for 5G networks, built on top of the O-RAN architecture. It integrates three main components:

1. **MobiFlow-Auditor**: Network telemetry collection and monitoring
2. **MobieXpert**: P-BEST rule-based detection engine
3. **OAI-5G**: OpenAirInterface 5G RAN with FBS simulation capabilities

The system uses Large Language Models (LLMs) to dynamically generate detection rules, design experiments, and analyze network security threats.

## Architecture

```
┌──────────────┐  REST API  ┌────────────────┐
│ LLM Agent    │◄──────────►│ MobiFlow-      │
│ (DSPy)       │            │ Auditor        │◄─── KPM/Telemetry
└──────────────┘            └────────────────┘
       │                            │
       │ POST /rules               │ SDL (Redis)
       ▼                           ▼
┌──────────────┐            ┌────────────────┐
│ MobieXpert   │◄───────────│ Detection      │
│              │ Hot-reload │ Rules          │
└──────────────┘            └────────────────┘
       ▲                           
       │ E2 Interface              
       │                           
┌──────────────┐    XRAN    ┌──────────────┐
│ OAI gNB (DU) │◄───────────►│ O-RU/FBS Sim │
└──────────────┘            └──────────────┘
```

## Repository Structure

```
workspace/
├── MobiFlow-Auditor/          # Telemetry collection xApp
│   └── src/
│       ├── llm_control_api.py # NEW: REST API for LLM
│       └── [modifications]     # Enhanced with FBS fields
├── MobieXpert/                # Detection engine xApp  
│   └── src/
│       ├── pypbest/
│       │   └── llm_rule_patch.py # NEW: Rule hot-reload
│       └── experiments/       # NEW: Rule templates
├── OAI-5G/                    # 5G Base station
│   └── tools/
│       └── fbs_scenarios/     # NEW: FBS simulation
│           ├── launch_fbs.sh
│           └── fake_gnb.cfg
└── llm_fbs_utils/            # NEW: LLM integration
    ├── llm_driver.py         # Main LLM orchestrator
    ├── llm_control_api_client.py
    ├── scenario_runner.py
    ├── dataset_playback.py
    ├── prompt_templates/
    ├── eval_scripts/
    ├── docker-compose.yml
    └── requirements.txt
```

## Installation

### Prerequisites

- Docker & Docker Compose
- Python 3.8+
- Redis
- DPDK 20.11 (for O-RAN fronthaul)
- Git

### Clone Repositories

```bash
# Clone the three main repositories
git clone https://github.com/5GSEC/MobiFlow-Auditor.git
git clone https://github.com/5GSEC/MobieXpert.git
git clone https://github.com/5GSEC/OAI-5G.git

# Create LLM utils directory
mkdir llm_fbs_utils
cd llm_fbs_utils
```

### Apply Modifications

1. **MobiFlow-Auditor**: Add the new files and modifications as documented
2. **MobieXpert**: Add rule patching capability
3. **OAI-5G**: Add FBS scenario tools

### Install Dependencies

```bash
cd llm_fbs_utils
pip install -r requirements.txt
```

### Configure Environment

```bash
# Set environment variables
export OPENAI_API_KEY="your-api-key"  # For LLM
export OAI_HOST="localhost"
export REDIS_HOST="localhost"
```

## Quick Start

### 1. Start Services with Docker Compose

```bash
cd llm_fbs_utils
docker-compose up -d
```

This starts:
- Redis SDL
- MobiFlow-Auditor with LLM API
- MobieXpert with rule reload API
- OAI gNB
- Optional: LLM Orchestrator

### 2. Verify Services

```bash
# Check health
curl http://localhost:8090/health  # Auditor
curl http://localhost:8091/rules   # Expert

# Run component tests
python eval_scripts/test_query_api.py
python eval_scripts/test_rule_reload.py
```

### 3. Run FBS Detection

#### Manual Mode

```bash
# Start legitimate gNB
./OAI-5G/tools/fbs_scenarios/launch_fbs.sh start

# Inject test data
python dataset_playback.py generate --duration 60

# Monitor detection
python llm_driver.py detect
```

#### Automated Mode

```bash
# Run complete scenario
python llm_driver.py experiment --hypothesis "FBS causes auth failures"
```

### 4. Run Evaluation

```bash
# Run complete evaluation pipeline
python eval_scripts/evaluate_pipeline.py

# Run specific test
bash eval_scripts/test_fbs_scenario.sh
```

## Usage Examples

### Query Network State

```python
from llm_control_api_client import get_kpm, get_mobiflow, get_stats

# Get KPM data for specific UE
kpm_data = get_kpm("001010123456789")

# Get recent MobiFlow records
records = get_mobiflow(n=100)

# Get statistics
stats = get_stats()
print(f"FBS Detections: {stats['fbs_detections']}")
```

### Generate Detection Rules with LLM

```python
from llm_driver import LLMOrchestrator

orchestrator = LLMOrchestrator()

# Analyze current state
analysis = orchestrator.run_detection_cycle()

if analysis['fbs_detected']:
    print(f"FBS detected with confidence: {analysis['confidence']}")
```

### Run FBS Scenario

```python
from scenario_runner import run_scenario, ScenarioLibrary

# Use predefined scenario
scenario = ScenarioLibrary.basic_fbs_attack()
run_scenario(scenario)

# Or load from file
run_scenario("experiments/scenarios/advanced_fbs_attack.json")
```

### Hot-Reload Detection Rules

```python
from llm_control_api_client import post_rule

rule_yaml = """
rules:
  - name: "Custom_FBS_Rule"
    condition:
      field: "attach_failures"
      gte: 5
    action:
      type: "alert"
      severity: "high"
"""

result = post_rule(rule_yaml)
print(f"Rule loaded: {result['status']}")
```

## Evaluation Metrics

The system evaluates:

1. **Detection Accuracy**
   - True Positive Rate
   - False Positive Rate
   - Detection Time

2. **Performance**
   - Throughput (records/sec)
   - Latency
   - Resource Usage

3. **Rule Quality**
   - Coverage
   - Precision
   - Adaptability

## Configuration Files

### Docker Compose

See `llm_fbs_utils/docker-compose.yml` for service orchestration.

### FBS Configuration

Edit `OAI-5G/tools/fbs_scenarios/fake_gnb.cfg` to customize:
- PLMN/PCI spoofing
- TX power levels
- Cipher algorithms
- Attack behavior

### LLM Configuration

Create `llm_fbs_utils/config.yaml`:

```yaml
llm:
  model: gpt-3.5-turbo
  temperature: 0.7

apis:
  auditor: http://localhost:8090
  expert: http://localhost:8091

experiment:
  max_duration: 300
  safety_checks: true
```

## Testing

### Unit Tests

```bash
# Test individual components
pytest eval_scripts/test_query_api.py
pytest eval_scripts/test_rule_reload.py
```

### Integration Tests

```bash
# End-to-end test
bash eval_scripts/test_fbs_scenario.sh
```

### Performance Tests

```bash
# Run evaluation pipeline
python eval_scripts/evaluate_pipeline.py --output-dir results
```

## Troubleshooting

### Common Issues

1. **Services not starting**: Check Docker logs
   ```bash
   docker-compose logs -f mobiflow-auditor
   ```

2. **LLM API errors**: Verify API key
   ```bash
   echo $OPENAI_API_KEY
   ```

3. **No detections**: Check rule loading
   ```bash
   curl http://localhost:8091/rules
   ```

4. **Performance issues**: Monitor Redis
   ```bash
   redis-cli monitor
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Make modifications (keep patches minimal)
4. Test thoroughly
5. Submit pull request

## Security Considerations

- This system is for research/testing only
- Never run FBS simulations on production networks
- Use isolated test environments
- Follow responsible disclosure for any vulnerabilities found

## License

Components have different licenses:
- MobiFlow-Auditor: Apache 2.0
- MobieXpert: Apache 2.0
- OAI-5G: OAI Public License
- LLM Utils: MIT

## References

- [O-RAN Alliance Specifications](https://www.o-ran.org/)
- [OpenAirInterface Documentation](https://gitlab.eurecom.fr/oai/openairinterface5g/-/wikis/home)
- [5GSEC Project](https://github.com/5GSEC)
- [DSPy Framework](https://github.com/stanfordnlp/dspy)

## Contact

For questions or issues, please open a GitHub issue in the respective repository.

## Acknowledgments

This project builds upon the excellent work of:
- The 5GSEC team for MobiFlow-Auditor and MobieXpert
- The OpenAirInterface Software Alliance
- The O-RAN Alliance
- The DSPy team at Stanford

---

**Note**: This is a research prototype. Use responsibly and ethically.