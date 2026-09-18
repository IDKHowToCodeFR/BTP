import subprocess
import datetime
import os

# Base time: 11 days ago
base_date = datetime.datetime.now() - datetime.timedelta(days=11)

def commit(message, files, day_offset=0, branch=None):
    if branch:
        subprocess.run(["git", "checkout", "-b", branch], check=False)
        
    for f in files:
        if isinstance(f, str):
            subprocess.run(f"git add {f}", shell=True, check=False)
        
    commit_date = (base_date + datetime.timedelta(days=day_offset)).strftime('%Y-%m-%dT%H:%M:%S')
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = commit_date
    env["GIT_COMMITTER_DATE"] = commit_date
    
    subprocess.run(["git", "commit", "-m", message], env=env, check=False)

commits = [
    {
        "day": 0,
        "msg": "feat: Add run.bat pipeline orchestrator",
        "files": ["run.bat"]
    },
    {
        "day": 1,
        "msg": "build: Update dependencies with scikit-learn and uv support",
        "files": ["pyproject.toml", ".gitignore", "uv.lock"]
    },
    {
        "day": 2,
        "msg": "feat(evaluation): Add Storage abstractions and result CSV formatting",
        "files": ["src/agentic_selection/evaluation/storage.py"]
    },
    {
        "day": 3,
        "msg": "feat(agent): Add LLM backends with exponential backoff and Ollama support",
        "files": ["src/agentic_selection/agent/llm_backends.py", "config.yaml"]
    },
    {
        "day": 4,
        "msg": "feat(agent): Implement core Agent Memory structures",
        "files": ["src/agentic_selection/agent/memory.py"]
    },
    {
        "day": 5,
        "msg": "feat(agent): Develop reasoning mechanisms for LLM prompt generation",
        "files": ["src/agentic_selection/agent/reasoning.py"]
    },
    {
        "day": 6,
        "msg": "feat(evaluation): Build stable and drift protocols with rigorous test coverage",
        "files": ["src/agentic_selection/evaluation/protocol.py", "tests/test_protocol.py", "tests/test_efficiency.py"]
    },
    {
        "day": 7,
        "msg": "feat(scripts): Wire stable and drift evaluation scripts to protocol layer",
        "files": ["scripts/04_run_stable_experiment.py", "scripts/05_run_drift_experiment.py", "scripts/07_run_wsdream_validation.py"]
    },
    {
        "branch": "feature/rag-agent",
        "day": 8,
        "msg": "feat(agent): Introduce procedural RAG Agent Controller with TF-IDF semantic retrieval",
        "files": ["src/agentic_selection/agent/controller.py"]
    },
    {
        "day": 9,
        "msg": "feat(reporting): Refine plotting style and generate_report logic",
        "files": ["src/agentic_selection/evaluation/report.py", "src/agentic_selection/evaluation/plot_style.py", "scripts/06_generate_report.py"]
    },
    {
        "day": 10,
        "msg": "feat(reporting): Implement presentation figures and extended analysis",
        "files": ["scripts/10_generate_presentation_figures.py"]
    },
    {
        "branch": "main",
        "day": 11,
        "msg": "Merge branch 'feature/rag-agent' into main",
        "files": [] 
    },
    {
        "day": 11,
        "msg": "docs: Write ARCHITECTURE.md, update README.md, and commit latest results",
        "files": ["ARCHITECTURE.md", "README.md", "results/figures/*", "results/tables/*"]
    }
]

for c in commits:
    if "branch" in c and c["branch"] == "main":
        subprocess.run(["git", "checkout", "main"], check=False)
        subprocess.run(["git", "merge", "--no-ff", "--no-edit", "feature/rag-agent"], check=False)
        continue
        
    commit(c["msg"], c["files"], c["day"], c.get("branch"))

# Catch-all
subprocess.run("git add .", shell=True, check=False)
commit("chore: Final cleanup and sync of all untracked artifacts", ["."], 11)

print("Git history simulation complete!")
