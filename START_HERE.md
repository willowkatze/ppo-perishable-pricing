# Start Here

This project studies dynamic markdown policies for perishable fresh-retail inventory using FreshRetailNet-informed demand recovery and a semi-synthetic pricing environment.

The strongest overall held-out policy was always_0pct.

The strongest learned held-out policy was balanced_recovered_ppo.

Balanced recovered PPO outperformed all evaluated positive fixed-markdown policies, random_uniform, two simple rule-based policies, the original PPO variants, and the locked DQN ensemble. However, it did not beat always_0pct. The result is therefore a useful learned-policy improvement over weaker alternatives, not proof that learned markdowns beat the strongest no-markdown baseline.

Recommended reading order:

1. [README.md](README.md)
2. [docs/01_project_and_research_design.md](docs/01_project_and_research_design.md)
3. [docs/03_environment_and_methods.md](docs/03_environment_and_methods.md)
4. [docs/04_results.md](docs/04_results.md)
5. [docs/05_conclusions_and_limitations.md](docs/05_conclusions_and_limitations.md)
6. [results/README.md](results/README.md)

