# Hill Fire Detection Sim

This repository is the planned Python simulation environment for the Hill Fire Detection project. Its role is to provide repeatable scenarios for evaluating early hill-fire detection and fire-perimeter tracking.

## Project scope

- Define simulation scenarios and configuration for repeatable runs.
- Produce simulation observations and reference outputs for evaluation.
- Document the data exchanged with the companion [Hill Fire Detection Model](https://github.com/CUHK-ROSE5770/Hill-Fire-Detection-Model) repository as the interface is implemented.

## Repository status

This repository is at the initial documentation stage. The simulator, dependency manifest, and runnable examples are planned. Setup instructions and a smoke test can be added when the first runnable implementation lands.

## Contributing

Open a pull request using the [PR template](.github/pull_request_template.md). Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages and PR titles in the form `type(scope): description`. The scope is optional.

```text
docs: describe the simulation environment
feat(sim): add scenario configuration
fix(sim): preserve the configured random seed
```

Record the checks you ran and their results. For scenario or data-format changes, include the reproduction details and link any corresponding model change.

## License

[Apache License 2.0](LICENSE).
