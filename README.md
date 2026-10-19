# DHH-SQL

Research implementation of Dynamic Schema Linking via Heterogeneous Hypergraphs
for Text-to-SQL.

## Public release scope

This is a partial research release. The original training recipe, numerical
hyperparameter defaults, and configuration-bearing regression fixtures are not
distributed. This repository does not provide a complete recipe for reproducing
the reported results.

The source retains heterogeneous schema hypergraphs, dynamic question-schema
linking, and relation-aware graph encoding. Numerical experiment settings must
be supplied explicitly by an authorized user; no replacement best settings are
provided. See the CLI argument declarations for available options.

## Data and environment

Install dependencies from `requirement.txt`. Obtain Spider and required language
resources separately. Datasets, processed graphs, model weights, private launch
scripts, logs, and experiment configurations are excluded from the public release.

Preprocessing and training must use consistent hypergraph metadata. Checkpoint
compatibility validation is retained.

## Tests

The remaining tests use small synthetic inputs to verify graph behavior. Their
dimensions and tensor values are not experiment configurations. Tests containing
withheld method settings are omitted from this public release.

## Handling private artifacts

Local execution still writes logs and checkpoints needed for research. Keep those
artifacts private: they can contain settings, optimizer state, and model metadata.
Ignore rules help prevent accidental tracking but do not remove committed history.
Previously published revisions and third-party copies are not revoked by this
cleanup.
