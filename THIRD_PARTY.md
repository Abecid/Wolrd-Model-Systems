# Third-party components

No third-party model source or weights are vendored in this repository.
`scripts/bootstrap_upstream.sh` clones the official Matrix-Game repository and
checks out a pinned commit into a gitignored directory.

Primary integration target:

- SkyworkAI/Matrix-Game, Matrix-Game-3, pinned commit
  `71c3cd7f741311f8100f6cf9cde942b6c1378d11`.

The optional integration patch inserts imports and call sites for this package;
it does not change ownership of upstream files. Review upstream license and
weight terms before use or redistribution.
