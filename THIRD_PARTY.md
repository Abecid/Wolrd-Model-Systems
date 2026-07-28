# Third-party model integrations

No upstream source code or model weights are vendored in this repository. Bootstrap scripts clone reviewed revisions into the gitignored `third_party/` directory.

| Adapter | Upstream | Pinned revision | Upstream terms |
|---|---|---|---|
| `matrix-game-3` | `SkyworkAI/Matrix-Game` | `71c3cd7f741311f8100f6cf9cde942b6c1378d11` | Matrix-Game 3 subproject: Apache-2.0; weights remain subject to upstream terms |
| `lingbot-world-v2` | `Robbyant/lingbot-world-v2` | `2648877f763a06cc743bcd919936da4d25f12e7b` | CC BY-NC-SA 4.0; non-commercial/share-alike |
| `hy-worldplay-1.5` | `Tencent-Hunyuan/HY-WorldPlay` | `1588e1336e842b03b0a7860c654ebd7c46bb065e` | Custom Tencent community license with territorial and use restrictions |

Optional integration patches insert imports and replace narrowly scoped expressions. They do not change ownership or licensing of upstream files. Review upstream source and weight licenses before use, redistribution or deployment.
