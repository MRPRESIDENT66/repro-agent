"""Named task hooks used only when a manifest cannot express the behavior."""

from collections.abc import Callable

from evals.manifest import OracleHooks, TaskManifest

HookProvider = Callable[[TaskManifest], OracleHooks]


def binding_for(manifest: TaskManifest) -> OracleHooks:
    if manifest.hook is None:
        return OracleHooks()

    from evals.hooks.detectors import binding as detectors
    from evals.hooks.distilbert import binding as distilbert

    providers: dict[str, HookProvider] = {
        "detectors": detectors,
        "distilbert": distilbert,
    }
    try:
        provider = providers[manifest.hook]
    except KeyError as exc:
        raise ValueError(f"unknown manifest hook {manifest.hook!r}") from exc
    return provider(manifest)
