def pytest_ignore_collect(collection_path, config):
    """Skip *importing* the Docker integration module unless integration tests are
    explicitly requested (`pytest -m integration`).

    The module probes the Docker daemon at import time; not collecting it on the
    default run keeps the unit suite from touching Docker at all, so a slow or
    unreachable daemon can never delay collection.
    """
    if collection_path.name == "test_docker_session.py":
        markexpr = (config.getoption("markexpr") or "").strip()
        return markexpr != "integration"
    return False
