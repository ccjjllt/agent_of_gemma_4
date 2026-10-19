class PackageDAG(Mapping):
    def from_pkgs(cls, pkgs):
        pkgs = [DistPackage(p) for p in pkgs]
        idx = {p.key: p for p in pkgs}
        m = {p: [ReqPackage(r, idx.get(r.key)) for r in p.requires()] for p in pkgs}
        return cls(m)

    def __init__(self, m) -> None: