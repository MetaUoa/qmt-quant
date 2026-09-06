from __future__ import annotations

import run_v5_c_nested_research as c1

from qmt_quant.research_runtime import install_v5_c_contracts


def main() -> int:
    install_v5_c_contracts(c1)
    return c1.main()


if __name__ == "__main__":
    raise SystemExit(main())
