"""
main
"""
from __future__ import annotations

import asyncio
from pathlib import Path


async def main() -> None:
    """
    main
    """
    prefix_directory: Path = Path(__file__).parent.parent
    setting_directory: Path = prefix_directory / "settings"
    try:
        pass
    except KeyboardInterrupt:
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
