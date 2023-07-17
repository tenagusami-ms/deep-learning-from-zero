"""
main
"""
from __future__ import annotations

import asyncio

from src.modules.lower_layer_modules.Exceptions import Error
from src.sections.section3_mnist import call_mnist


async def main() -> None:
    """
    main
    """
    # prefix_directory: Path = Path(__file__).parent.parent
    # setting_directory: Path = prefix_directory / "settings"
    try:
        # gates()
        call_mnist()

    except KeyboardInterrupt:
        exit(1)
    except Error as error:
        print(error.args[0])
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
