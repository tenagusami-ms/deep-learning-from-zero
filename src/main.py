"""
main
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from src.modules.lower_layer_modules.Exceptions import Error
from src.sections.section5_main import backprop


async def main() -> None:
    """
    main
    """
    prefix_directory: Path = Path(__file__).parent.parent
    data_directory: Path = prefix_directory / "data"
    # setting_directory: Path = prefix_directory / "settings"
    try:
        # gates()
        # call_mnist()
        # make_sample_prediction(data_directory / "MNIST" / "sample_weight.pkl")
        # mini_batch_learning()
        # gradient_check()
        backprop()

    except KeyboardInterrupt:
        exit(1)
    except Error as error:
        print(error.args[0])
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
