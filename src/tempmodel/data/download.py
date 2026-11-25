"""Command line entry point: download the daily temperature dataset."""

import logging
import sys

import hydra
from omegaconf import DictConfig
from pydantic import ValidationError

from tempmodel.config import Settings
from tempmodel.data.dataset import DataQualityError, get_daily_temperature, save_csv
from tempmodel.data.openmeteo import OpenMeteoError

logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3", config_path="pkg://tempmodel.conf", config_name="config"
)
def main(config: DictConfig) -> None:
    """Download the dataset described by the configuration."""
    try:
        settings = Settings.from_hydra(config)
        data = get_daily_temperature(settings)
        path = settings.output.path.absolute()
        save_csv(data, path)
    except ValidationError as error:
        logger.error("Invalid configuration: %s", error)
        sys.exit(1)
    except (OpenMeteoError, DataQualityError, OSError) as error:
        logger.error("Download failed: %s", error)
        sys.exit(1)

    logger.info(
        "Saved %d days (%s to %s) to %s",
        len(data),
        data.index[0].date(),
        data.index[-1].date(),
        path,
    )


if __name__ == "__main__":
    main()
