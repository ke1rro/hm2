import argparse
import logging
import sys
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Authorize user-owned Drive files locally and save a private refreshable Kaggle secret."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-json", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path.home() / ".config/hm2/drive_token.json")
    arguments = parser.parse_args()
    flow = InstalledAppFlow.from_client_secrets_file(
        str(arguments.client_json), scopes=["https://www.googleapis.com/auth/drive"]
    )
    credentials = flow.run_local_server(
        port=0, access_type="offline", prompt="consent", authorization_prompt_message=""
    )
    if not credentials.refresh_token:
        raise RuntimeError(
            "Google did not issue a refresh token; repeat consent with offline access."
        )
    destination = arguments.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.touch(mode=0o600, exist_ok=True)
    destination.chmod(0o600)
    destination.write_text(credentials.to_json())
    LOGGER.info(
        "Saved private credentials to %s. Copy its contents to Kaggle Secret HM2_DRIVE_TOKEN.",
        destination,
    )


if __name__ == "__main__":
    main()
