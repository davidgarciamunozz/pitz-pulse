"""Local, statistical acceptance policy for newly generated Spanish summaries."""

from importlib.metadata import version

from langdetect import DetectorFactory, LangDetectException, detect

POLICY_VERSION = "spanish-summary-v1"
DETECTOR_PACKAGE = "langdetect"
DETECTOR_VERSION = version(DETECTOR_PACKAGE)

# langdetect shares its loaded profiles and otherwise randomizes short-text results.
DetectorFactory.seed = 0


def accepts_spanish_summary(summary: str) -> bool:
    """Accept only a Spanish top verdict; abstain on undetectable text."""
    try:
        return detect(summary) == "es"
    except LangDetectException:
        return False
