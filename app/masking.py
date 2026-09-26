"""Narrow masking of assessment-required identifiers before classification."""

import re

_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+(?![A-Za-z0-9_%+-])"
)
_CNPJ = re.compile(
    r"(?<![A-Za-z0-9])(?:[0-9]{2}\.[0-9]{3}\.[0-9]{3}/[0-9]{4}-[0-9]{2}"
    r"|[0-9]{14})(?![A-Za-z0-9])"
)
_CNPJ_CONTEXT = re.compile(r"\bCNPJ\s*[:#-]?\s*$", re.IGNORECASE)
_RFC = re.compile(
    r"(?<![A-Za-z0-9])[A-ZÑ&]{3,4}[- ]?[0-9]{2}"
    r"(?:0[1-9]|1[0-2])(?:0[1-9]|[12][0-9]|3[01])[- ]?[A-Z0-9]{3}"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_COUNTRY_PHONE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"\+52[ .-]?(?:1[ .-]?)?(?:"
    r"(?:\([0-9]{2}\)|[0-9]{2})[ .-]?[0-9]{4}[ .-]?[0-9]{4}"
    r"|(?:\([0-9]{3}\)|[0-9]{3})[ .-]?[0-9]{3}[ .-]?[0-9]{4}"
    r")"
    r"|\+55[ .-]?(?:\([0-9]{2}\)|[0-9]{2})[ .-]?(?:"
    r"9[ .-]?[0-9]{4}[ .-]?[0-9]{4}"
    r"|[0-9]{4}[ .-]?[0-9]{4}"
    r")"
    r")(?![A-Za-z0-9])"
)
_LOCAL_PHONE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"(?:\([0-9]{2}\)|[0-9]{2})[ .-]*9[ .-]?[0-9]{4}[ .-]?[0-9]{4}"
    r"|(?:\([0-9]{2}\)|[0-9]{2})[ .-]*[0-9]{4}[ .-]+[0-9]{4}"
    r"|(?:\([0-9]{3}\)|[0-9]{3})[ .-]+[0-9]{3}[ .-]+[0-9]{4}"
    r")(?![A-Za-z0-9])"
)
_BARE_PHONE = re.compile(r"(?<![A-Za-z0-9])[0-9]{10,11}(?![A-Za-z0-9])")
_PHONE_CONTEXT = re.compile(
    r"\b(?:tel(?:e(?:fone|fono)|éfono)?|celular|m[oó]vil|whats?app|phone|fone|contato|contacto)"
    r"\s*[:#-]?\s*$",
    re.IGNORECASE,
)
_REFERENCE_CONTEXT = re.compile(
    r"\b(?:ord(?:er|en)?|pedido|ref(?:erencia)?|id)\s*[:#-]?\s*$", re.IGNORECASE
)


def _valid_cnpj(value: str) -> bool:
    """Use check digits to distinguish bare CNPJ from arbitrary 14-digit references."""
    digits = [int(character) for character in value if character.isdigit()]
    if len(digits) != 14 or len(set(digits)) == 1:
        return False
    for length, expected in ((12, digits[12]), (13, digits[13])):
        weights = [((length - index - 1) % 8) + 2 for index in range(length)]
        remainder = (
            sum(digit * weight for digit, weight in zip(digits[:length], weights, strict=True)) % 11
        )
        if (0 if remainder < 2 else 11 - remainder) != expected:
            return False
    return True


def _mask_cnpj(match: re.Match[str]) -> str:
    value = match.group()
    preceding_text = match.string[max(0, match.start() - 24) : match.start()]
    if "." in value or _valid_cnpj(value) or _CNPJ_CONTEXT.search(preceding_text):
        return "[CNPJ]"
    return value


def _mask_local_phone(match: re.Match[str]) -> str:
    if _REFERENCE_CONTEXT.search(match.string[max(0, match.start() - 24) : match.start()]):
        return match.group()
    return "[PHONE]"


def _mask_bare_phone(match: re.Match[str]) -> str:
    if _PHONE_CONTEXT.search(match.string[max(0, match.start() - 24) : match.start()]):
        return "[PHONE]"
    return match.group()


def mask_sensitive_data(message: str) -> str:
    """Replace common email, CNPJ, RFC, and BR/MX phone forms without changing other text."""
    masked = _EMAIL.sub("[EMAIL]", message)
    masked = _CNPJ.sub(_mask_cnpj, masked)
    masked = _RFC.sub("[RFC]", masked)
    masked = _COUNTRY_PHONE.sub("[PHONE]", masked)
    masked = _LOCAL_PHONE.sub(_mask_local_phone, masked)
    return _BARE_PHONE.sub(_mask_bare_phone, masked)
