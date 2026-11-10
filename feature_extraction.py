
import re
import math
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "verify", "account", "secure", "update",
    "signin", "password", "confirm", "banking",
    "wallet", "auth", "recover", "webscr", "support"
]

SHORTENERS = [
    "bit.ly", "tinyurl.com", "t.co", "goo.gl",
    "ow.ly", "is.gd", "buff.ly", "cutt.ly"
]

COMMON_TLDS = {
    "com", "org", "net", "edu", "gov", "in",
    "uk", "de", "fr", "io", "co", "au"
}


def calculate_entropy(text):
    if not text:
        return 0

    probabilities = [
        text.count(char) / len(text)
        for char in set(text)
    ]

    return -sum(p * math.log2(p) for p in probabilities)


def extract_features(url):

    url = str(url).strip()

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    host_parts = hostname.split(".")
    tld = host_parts[-1] if host_parts else ""

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),

        "count_dots": url.count("."),
        "count_hyphens": url.count("-"),
        "count_at": url.count("@"),
        "count_question": url.count("?"),
        "count_equal": url.count("="),
        "count_digits": sum(c.isdigit() for c in url),
        "count_slash": url.count("/"),

        "https": int(parsed.scheme == "https"),

        "ip_address": int(bool(re.match(
            r"^\d{1,3}(\.\d{1,3}){3}$", hostname
        ))),

        "subdomain_count": max(0, len(host_parts) - 2),

        "suspicious_word_count": sum(
            word in url.lower()
            for word in SUSPICIOUS_WORDS
        ),

        "shortener": int(any(
            hostname == domain or hostname.endswith("." + domain)
            for domain in SHORTENERS
        )),

        "punycode": int("xn--" in hostname.lower()),

        "unusual_tld": int(
            bool(tld) and tld.lower() not in COMMON_TLDS
        ),

        "hostname_has_digits": int(
            any(c.isdigit() for c in hostname)
        ),

        "hostname_entropy": calculate_entropy(hostname),

        "url_entropy": calculate_entropy(url),

        "double_slash_in_path": int("//" in path),

        "query_present": int(bool(query)),

        "suspicious_extension": int(
            bool(re.search(
                r"\.(exe|zip|scr|bat|apk|php)$",
                path.lower()
            ))
        )
    }

    return features
