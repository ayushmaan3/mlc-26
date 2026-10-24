"""Stage 1: normalize names/addresses (country-agnostic rules; France unseen in train).
Native scripts are transliterated to ASCII (anyascii) first, so Devanagari/Odia/Gujarati names compare to English.
Out per split: id, src, country, name_n, core, nosp, skel, addr_n, nums(list), atoks(list), stoks(list)."""
import re

import polars as pl
from anyascii import anyascii

from common import SPLITS, WORK, done, timer

# ponytail: hand-written maps, extend from error analysis
LEGAL_MAP = {"private": "pvt", "pvt": "pvt", "limited": "ltd", "corporation": "corp", "company": "co",
             "incorporated": "inc", "compagnie": "cie", "praivet": "pvt", "praivett": "pvt", "pra": "pvt",
             "prai": "pvt", "li": "ltd", "limitedd": "ltd", "limited": "ltd", "limitedda": "ltd", "elelpi": "llp",
             "enterprises": "ent", "enterprise": "ent", "services": "svc", "service": "svc",
             "industries": "ind", "industry": "ind", "technologies": "tech", "technology": "tech",
             "international": "intl", "associates": "assoc", "brothers": "bros"}
LEGAL = {"pvt", "ltd", "llc", "llp", "inc", "corp", "co", "cie", "sarl", "sas", "sasu", "eurl", "sa", "sci",
         "ei", "pllc", "lp", "plc", "the", "and", "et", "of", "smt", "shri", "sri", "m", "s", "ms", "dba", ""}
STREET = {"street": "st", "road": "rd", "lane": "ln", "avenue": "ave", "av": "ave", "drive": "dr",
          "court": "ct", "circle": "cir", "boulevard": "blvd", "bd": "blvd", "bld": "blvd", "place": "pl",
          "highway": "hwy", "parkway": "pkwy", "terrace": "ter", "trail": "trl", "square": "sq",
          "rue": "r", "allee": "all", "chemin": "ch", "impasse": "imp", "route": "rte", "north": "n",
          "south": "s", "east": "e", "west": "w", "calcutta": "kolkata", "bombay": "mumbai",
          "bangalore": "bengaluru", "madras": "chennai", "gurgaon": "gurugram", "null": "", "na": "",
          "none": "", "no": "", "bis": "", "nd": "", "th": "", "rd_": "", "st_": "", "first": "1", "second": "2",
          "third": "3", "fourth": "4", "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9",
          "tenth": "10", "saint": "st", "mount": "mt", "fort": "ft", "nagara": "nagar"}
US_STATES = {"alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
             "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
             "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
             "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
             "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
             "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
             "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
             "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
             "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
             "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
             "district of columbia": "dc"}
IN_STATES = {"andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
             "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
             "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh",
             "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od",
             "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
             "telangana": "tg", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
             "west bengal": "wb", "delhi": "dl", "new delhi": "dl", "jammu and kashmir": "jk",
             "chandigarh": "ch", "puducherry": "py", "pondicherry": "py"}
PHRASES = {f" {k} ": f" {v} " for k, v in {**US_STATES, **IN_STATES}.items() if " " in k}
ASTOP = ["floor", "ground", "first", "second", "unit", "apartment", "suite", "city", "near", "plot",
         "sector", "shop", "site", "behind", "opposite", "building", "main", "cross", "nagar", "village",
         "town", "estate", "colony", "house", "office", "door", "flat", "block", "road", "street"]
WORDS = {**STREET, **{k: v for k, v in {**US_STATES, **IN_STATES}.items() if " " not in k}}


ANUSVARA = re.compile(r"m(?=[^aeiouybpm\W])")  # anyascii writes anusvara as 'm': kmsltemts -> knsltents


def _tr(x):
    return ANUSVARA.sub("n", anyascii(x).lower()) if x and not x.isascii() else x


def translit(c):
    """ASCII-fold everything (Indic scripts, accents). ASCII strings pass through untouched."""
    return c.map_batches(lambda s: pl.Series([_tr(x) for x in s.to_list()], dtype=pl.Utf8), return_dtype=pl.Utf8)


def base(c):
    return translit(c).str.to_lowercase().str.replace_all("&", " and ", literal=True)


def tokens(c):
    return c.str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars().str.split(" ")


def skel_expr(c):
    """Phonetic consonant skeleton per token (handles translit vowel loss: laxmi/lksmi -> lksm)."""
    return (c.str.replace_all("x", "ks").str.replace_all("ph", "f").str.replace_all("q", "k")
            .str.replace_all("c", "k").str.replace_all("w", "v").str.replace_all("z", "j")
            .str.replace_all("g", "j").str.replace_all(r"\B[aeiouyh]+", "").str.replace_many([c + c for c in "bdfjklmnprstv"], list("bdfjklmnprstv"))
            .str.replace_many([c + c for c in "bdfjklmnprstv"], list("bdfjklmnprstv")))


def normalize(df):
    name = (base(pl.col("name"))
            .str.replace_all(r"\(\s*id:?\s*\d+\s*\)|\[?#\s*\d+\]?|\+?\d[\d\- ]{8,}\d", " ")  # ids, #123, phones
            .str.replace_all(r"https?://|www\.|\.(com|in|net|org|co|fr|biz|io|us)\b|^@", " ")
            .str.replace_all(r"([a-z])\.([a-z])\.", "$1$2")   # l.l.p. -> llp
            .str.replace_all(r"([a-z])\.([a-z])\.", "$1$2")
            .str.replace_all(".", "", literal=True).str.replace_all("'", "", literal=True))
    ntok = tokens(name).list.eval(pl.element().replace(LEGAL_MAP))
    ntok = ntok.list.eval(pl.element().filter(pl.element() != ""))
    core_l = ntok.list.eval(pl.element().filter(~pl.element().is_in(list(LEGAL))))
    addr = (" " + base(pl.col("addr")).str.replace_all(r"[^a-z0-9]+", " ") + " ")
    addr = addr.str.replace_many(list(PHRASES), list(PHRASES.values()))
    atok = tokens(addr).list.eval(pl.element().replace(WORDS)).list.eval(
        pl.element().filter(pl.element() != ""))
    return df.select(
        "id", "src", "country",
        ntok.list.join(" ").alias("name_n"),
        core_l.list.unique().list.sort().list.join(" ").alias("core"),
        core_l.list.join("").alias("nosp"),
        core_l.list.eval(skel_expr(pl.element())).list.unique().list.sort().list.join(" ").alias("skel"),
        atok.list.join(" ").alias("addr_n"),
        addr.str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0"))
            .list.eval(pl.element().filter(pl.element() != "")).list.unique(maintain_order=True).list.head(5).alias("nums"),
        atok.list.eval(pl.element().filter(pl.element().str.contains(r"^[a-z]{4,}$") & ~pl.element().is_in(ASTOP)))
            .list.unique().list.head(6).alias("atoks"),
        core_l.list.eval(skel_expr(pl.element())).list.eval(pl.element().filter(pl.element().str.len_chars() >= 3))
            .list.unique().alias("stoks"),
    )


def main():
    for split in SPLITS:
        out = WORK / f"{split}_norm2.parquet"
        if done(out):
            continue
        with timer(f"normalize {split}"):
            normalize(pl.scan_parquet(WORK / f"{split}_records.parquet")).sink_parquet(out)


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_rows(30)
        d = pl.read_parquet(WORK / "train_records.parquet", n_rows=2_000_000).sample(25, seed=int(sys.argv[-1]))
        print(pl.concat([d.select("name", "addr"), normalize(d.lazy()).collect()
                         .select("core", "skel", "nosp", "addr_n", "nums")], how="horizontal"))
    else:
        main()
