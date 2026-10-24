"""Stage 1: normalize names/addresses (country-agnostic rules; France unseen in train).
Out per split: id, src, country, name_n, core, addr_n, nums(list), atoks(list)."""
import polars as pl

from common import SPLITS, WORK, done, timer

# ponytail: hand-written maps, extend from error analysis
LEGAL_MAP = {"private": "pvt", "pvt": "pvt", "limited": "ltd", "corporation": "corp", "company": "co",
             "incorporated": "inc", "compagnie": "cie", "प्राइवेट": "pvt", "लिमिटेड": "ltd",
             "एलएलपी": "llp", "प्रा": "pvt", "लि": "ltd", "প্রাইভেট": "pvt", "লিমিটেড": "ltd", "এলএলপি": "llp"}
LEGAL = {"pvt", "ltd", "llc", "llp", "inc", "corp", "co", "cie", "sarl", "sas", "sasu", "eurl", "sa", "sci",
         "ei", "pllc", "lp", "plc", "the", "and", "et", "of", ""}
STREET = {"street": "st", "road": "rd", "lane": "ln", "avenue": "ave", "av": "ave", "drive": "dr",
          "court": "ct", "circle": "cir", "boulevard": "blvd", "bd": "blvd", "bld": "blvd", "place": "pl",
          "highway": "hwy", "parkway": "pkwy", "terrace": "ter", "trail": "trl", "square": "sq",
          "rue": "r", "allee": "all", "chemin": "ch", "impasse": "imp", "route": "rte", "north": "n",
          "south": "s", "east": "e", "west": "w", "calcutta": "kolkata", "bombay": "mumbai",
          "bangalore": "bengaluru", "madras": "chennai", "null": "", "na": "", "none": "", "no": "",
          "bis": "", "nd": "", "th": ""}
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


def base(c):
    """lower, strip Latin accents (only U+0300-036F so Indic vowel signs survive), punctuation -> space."""
    return (c.str.to_lowercase().str.normalize("NFKD").str.replace_all(r"[̀-ͯ]", "")
            .str.replace_all("&", " and ", literal=True))


def tokens(c):
    return c.str.replace_all(r"[^\p{L}\p{N}\p{M}]+", " ").str.strip_chars().str.split(" ")


def normalize(df):
    name = (base(pl.col("name"))
            .str.replace_all(r"\(\s*id:?\s*\d+\s*\)|\[?#\s*\d+\]?|\+?\d[\d\- ]{8,}\d", " ")  # ids, #123, phones
            .str.replace_all(r"https?://|www\.|\.(com|in|net|org|co|fr|biz|io)\b", " ")
            .str.replace_all(r"(\p{L})\.(\p{L})\.", "$1$2")   # l.l.p. -> llp
            .str.replace_all(r"(\p{L})\.(\p{L})\.", "$1$2")
            .str.replace_all(".", "", literal=True))
    ntok = tokens(name).list.eval(pl.element().replace(LEGAL_MAP))
    addr = (" " + base(pl.col("addr")).str.replace_all(r"[^\p{L}\p{N}\p{M}]+", " ") + " ")
    addr = addr.str.replace_many(list(PHRASES), list(PHRASES.values()))
    atok = tokens(addr).list.eval(pl.element().replace(WORDS)).list.eval(
        pl.element().filter(pl.element() != ""))
    return df.select(
        "id", "src", "country",
        ntok.list.eval(pl.element().filter(pl.element() != "")).list.join(" ").alias("name_n"),
        ntok.list.eval(pl.element().filter(~pl.element().is_in(list(LEGAL)))).list.unique()
            .list.sort().list.join(" ").alias("core"),
        atok.list.join(" ").alias("addr_n"),
        pl.col("addr").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0"))
            .list.eval(pl.element().filter(pl.element() != "")).list.unique().list.head(4).alias("nums"),
        atok.list.eval(pl.element().filter(pl.element().str.contains(r"^[a-z]{4,}$") & ~pl.element().is_in(ASTOP))).list.unique()
            .list.head(6).alias("atoks"),
    )


def main():
    for split in SPLITS:
        out = WORK / f"{split}_norm.parquet"
        if done(out):
            continue
        with timer(f"normalize {split}"):
            normalize(pl.scan_parquet(WORK / f"{split}_records.parquet")).sink_parquet(out)


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:  # eyeball check on a few rows
        pl.Config.set_fmt_str_lengths(90); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(30)
        d = pl.read_parquet(WORK / "train_records.parquet", n_rows=300_000).sample(20, seed=3)
        print(pl.concat([d.select("name", "addr"), normalize(d.lazy()).collect()
                         .select("core", "addr_n", "nums", "atoks")], how="horizontal"))
    else:
        main()
